import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import math
import pickle as pkl
from tqdm import tqdm
import joblib
import fasttext
from transformers import CamembertForCausalLM, CamembertConfig, CamembertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from models.classes import BiLSTMEncoder, LSTMAttentionDecoder
from models.classes.Search import BeamSearch
from utils import print_tilde_to_csv
from utils.models import masked_softmax, embed_and_avg_attributes, sample_y_tilde, compute_bleu_score, \
    handle_fb_preds, get_metrics, decode_tokenized_words, get_metrics_at_k, beam_search_decoder, prettify_bleu_score, handle_svm_pred


class Rewriter(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, fine_tune_word_emb, vocab_size, alpha, beta, model_path, mode,
                 save_step, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.alpha = alpha
        self.beta = beta
        self.emb_dim = emb_size
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.num_ind = 147
        self.num_exp_level = 5
        self.mode = mode
        self.save_step = save_step

        if hp.tf != -1:
            self.tf = hp.tf

        with open(os.path.join(self.datadir, "vocab_dict.pkl"), "rb") as f:
            self.vocab_dict = pkl.load(f)
        self.rev_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
        # self.beam_search = BeamSearch(self.rev_vocab_dict)
        self.pad = self.rev_vocab_dict['<pad>']
        self.unk = self.rev_vocab_dict['<unk>']
        self.eos = self.rev_vocab_dict['</s>']
        self.vocab_size = len(self.rev_vocab_dict)

        self.exp_dict = {0: "novice",
                    1: "junior",
                    2: "intermediary",
                    3: "senior",
                    4: "expert"}
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

        self.word_embedder = torch.nn.Embedding(len(self.vocab_dict), emb_size)
        with open(os.path.join(self.datadir, "word_embs_dynamics.t"), "rb") as f:
            vocab_embeddings = torch.load(f)
        self.word_embedder.from_pretrained(vocab_embeddings, freeze=fine_tune_word_emb)

        with open(os.path.join(self.datadir, "embs_exp_ind.t"), "rb") as f:
            emb_attributes = torch.load(f)
        self.attribute_embedder_as_tokens = torch.nn.Embedding(self.num_ind + self.num_exp_level,
                                                               emb_attributes.shape[-1])
        self.attribute_embedder_as_tokens.from_pretrained(emb_attributes, freeze=False)
        self.attribute_embedder_as_bias = torch.nn.Embedding(self.num_ind + self.num_exp_level, self.voc_size)
        self.encoder = BiLSTMEncoder(self.emb_dim, self.hp.hidden_size, hp.pooling_window, hp)
        self.decoder = LSTMAttentionDecoder(self.hp.hidden_size, self.emb_dim, self.voc_size,
                                            hp)

    def forward(self, jobs, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices, fwd_type):
        if fwd_type == "DAE":
            attributes_as_first_token = embed_and_avg_attributes(exp_indices,
                                                                 ind_indices,
                                                                 self.attribute_embedder_as_tokens.weight.shape[-1],
                                                                 self.attribute_embedder_as_tokens)
            attributes_as_bias = embed_and_avg_attributes(exp_indices,
                                                          ind_indices,
                                                          self.voc_size,
                                                          self.attribute_embedder_as_bias)
            jobs_corrupted = self.add_noise(jobs, masks, self.max_len)

            jobs_corrupted_embedded = self.word_embedder(jobs_corrupted)
            encoder_outputs, enc_hidden = self.encoder.forward(jobs_corrupted_embedded)

            # labels
            jobs_embedded = self.word_embedder(jobs)
            jobs_embedded[:, 0, :] = attributes_as_first_token[:, 0, :]

            resized_conditioning = self.decoder.lin_att_in_para(jobs_corrupted_embedded[:, :-1, :])
            tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, masks, self.max_len)
            attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs, attn_weights)
            decoder_inputs = torch.cat((jobs_embedded[:, :-1, :], attn_applied), -1)
            decoder_output = self.decoder.forward(decoder_inputs, attributes_as_bias)

            loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                     jobs[:, 1:], reduction="sum", ignore_index=1)
            if torch.isnan(loss):
                ipdb.set_trace()
            return loss

        elif fwd_type == "BT":
            sample_len = len(jobs)

            atts = self.get_y_and_y_tilde(exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
            atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[
                2], atts[3]
            jobs_embedded = self.word_embedder(jobs)
            encoder_outputs, enc_hidden = self.encoder.forward(jobs_embedded)

            previous_token = atts_tilde_as_first_token
            decoded_tokens = []
            for di in range(self.max_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
                attn_weights = masked_softmax(tmp, masks, 1)
                assert attn_weights.shape == torch.Size([sample_len, self.max_len, 1])
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)

                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

                output_lstm, _ = self.decoder.LSTM(output)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = output_lin + atts_as_bias
                # you cannot teacher force it, you don't have the label
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                # prev_hidden = hidden
                previous_token = self.word_embedder(decoder_tok)

            sos = torch.stack(
                [(torch.ones(sample_len, 1).type(torch.LongTensor) * 5).type_as(decoded_tokens[0])]).view(
                self.hp.b_size, 1, 1)
            tmp = torch.stack(decoded_tokens).view(self.hp.b_size, 1, self.max_len - 1)
            decoded_x_tilde = torch.cat([sos, tmp], dim=-1).squeeze(1).type_as(decoded_tokens[0])
            # computing new masks
            tmp = (decoded_x_tilde == torch.ones(decoded_x_tilde.shape[0], decoded_x_tilde.shape[1]).type_as(
                decoded_x_tilde)).type(torch.cuda.BoolTensor)
            new_masks = ~tmp

            embedded_x_tiled = self.word_embedder(decoded_x_tilde)
            encoder_outputs_2, enc_hidden_2 = self.encoder.forward(embedded_x_tiled)
            # labels
            jobs_embedded = self.word_embedder(jobs)
            jobs_embedded[:, 0, :] = atts_as_first_token[:, 0, :]

            resized_conditioning = self.decoder.lin_att_in_para(jobs_embedded[:, :-1, :])
            tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, new_masks, self.max_len)
            assert attn_weights.shape == torch.Size([sample_len, self.max_len, self.max_len - 1])
            attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs_2, attn_weights)
            decoder_inputs = torch.cat((jobs_embedded[:, :-1, :], attn_applied), -1)
            decoder_output = self.decoder.forward(decoder_inputs, atts_as_bias)

            loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                     jobs[:, 1:], reduction="sum", ignore_index=1)
            if torch.isnan(loss):
                ipdb.set_trace()
            return loss
        else:
            raise Exception("wrong forward type, can be either \"DAE\" or \"BT\", " + str(fwd_type) + " was given.")

    def inference(self, jobs, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices):
        if self.hp.debug_noiseless == "True":
            jobs_embedded = self.word_embedder(jobs)
            encoder_outputs, _ = self.encoder.forward(jobs_embedded)
            previous_token = self.word_embedder(torch.ones(1, 1).type(torch.LongTensor).cuda() * 5)
            decoded_tokens, posteriors = [], []
            for di in range(self.max_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1).type_as(encoder_outputs))
                attn_weights = masked_softmax(tmp, masks, 1)
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)
                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)
                output_lstm, hidden = self.decoder.LSTM(output)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = torch.softmax(output_lin, dim=-1)
                posteriors.append(decoder_output)
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.word_embedder(decoder_tok)
            resized_preds = torch.stack(posteriors).view(1, self.max_len - 1, self.voc_size)
            indices, log_prob = beam_search_decoder(resized_preds, 3)
            selected_seq = indices[0, torch.argmax(log_prob)]
            return selected_seq.unsqueeze(0)
            # return torch.stack(decoded_tokens).squeeze(-1).view(1, -1)
        else:
            atts = self.get_y_and_y_tilde(exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
            atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[
                2], atts[3]
            jobs_embedded = self.word_embedder(jobs)
            encoder_outputs, enc_hidden = self.encoder.forward(jobs_embedded)
            previous_token = atts_tilde_as_first_token
            decoded_tokens, posteriors = [], []
            for di in range(self.max_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
                attn_weights = masked_softmax(tmp, masks, 1)
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)

                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

                output_lstm, hidden = self.decoder.LSTM(output)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = torch.softmax(output_lin + atts_as_bias, dim=-1)
                posteriors.append(output_lin + atts_as_bias)
                # you cannot teacher force it, you don't have the label
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.word_embedder(decoder_tok)
            if self.hp.beam_search == "True":
                resized_preds = torch.stack(posteriors).view(1, self.max_len - 1, self.voc_size)
                indices, log_prob = beam_search_decoder(torch.softmax(resized_preds, dim=-1), 3)
                if torch.isnan(log_prob).any():
                    ipdb.set_trace()
                selected_seq = indices[0, torch.argmax(log_prob)]
                return selected_seq.unsqueeze(0)
            else:
                return torch.stack(decoded_tokens).squeeze(-1).view(1, -1)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        sample_len = len(indices)
        if self.hp.att_type == "exp":
            ind_indices = None
        elif self.hp.att_type == "ind":
            exp_indices = None

        loss_dae_tmp = self.forward(indices, masks, exp_indices, ind_indices, None, None, 'DAE')
        loss_dae = loss_dae_tmp / sample_len

        exp_tilde_indices, ind_tilde_indices = sample_y_tilde(exp_indices, ind_indices, len(self.exp_dict), len(self.industry_dict), self.hp.no_tilde)
        loss_bt_tmp = self.forward(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices, 'BT')
        loss_bt = loss_bt_tmp / sample_len

        loss = self.alpha * loss_dae + self.beta * loss_bt

        self.log('train_loss_ep', loss, on_step=False, on_epoch=True)
        self.log('train_loss_st', loss, on_step=True, on_epoch=False)

        self.log('train_loss_dae_ep', loss_dae, on_step=False, on_epoch=True)
        self.log('train_loss_dae_st', loss_dae, on_step=True, on_epoch=False)

        self.log('train_loss_bt_ep', loss_bt, on_step=False, on_epoch=True)
        self.log('train_loss_bt_st', loss_bt, on_step=True, on_epoch=False)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].mean()}

    def training_epoch_end(self, training_step_outputs):
        tmp = torch.stack([i["loss"] for i in training_step_outputs]).mean()
        training_step_outputs = tmp

    def validation_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        sample_len = len(indices)
        if self.hp.att_type == "exp":
            ind_indices = None
        elif self.hp.att_type == "ind":
            exp_indices = None

        loss_dae_tmp = self.forward(indices, masks, exp_indices, ind_indices, None, None, 'DAE')
        loss_dae = loss_dae_tmp / sample_len

        exp_tilde_indices, ind_tilde_indices = sample_y_tilde(exp_indices, ind_indices, len(self.exp_dict), len(self.industry_dict), self.hp.no_tilde)
        loss_bt_tmp = self.forward(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices, 'BT')
        loss_bt = loss_bt_tmp / sample_len

        val_loss = self.alpha * loss_dae + self.beta * loss_bt

        self.log('valid_loss_ep', val_loss, on_step=False, on_epoch=True)
        self.log('valid_loss_dae_ep', loss_dae, on_step=False, on_epoch=True)
        self.log('valid_loss_bt_ep', loss_bt, on_step=False, on_epoch=True)

        self.log('val_loss', val_loss)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_epoch_outputs):
        tmp = torch.stack([i["val_loss"] for i in validation_epoch_outputs]).mean()
        validation_epoch_outputs = tmp
        return {'val_loss': tmp}

    def on_test_epoch_start(self):
        self.test_decoded_outputs = []
        self.test_decoded_labels = []
        config = CamembertConfig.from_pretrained("camembert-base")
        config.is_decoder = True
        self.ppl_model = CamembertForCausalLM.from_pretrained("camembert-base", config=config).cuda()
        self.tokenizer_fr = CamembertTokenizer.from_pretrained("camembert-base")
        self.exp_dict = {0: "novice",
                    1: "junior",
                    2: "intermediary",
                    3: "senior",
                    4: "expert"}
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

    def test_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        if self.hp.input_recopy == "True":
            self.test_decoded_labels.append((indices, exp_indices, ind_indices))
        else:
            exp_tilde_indices, ind_tilde_indices = sample_y_tilde(exp_indices, ind_indices, len(self.exp_dict), len(self.industry_dict), self.hp.no_tilde)
            decoded_outputs = self.inference(indices, masks, exp_indices, ind_indices, exp_tilde_indices,
                                             ind_tilde_indices)
            self.test_decoded_outputs.append(decoded_outputs)
            self.test_decoded_labels.append((indices, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices))

    def test_epoch_end(self, outputs):
        print("Inference on testset completed. Commencing evaluation...")
        initial_jobs = [i[0] for i in self.test_decoded_labels]
        initial_exp = [i[1] for i in self.test_decoded_labels]
        initial_ind = [i[2] for i in self.test_decoded_labels]
        expected_exp = [i[3] for i in self.test_decoded_labels]
        expected_ind = [i[4] - len(self.exp_dict) for i in self.test_decoded_labels]
        if self.hp.input_recopy == "True":
            stacked_outs = torch.stack(initial_jobs)
            ppl = self.get_ppl(stacked_outs[:, :, 1:])
            ind_metrics, exp_metrics, ind_preds, exp_preds = self.get_att_control_score(stacked_outs, expected_exp, expected_ind)
            ipdb.set_trace()
            bleu = self.get_bleu_score(stacked_outs[:, :, 1:], initial_jobs)
        else:
            ## NORMAL EVAL
            ppl = self.get_ppl(self.test_decoded_outputs)
            stacked_outs = torch.stack(self.test_decoded_outputs)
            ind_metrics, exp_metrics, ind_preds, exp_preds = self.get_att_control_score(stacked_outs, expected_exp, expected_ind)
            bleu = prettify_bleu_score((self.get_bleu_score(stacked_outs, initial_jobs)))
        if self.hp.print_to_csv == "True":
            csv_file = print_tilde_to_csv(initial_jobs, initial_exp, initial_ind, self.test_decoded_outputs, expected_exp, expected_ind, exp_preds, ind_preds, self.desc, self.vocab_dict, self.industry_dict, self.exp_dict)
            df = pd.read_csv(csv_file)
            df.to_html(f'html/{self.desc}.html')
        return {"Avg ppl": ppl, **bleu, **ind_metrics, **exp_metrics}

    def get_y_and_y_tilde(self, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices):
        atts_as_first_token = embed_and_avg_attributes(exp_indices,
                                                       ind_indices,
                                                       self.attribute_embedder_as_tokens.weight.shape[-1],
                                                       self.attribute_embedder_as_tokens)
        atts_as_bias = embed_and_avg_attributes(exp_indices,
                                                ind_indices,
                                                self.voc_size,
                                                self.attribute_embedder_as_bias)
        atts_tilde_as_first_token = embed_and_avg_attributes(exp_tilde_indices,
                                                             ind_tilde_indices,
                                                             self.attribute_embedder_as_tokens.weight.shape[
                                                                 -1],
                                                             self.attribute_embedder_as_tokens)
        atts_tilde_as_bias = embed_and_avg_attributes(exp_tilde_indices,
                                                      ind_tilde_indices,
                                                      self.voc_size,
                                                      self.attribute_embedder_as_bias)
        return atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias

    def save_at_step(self, batch_nb):
        if not os.path.isdir(self.model_path):
            os.system("mkdir -p " + self.model_path)
        prev_file = os.path.join(self.model_path, "step_" + str(batch_nb - self.save_step) + "_save.ckpt")
        if os.path.isfile(prev_file):
            os.system('rm ' + prev_file)
        tgt_file = os.path.join(self.model_path, "step_" + str(batch_nb) + "_save.ckpt")
        if not os.path.isfile(tgt_file):
            torch.save(self, tgt_file)
            print("checkpoint saved @ " + str(tgt_file))

    def get_bleu_score(self, test_outputs, test_labels):
        lab_file = os.path.join(self.datadir, self.desc + '_bleu_LABELS.txt')
        pred_file = os.path.join(self.datadir, self.desc + '_bleu_PREDS.txt')

        if os.path.isfile(lab_file):
            os.system('rm ' + lab_file)
            print("Removed " + lab_file)
        if os.path.isfile(pred_file):
            os.system('rm ' + pred_file)
            print("Removed " + pred_file)
        # make bleu label file
        for num, predicted_tokens in tqdm(enumerate(test_outputs), desc="Building txt files for BLEU score..."):
            str_preds = decode_tokenized_words(predicted_tokens[0], self.vocab_dict)
            labels = decode_tokenized_words(test_labels[num][0], self.vocab_dict)
            preds = str_preds.split(" ")
            # we take the <s> token out because the decoder id not supposed to predict it
            labs = labels.split(" ")[1:]
            eos_reached = False
            num_tok = 0
            if len(preds) < 1:
                ipdb.set_trace()
            if len(labs) < 1:
                ipdb.set_trace()
            with open(pred_file, "a+") as f:
                while (not eos_reached) and (num_tok < len(preds)):
                    f.write(preds[num_tok])
                    f.write(" ")
                    if preds[num_tok] == "</s>":
                        eos_reached = True
                    num_tok += 1
                f.write("\n")

            num_tok = 0
            with open(lab_file, "a+") as f:
                while num_tok < len(labs):
                    f.write(labs[num_tok])
                    f.write(" ")
                    num_tok += 1
                f.write("\n")
            # get score
        print(f"Computing BLEU score for model {self.desc}")
        print(f"Pred file: {pred_file}")
        return compute_bleu_score(pred_file, lab_file)

    def get_att_control_score(self, predicted_tokens, expected_exp, expected_ind):
        mod_type = self.hp.classifier_type
        rev_exp_dict = {v: k for k, v in self.exp_dict.items()}
        predicted_inds = []
        predicted_exps = []

        rev_ind_dict = dict()
        for k, v in self.industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        ind_labels = torch.stack(expected_ind).cpu().numpy()
        exp_labels = torch.stack(expected_exp).cpu().numpy()
        if mod_type == "ft":
            if self.max_len == 10:
                suffix = '_10'
            else:
                suffix = ""
            classifier_ind = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_ind{suffix}.bin")
            classifier_exp = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_exp_5{suffix}.bin")
            for sequence in tqdm(predicted_tokens, desc=f"evaluating attribute control with {mod_type} model..."):
                sentence = decode_tokenized_words(sequence[0], self.vocab_dict)
                predictions_ind = handle_fb_preds(classifier_ind.predict(sentence, k=10))
                predictions_exp = handle_fb_preds(classifier_exp.predict(sentence, k=2))
                predicted_inds.append([rev_ind_dict[i] for i in predictions_ind])
                predicted_exps.append([rev_exp_dict[i] for i in predictions_exp])
            predicted_exps = np.array(predicted_exps)
            predicted_inds = np.array(predicted_inds)
            preds_ind_at_1 = np.expand_dims(predicted_inds[:, 0], 1)
            preds_exp_at_1 = np.expand_dims(predicted_exps[:, 0], 1)
        elif mod_type == "SVM" or mod_type == "NB":
            classifier_exp = joblib.load(f"/data/gainondefor/dynamics/exp_best_{mod_type}.joblib")
            classifier_ind = joblib.load(f"/data/gainondefor/dynamics/ind_best_{mod_type}.joblib")
            vectorizer_exp = joblib.load(f"/data/gainondefor/dynamics/exp_best_{mod_type}_vectorizer.joblib")
            vectorizer_ind = joblib.load(f"/data/gainondefor/dynamics/ind_best_{mod_type}_vectorizer.joblib")
            predict_fn_exp = classifier_exp.decision_function if mod_type == "SVM" else classifier_exp.predict_proba
            predict_fn_ind = classifier_ind.decision_function if mod_type == "SVM" else classifier_ind.predict_proba
            for sequence in tqdm(predicted_tokens, desc=f"evaluating attribute control with {mod_type} model..."):
                sentence = decode_tokenized_words(sequence[0], self.vocab_dict)
                vectorized_sentence_exp = vectorizer_exp.transform([sentence])
                vectorized_sentence_ind = vectorizer_ind.transform([sentence])
                predictions_exp = handle_svm_pred(predict_fn_exp(vectorized_sentence_exp), k=2)
                predictions_ind = handle_svm_pred(predict_fn_ind(vectorized_sentence_ind), k=10)
                predicted_exps.append(predictions_exp)
                predicted_inds.append(predictions_ind)
            predicted_exps = np.stack(predicted_exps).squeeze(1)
            predicted_inds = np.stack(predicted_inds).squeeze(1)
            preds_exp_at_1 = predicted_exps[:, 0]
            preds_ind_at_1 = predicted_inds[:, 0]
        else:
            raise Exception(f"wrong classifier type, can be SVM, NB or ft, {mod_type} was given.")

        ind_metrics = get_metrics(preds_ind_at_1, ind_labels, len(self.industry_dict), 'ind')
        exp_metrics = get_metrics(preds_exp_at_1, exp_labels, len(self.exp_dict), 'exp')

        ind_metrics_at10 = get_metrics_at_k(predicted_inds, ind_labels, len(self.industry_dict), 'ind@10')
        exp_metrics_at2 = get_metrics_at_k(predicted_exps, exp_labels, len(self.exp_dict), 'exp@2')
        return {**ind_metrics, **ind_metrics_at10}, {**exp_metrics, **exp_metrics_at2}, predicted_inds, predicted_exps

    def get_ppl(self, decoded_tokens_indices):
        # for debug, don't put on cuda
        # self.ppl_model = self.ppl_model.cpu()
        ce = []
        for sentence in tqdm(decoded_tokens_indices, desc="computing perplexity..."):
            sentence_str = decode_tokenized_words(sentence[0], self.vocab_dict)
            tmp = self.tokenizer_fr(sentence_str, max_length=self.max_len, truncation=True, return_tensors="pt")
            tokenized_sentence, mask = tmp["input_ids"].cuda(), tmp["attention_mask"].cuda()
            outputs = self.ppl_model(tokenized_sentence, return_dict=True)
            logits = outputs.logits
            tmp2 = torch.nn.functional.cross_entropy(logits.transpose(-1, 1)[:, :, :-1], tokenized_sentence[:, 1:],
                                                     reduction='none')
            ce.append((tmp2 * mask[:, 1:]).sum() / mask[:, 1:].sum())
            # ipdb.set_trace()
        try:
            return 2 ** (torch.stack(ce).mean().item() / math.log(2))
        except OverflowError:
            return math.inf

    def add_noise(self, jobs, masks, max_len):
        sample_len = len(jobs)
        # https://arxiv.org/pdf/1711.00043.pdf p4, "noise model"
        if self.hp.drop_proba > 0:
            drop_proba = self.hp.drop_proba
            # word drop
            seq_len = [sum(i).item() for i in masks]
            keep = np.random.rand(sample_len, max_len) >= drop_proba
            keep[:, 0] = 1
            keep = torch.from_numpy(keep).type_as(jobs)
            dropped_jobs = torch.ones(sample_len, max_len).type_as(jobs)
            for i in range(max_len - 2):
                dropped_jobs[:, i] = keep[:, i] * jobs[:, i]
            for b in range(sample_len):
                dropped_jobs[b, seq_len[b] - 1] = 6
            dropped_jobs = dropped_jobs.type_as(jobs)
        else:
            dropped_jobs = jobs
        # word shuffle
        if self.hp.shuffle_dist > 0:
            shuffle_dist = self.hp.shuffle_dist
            shuffled_jobs = dropped_jobs.clone()
            permutation = np.arange(max_len)
            perm_tmp = torch.from_numpy(permutation).expand(sample_len, max_len)
            rdn_tensor = torch.zeros(sample_len, max_len)
            for i in range(max_len):
                rdn_tensor[:, i] = torch.from_numpy(np.random.uniform(0, shuffle_dist + 1, sample_len))
            perm_exp = perm_tmp + rdn_tensor
            perm_exp[:, 0] = -1
            new_indices = perm_exp.argsort()
            for b in range(sample_len):
                for i in range(max_len):
                    shuffled_jobs[b][i] = dropped_jobs[b][new_indices[b][i]]
            return shuffled_jobs.type_as(jobs)
        else:
            return dropped_jobs

    @staticmethod
    def get_prediction_mask(sentence):
        mask = torch.zeros(sentence.shape[0], sentence.shape[1])
        eos_flag = False
        for pos, token in enumerate(sentence[0]):
            if not eos_flag:
                mask[0, pos] = 1
            if token.item() == 6:  # token for eos
                eos_flag = True
        return mask.type(torch.cuda.LongTensor)
