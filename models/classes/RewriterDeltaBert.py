import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import math
import pickle as pkl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import joblib
import fasttext
import pandas as pd
from models.classes import BiLSTMEncoder, LSTMAttentionDecoder
from models.classes.CausalLM import CausalLM
from utils import print_tilde_to_csv
from utils.models import masked_softmax, embed_and_avg_attributes, sample_y_tilde, compute_bleu_score, \
    handle_fb_preds, get_metrics, decode_tokenized_words, get_metrics_at_k, beam_search_decoder, prettify_bleu_score, \
    handle_svm_pred, plot_grad_flow, print_cm
from transformers import CamembertTokenizer, CamembertModel, CamembertForCausalLM


class RewriterDeltaBert(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, vocab_size, alpha, beta, model_path, hp):
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

        with open(os.path.join(self.datadir, "delta_dict.pkl"), 'rb') as f_name:
            self.delta_dict = pkl.load(f_name)
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

        self.num_ind = len(self.industry_dict)
        self.num_delta_level = len(self.delta_dict)

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.encoder = CamembertModel.from_pretrained('camembert-base')
        if self.hp.feed_z == 'True':
            self.decoder = LSTMAttentionDecoder(self.hp.dec_hs, self.emb_dim, self.voc_size, hp)
        else:
            self.decoder = LSTMAttentionDecoder(self.hp.dec_hs, self.emb_dim, self.voc_size, hp)
        self.attribute_embedder_as_tokens_exp = torch.nn.Embedding(self.num_delta_level,
                                                               self.emb_dim)
        self.attribute_embedder_as_bias_exp = torch.nn.Embedding(self.num_delta_level, self.voc_size)
        self.attribute_embedder_as_tokens_ind = torch.nn.Embedding(self.num_ind,
                                                               self.emb_dim)
        self.attribute_embedder_as_bias_ind = torch.nn.Embedding(self.num_ind, self.voc_size)

    def forward(self, jobs, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices, fwd_type, batch_nb):
        if fwd_type == "DAE":
            attributes_as_first_token = embed_and_avg_attributes(delta_indices,
                                                                 ind_indices,
                                                                 self.attribute_embedder_as_tokens_exp.weight.shape[-1],
                                                                 self.attribute_embedder_as_tokens_exp,
                                                                 self.attribute_embedder_as_tokens_ind)
            attributes_as_bias = embed_and_avg_attributes(delta_indices,
                                                          ind_indices,
                                                          self.voc_size,
                                                          self.attribute_embedder_as_bias_exp,
                                                          self.attribute_embedder_as_bias_ind)
            inputs = self.tokenizer(jobs, truncation=True, padding="max_length", max_length=self.max_len,
                                    return_tensors="pt")
            input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            max_seq_len = input_tokenized.shape[-1]
            inputs_corrupted = self.add_noise(input_tokenized,
                                              mask,
                                              max_seq_len)
            inputs_corrupted_embedded = self.encoder.embeddings(inputs_corrupted)
            encoder_outputs = self.encoder(inputs_corrupted, mask)['last_hidden_state']
            # labels
            jobs_embedded = self.encoder.embeddings(input_tokenized)
            jobs_embedded[:, 0, :] = attributes_as_first_token[:, 0, :]

            h0 = (torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(jobs_embedded),
                  torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(jobs_embedded))

            resized_conditioning = self.decoder.lin_att_in_para(inputs_corrupted_embedded[:, :-1, :])
            tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, mask, max_seq_len)
            attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs, attn_weights)
            decoder_inputs = torch.cat((encoder_outputs[:, :-1, :], jobs_embedded[:, :-1, :], attn_applied), -1)
            decoder_output, hs = self.decoder.forward(decoder_inputs, attributes_as_bias, h0)

            loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                     input_tokenized[:, 1:], reduction="sum", ignore_index=1)
            # if self.hp.print_preds == "True":
            #     print(f"X = {jobs}")
            #     print(f"X_c = {self.tokenizer.batch_decode(inputs_corrupted, skip_special_tokens=True)}")
            #     print(
            #          f"X_hat = {self.tokenizer.batch_decode(torch.argmax(decoder_output, dim=-1), skip_special_tokens=True)}")
            #     print("================================================================")

            if torch.isnan(loss):
                ipdb.set_trace()
            return loss, mask

        elif fwd_type == "BT":
            sample_len = len(jobs)

            atts = self.get_y_and_y_tilde(delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices)
            atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[
                2], atts[3]

            inputs = self.tokenizer(jobs, truncation=True, padding="max_length", max_length=self.max_len,
                                    return_tensors="pt")
            input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            max_seq_len = input_tokenized.shape[-1]
            input_embedded = self.encoder.embeddings(input_tokenized)
            encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']

            previous_token = atts_tilde_as_first_token
            prev_hidden_state = (torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(previous_token),
                                 torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(previous_token))

            decoded_tokens = []
            decoder_outputs = []
            for di in range(max_seq_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
                attn_weights = masked_softmax(tmp, mask, 1)
                assert attn_weights.shape == torch.Size([sample_len, max_seq_len, 1])
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)
                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)
                if self.hp.feed_z == 'True':
                    lstm_input = torch.cat((encoder_outputs[:, di, :].unsqueeze(1), previous_token, attn_applied.unsqueeze(1)), -1)
                    output_lstm, hidden_state = self.decoder.LSTM(lstm_input,
                                                                  prev_hidden_state)
                else:
                    output_lstm, hidden_state = self.decoder.LSTM(output, prev_hidden_state)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = output_lin + atts_as_bias
                decoder_outputs.append(decoder_output)
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.encoder.embeddings(decoder_tok)
                prev_hidden_state = hidden_state

            reshaped_decoder_outputs = torch.stack(decoder_outputs).squeeze(1).transpose(1, 0)
            decoded_jobs = self.tokenizer.batch_decode(torch.stack(decoded_tokens).squeeze(-1).T,
                                                       skip_special_tokens=True)
            inputs2 = self.tokenizer(decoded_jobs, truncation=True, padding="max_length", max_length=self.max_len,
                                     return_tensors="pt")
            decoded_x_tilde, new_mask = inputs2["input_ids"].cuda(), inputs2["attention_mask"].cuda()
            max_seq_len = decoded_x_tilde.shape[-1]

            encoder_outputs_2 = self.encoder(decoded_x_tilde, new_mask)["last_hidden_state"]
            embedded_decoded_x_tilde = self.encoder.embeddings(decoded_x_tilde)
            # labels
            input_embedded[:, 0, :] = atts_as_first_token[:, 0, :]

            h0 = (torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(input_embedded),
                                 torch.zeros(self.decoder.num_layer, self.hp.b_size, self.decoder.hs).type_as(input_embedded))

            resized_conditioning = self.decoder.lin_att_in_para(input_embedded[:, :-1, :])
            tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, new_mask, max_seq_len)
            assert attn_weights.shape == torch.Size([sample_len, max_seq_len, max_seq_len - 1])
            #attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs_2, attn_weights)
            attn_applied = torch.einsum("beh,bed->beh", encoder_outputs_2, attn_weights)

            # decoder_inputs = torch.cat((input_embedded[:, :-1, :], attn_applied), -1)
            if self.hp.feed_z == 'True':
                # decoder_inputs = torch.cat((encoder_outputs[:, :-1, :], embedded_decoded_x_tilde[:, :-1, :], attn_applied), -1)
                decoder_inputs = torch.cat((encoder_outputs, embedded_decoded_x_tilde, attn_applied), -1)
                decoder_output, hs = self.decoder.forward(decoder_inputs, atts_as_bias, h0)
            else:
                decoder_inputs = torch.cat((embedded_decoded_x_tilde[:, :-1, :], attn_applied),
                                           -1)  # unsure about this, should be tested
                decoder_output, hs = self.decoder.forward(decoder_inputs, atts_as_bias, h0)

            loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                         input_tokenized, reduction="sum", ignore_index=1)
            if self.hp.print_preds == "True" and batch_nb == 100:
                print(f"X = {jobs[0]}")
                print(f"X_tilde = {self.tokenizer.decode(decoded_x_tilde[0], skip_special_tokens=True)}")
                #print(f"label to guess IND = {self.industry_dict[ind_tilde_indices[0].item()]}")
                print(f"label to guess EXP = {delta_tilde_indices[0].item()}")
                print(
                    f"X_hat = {self.tokenizer.decode(torch.argmax(decoder_output[0], dim=-1), skip_special_tokens=True)}")
            if torch.isnan(loss):
                ipdb.set_trace()
            return loss, mask
        else:
            raise Exception("wrong forward type, can be either \"DAE\" or \"BT\", " + str(fwd_type) + " was given.")

    def inference(self, jobs, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices):
        if self.hp.debug_noiseless == "True":
            inputs = self.tokenizer(jobs, truncation=True, padding="max_length", max_length=self.max_len,
                                    return_tensors="pt")
            input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']
            previous_token = self.encoder.embeddings(torch.ones(1, 1).type(torch.LongTensor).cuda() * 5)

            prev_hidden = (torch.zeros(self.decoder.num_layer, 1, self.decoder.hs).type_as(encoder_outputs),
                                 torch.zeros(self.decoder.num_layer, 1, self.decoder.hs).type_as(encoder_outputs))

            decoded_tokens, posteriors = [], []
            for di in range(self.max_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1).type_as(encoder_outputs))
                attn_weights = masked_softmax(tmp, mask, 1)
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)
                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)
                output_lstm, hidden = self.decoder.LSTM(output, prev_hidden)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = torch.softmax(output_lin, dim=-1)
                posteriors.append(decoder_output)
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.encoder.embeddings(decoder_tok)
                prev_hidden = hidden
            resized_preds = torch.stack(posteriors).view(1, self.max_len - 1, self.voc_size)
            indices, log_prob = beam_search_decoder(resized_preds, 3)
            selected_seq = indices[0, torch.argmax(log_prob)]
            return selected_seq.unsqueeze(0)
            # return torch.stack(decoded_tokens).squeeze(-1).view(1, -1)
        else:
            sample_len = len(jobs)
            atts = self.get_y_and_y_tilde(delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices)
            atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[
                2], atts[3]

            inputs = self.tokenizer(jobs, truncation=True, padding="max_length", max_length=self.max_len,
                                    return_tensors="pt")
            input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            max_seq_len = input_tokenized.shape[-1]
            encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']
            previous_token = atts_tilde_as_first_token
            prev_hidden_state = (torch.zeros(self.decoder.num_layer, self.hp.test_b_size, self.decoder.hs).type_as(previous_token),
                                 torch.zeros(self.decoder.num_layer, self.hp.test_b_size, self.decoder.hs).type_as(previous_token))

            decoded_tokens, posteriors = [], []
            for di in range(max_seq_len - 1):
                resized_in_token = self.decoder.lin_att_in_para(previous_token)
                tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
                attn_weights = masked_softmax(tmp, mask, 1)
                assert attn_weights.shape == torch.Size([sample_len, max_seq_len, 1])
                attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)
                output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)
                if self.hp.feed_z == 'True':
                    lstm_input = torch.cat((encoder_outputs[:, di, :].unsqueeze(1), previous_token, attn_applied.unsqueeze(1)), -1)
                    output_lstm, hidden_state = self.decoder.LSTM(lstm_input,
                                                                  prev_hidden_state)
                else:
                    output_lstm, hidden_state = self.decoder.LSTM(output, prev_hidden_state)
                output_lin = self.decoder.lin_lstm_out(output_lstm)
                decoder_output = output_lin + atts_as_bias
                posteriors.append(decoder_output)
                decoder_tok = torch.argmax(decoder_output, dim=-1)
                decoded_tokens.append(decoder_tok)
                previous_token = self.encoder.embeddings(decoder_tok)
                prev_hidden_state = hidden_state
            if self.hp.beam_search == "True":
                resized_preds = torch.stack(posteriors).view(self.hp.test_b_size, self.max_len - 1, self.voc_size)
                indices, log_prob = beam_search_decoder(torch.softmax(resized_preds, dim=-1), 3)
                if torch.isnan(log_prob).any():
                    ipdb.set_trace()
                selected_seq = torch.zeros(self.hp.b_size, self.max_len-1)
                for i, j in enumerate(torch.argmax(log_prob, dim=-1)):
                    selected_seq[i, :] = indices[i, j, :]
                return selected_seq
                ipdb.set_trace()
            else:
                return torch.stack(decoded_tokens).squeeze(-1).view(1, -1)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())

        if self.hp.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hp.lr)

        else:
            optimizer = torch.optim.SGD(params, lr=self.hp.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer),
            'monitor': 'train_loss_ep',
            'interval': 'step',
            'frequency': 10,
            'strict': True
        }

    def training_step(self, batch, batch_nb):
        sentences, ind_indices, delta_indices = batch[0], batch[1], batch[2]
        if self.hp.att_type == "exp":
            ind_indices = None
        elif self.hp.att_type == "ind":
            delta_indices = None

        loss_dae_tmp, mask = self.forward(sentences, delta_indices, ind_indices, None, None, 'DAE', batch_nb)
        sample_len = sum(sum(mask))

        loss_dae = loss_dae_tmp / sample_len

        if self.hp.denoising_only == "False":
            delta_tilde_indices, ind_tilde_indices = sample_y_tilde(delta_indices, ind_indices, len(self.delta_dict),
                                                                    len(self.industry_dict), self.hp.no_tilde)
            loss_bt_tmp, mask = self.forward(sentences, delta_indices, ind_indices, delta_tilde_indices,
                                             ind_tilde_indices, 'BT', batch_nb)
            sample_len = sum(sum(mask))
            loss_bt = loss_bt_tmp / sample_len
            loss = self.alpha * loss_dae + self.beta * loss_bt
        else:
            loss = loss_dae
        self.log('train_loss_ep', loss, on_step=False, on_epoch=True)
        self.log('train_loss_st', loss, on_step=True, on_epoch=False)

        self.log('train_loss_dae_ep', loss_dae, on_step=False, on_epoch=True)
        self.log('train_loss_dae_st', loss_dae, on_step=True, on_epoch=False)

        # self.log('train_loss_bt_ep', loss_bt, on_step=False, on_epoch=True)
        # self.log('train_loss_bt_st', loss_bt, on_step=True, on_epoch=False)
        # loss.backward()
        # plot_grad_flow(self.named_parameters(), self.desc)

        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].mean()}

    def training_epoch_end(self, training_step_outputs):
        tmp = torch.stack([i["loss"] for i in training_step_outputs]).mean()
        training_step_outputs = tmp

    def validation_step(self, batch, batch_nb):
        sentences, ind_indices, delta_indices = batch[0], batch[1], batch[2]
        if self.hp.att_type == "exp":
            ind_indices = None
        elif self.hp.att_type == "ind":
            delta_indices = None

        loss_dae_tmp, mask = self.forward(sentences, delta_indices, ind_indices, None, None, 'DAE', batch_nb)
        sample_len = sum(sum(mask))
        loss_dae = loss_dae_tmp / sample_len

        if self.hp.denoising_only == "False":
            delta_tilde_indices, ind_tilde_indices = sample_y_tilde(delta_indices, ind_indices, len(self.delta_dict),
                                                                    len(self.industry_dict), self.hp.no_tilde)
            loss_bt_tmp, mask = self.forward(sentences, delta_indices, ind_indices, delta_tilde_indices,
                                             ind_tilde_indices, 'BT', batch_nb)
            sample_len = sum(sum(mask))
            loss_bt = loss_bt_tmp / sample_len

            val_loss = self.alpha * loss_dae + self.beta * loss_bt
        else:
            val_loss = loss_dae

        self.log('valid_loss_ep', val_loss, on_step=False, on_epoch=True)
        self.log('valid_loss_dae_ep', loss_dae, on_step=False, on_epoch=True)
        # self.log('valid_loss_bt_ep', loss_bt, on_step=False, on_epoch=True)

        self.log('val_loss', val_loss)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_epoch_outputs):
        tmp = torch.stack([i["val_loss"] for i in validation_epoch_outputs]).mean()
        validation_epoch_outputs = tmp
        return {'val_loss': tmp}

    def on_test_epoch_start(self):
        self.test_decoded_outputs = []
        self.test_decoded_labels = []
        self.classifier_ind = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_ind_10.bin")
        self.classifier_exp = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_delta_5_10.bin")
        self.ppl_model = CausalLM(self.datadir, "causal_lm_bs100_lr1e-07_adam", self.voc_size, self.model_path,
                                  self.hp).cuda()
        self.ppl_model.is_decoder = True
        self.ppl_model.load_state_dict(
            torch.load("/data/gainondefor/dynamics/causal/lm/bs100/lr1e-07/adam/epoch=05.ckpt")["state_dict"])
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

    def test_step(self, batch, batch_nb):
        sentences, ind_indices, delta_indices = batch[0], batch[1], batch[2]
        if self.hp.input_recopy == "True":
            self.test_decoded_labels.append((sentences, delta_indices, ind_indices))
        else:
            delta_tilde_indices, ind_tilde_indices = sample_y_tilde(delta_indices, ind_indices, len(self.delta_dict),
                                                                    len(self.industry_dict), self.hp.no_tilde)
            decoded_outputs = self.inference(sentences, delta_indices, ind_indices, delta_tilde_indices,
                                             ind_tilde_indices)
            self.test_decoded_outputs.append(decoded_outputs)
            self.test_decoded_labels.append(
                (sentences, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices))

    def test_epoch_end(self, outputs):
        print("Inference on testset completed. Commencing evaluation...")
        initial_jobs = [i[0] for i in self.test_decoded_labels]
        initial_exp = [i[1] for i in self.test_decoded_labels]
        initial_ind = [i[2] for i in self.test_decoded_labels]
        if self.hp.input_recopy == "True":
            str_list = []
            for i in initial_jobs:
                for sent in i:
                    str_list.append(sent)
            print(f"Tokenizing {len(str_list)} sentences...")
            tokenized = self.tokenizer(str_list, truncation=True, padding="max_length", max_length=self.max_len,
                                       return_tensors="pt")
            ppl = self.get_ppl(tokenized["input_ids"])
            ind_metrics, delta_metrics, ind_preds, delta_preds = self.get_att_control_score(tokenized["input_ids"],
                                                                                            initial_exp, initial_ind)
            ipdb.set_trace()
            bleu = self.get_bleu_score(tokenized["input_ids"], initial_jobs)
        else:
            expected_delta = [i[3] for i in self.test_decoded_labels]
            expected_ind = [i[4] for i in self.test_decoded_labels]
            ## NORMAL EVAL
            ppl = self.get_ppl(self.test_decoded_outputs)
            stacked_outs = torch.stack(self.test_decoded_outputs)
            ind_metrics, delta_metrics, ind_preds, delta_preds = self.get_att_control_score(stacked_outs,
                                                                                            expected_delta,
                                                                                            expected_ind)
            bleu = prettify_bleu_score((self.get_bleu_score(stacked_outs.squeeze(1), initial_jobs)))
            # ipdb.set_trace()
        if self.hp.print_to_csv == "True":
            csv_file = print_tilde_to_csv(initial_jobs, initial_exp, initial_ind, self.test_decoded_outputs,
                                          expected_delta, expected_ind, delta_preds, ind_preds, self.desc,
                                          self.vocab_dict, self.industry_dict, self.delta_dict)
            df = pd.read_csv(csv_file)
            df.to_html(f'html/{self.desc}.html')
        print({"Avg ppl": ppl, **bleu, **ind_metrics, **delta_metrics})
        return {"Avg ppl": ppl, **bleu, **ind_metrics, **delta_metrics}

    def get_y_and_y_tilde(self, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices):
        atts_as_first_token = embed_and_avg_attributes(delta_indices,
                                                       ind_indices,
                                                       self.attribute_embedder_as_tokens_exp.weight.shape[-1],
                                                       self.attribute_embedder_as_tokens_exp,
                                                       self.attribute_embedder_as_tokens_ind)
        atts_as_bias = embed_and_avg_attributes(delta_indices,
                                                ind_indices,
                                                self.voc_size,
                                                self.attribute_embedder_as_bias_exp,
                                                self.attribute_embedder_as_bias_ind)
        atts_tilde_as_first_token = embed_and_avg_attributes(delta_tilde_indices,
                                                             ind_tilde_indices,
                                                             self.attribute_embedder_as_tokens_exp.weight.shape[
                                                                 -1],
                                                             self.attribute_embedder_as_tokens_exp,
                                                             self.attribute_embedder_as_tokens_ind
                                                             )
        atts_tilde_as_bias = embed_and_avg_attributes(delta_tilde_indices,
                                                      ind_tilde_indices,
                                                      self.voc_size,
                                                      self.attribute_embedder_as_bias_exp,
                                                      self.attribute_embedder_as_bias_ind)
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
        reshaped_predicted_tokens = test_outputs.view(-1, test_outputs.shape[-1])
        reshaped_labels = []
        for i in test_labels:
            for j in i:
                reshaped_labels.append(j)
        for num, predicted_tokens in tqdm(enumerate(reshaped_predicted_tokens), desc="Building txt files for BLEU score..."):
            str_preds = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            labels = self.tokenizer.decode(
                self.tokenizer(reshaped_labels[num], truncation=True, padding="max_length", max_length=self.max_len,
                               return_tensors="pt")["input_ids"][0], skip_special_tokens=True)
            if self.hp.input_recopy == "True":
                if str_preds != labels:
                    ipdb.set_trace()
            if len(str_preds.split(" ")) < 1:
                ipdb.set_trace()
            if len(labels.split(" ")) < 1:
                ipdb.set_trace()
            with open(pred_file, "a+") as f:
                f.write(str_preds + " \n")
            with open(lab_file, "a+") as f:
                f.write(labels + " \n")
            # get score
        print(f"Computing BLEU score for model {self.desc}")
        print(f"Pred file: {pred_file}")
        return compute_bleu_score(pred_file, lab_file)

    def get_att_control_score(self, predicted_tokens, expected_exp, expected_ind):
        mod_type = self.hp.classifier_type
        predicted_inds, predicted_exps = [], []

        reshaped_predicted_tokens = predicted_tokens.view(-1, predicted_tokens.shape[-1])

        rev_ind_dict = dict()
        for k, v in self.industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        ind_labels = torch.stack(expected_ind).view(-1, 1).cpu().numpy()
        assert 0 <= np.min(ind_labels)
        assert len(self.industry_dict) > np.max(ind_labels)
        delta_labels = torch.stack(expected_exp).view(-1, 1).cpu().numpy()

        for sequence in tqdm(reshaped_predicted_tokens, desc=f"evaluating attribute control with {mod_type} model..."):
            sentence = self.tokenizer.decode(sequence, skip_special_tokens=True)
            predictions_ind = handle_fb_preds(self.classifier_ind.predict(sentence, k=10))
            predictions_exp = handle_fb_preds(self.classifier_exp.predict(sentence, k=2))
            tmp = []
            for i in predictions_ind:
                tmp.append(rev_ind_dict[i])
            predicted_inds.append(tmp)
            predicted_exps.append([int(i) for i in predictions_exp])
        predicted_exps = np.array(predicted_exps)
        predicted_inds = np.array(predicted_inds)
        preds_ind_at_1 = np.expand_dims(predicted_inds[:, 0], 1)
        preds_delta_at_1 = np.expand_dims(predicted_exps[:, 0], 1)

        ind_metrics = get_metrics(preds_ind_at_1, ind_labels, len(self.industry_dict), 'ind')
        delta_metrics = get_metrics(preds_delta_at_1, delta_labels, len(self.delta_dict), 'exp')

        if self.hp.print_cm == "True":
            print_cm(preds_ind_at_1, ind_labels, self.desc, self.industry_dict, "ind")
            print_cm(preds_delta_at_1, delta_labels, self.desc, self.delta_dict, "delta")

        ind_metrics_at10 = get_metrics_at_k(predicted_inds, ind_labels, len(self.industry_dict), 'ind@10')
        delta_metrics_at2 = get_metrics_at_k(predicted_exps, delta_labels, len(self.delta_dict), 'exp@2')
        return {**ind_metrics, **ind_metrics_at10}, {**delta_metrics,
                                                     **delta_metrics_at2}, predicted_inds, predicted_exps

    def get_ppl(self, decoded_tokens_indices):
        # for debug, don't put on cuda
        # self.ppl_model = self.ppl_model.cpu()
        ce = []
        for sentence in tqdm(decoded_tokens_indices, desc="computing perplexity..."):
            sentence_str = self.tokenizer.decode(sentence[0], skip_special_tokens=True)
            tmp = self.tokenizer(sentence_str, truncation=True, padding="max_length", max_length=self.max_len,
                                 return_tensors="pt")
            tokenized_sentence, mask = tmp["input_ids"].cuda(), tmp["attention_mask"].cuda()
            outputs = self.ppl_model.ppl_model(tokenized_sentence, return_dict=True)
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
