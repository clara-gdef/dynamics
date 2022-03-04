import itertools
import ipdb
import torch
from utils.models import compute_bleu_score, embed_and_avg_attributes, masked_softmax
from nltk.tokenize import word_tokenize
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from tqdm import tqdm
import random
from transformers import CamembertModel, CamembertTokenizer

from models.classes import BiLSTMEncoder, LSTMAttentionDecoder


class BackTranslator(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, fine_tune_word_emb, vocab_size, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_size
        self.desc = desc
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.num_ind = 147
        self.num_exp_level = 5

        with open(os.path.join(self.datadir, "lookup_att_emb.pkl"), "rb") as f:
            self.attribute_lookup = pkl.load(f)

        self.word_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.word_embedder = CamembertModel.from_pretrained('camembert-base')
        if not fine_tune_word_emb:
            self.word_embedder.requires_grad = False
        with open(os.path.join(self.datadir, "embs_exp_ind.t"), "rb") as f:
            emb_atts = torch.load(f)
        self.attribute_embedder_as_tokens = torch.nn.Embedding(self.num_ind + self.num_exp_level,
                                                               emb_atts.shape[-1])
        self.attribute_embedder_as_tokens.from_pretrained(emb_atts, freeze=False)
        self.attribute_embedder_as_bias = torch.nn.Embedding(self.num_ind + self.num_exp_level, self.voc_size)

        self.encoder = BiLSTMEncoder(self.emb_dim, self.hp.hidden_size, hp.pooling_window, hp)
        self.decoder = LSTMAttentionDecoder(self.hp.hidden_size, self.emb_dim, self.voc_size,
                                            hp)


    def forward_para(self, jobs, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices):
        sample_len = len(jobs)

        atts = self.get_y_and_y_tilde(exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[2], \
                                                                                           atts[3]
        jobs_embedded = self.word_embedder(jobs, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs, enc_hidden = self.encoder.forward(jobs_embedded, masks, enc_hidden)

        previous_token = atts_tilde_as_first_token
        prev_hidden = self.decoder.init_hidden()
        decoded_tokens = []
        for di in range(self.max_len - 1):
            resized_in_token = self.decoder.lin_att_in_para(previous_token)
            tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, masks, self.max_len)
            attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)

            output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

            output_lstm, hidden = self.decoder.LSTM(output, prev_hidden)
            output_lin = self.decoder.lin_lstm_out(output_lstm)
            decoder_output = output_lin + atts_as_bias
            # you cannot teacher force it, you don't have the label
            decoder_tok = torch.argmax(decoder_output, dim=-1)
            decoded_tokens.append(decoder_tok)
            prev_hidden = hidden
            previous_token = self.word_embedder(decoder_tok, return_dict=True)["last_hidden_state"]

        # artificially add an "end of sequence" token
        decoded_tokens.append((torch.ones(sample_len, 1).type(torch.LongTensor) * 6).type_as(decoded_tokens[0]))
        decoded_x_tilde = torch.stack(decoded_tokens).view(sample_len, self.max_len).type_as(decoded_tokens[0])
        # computing new masks
        tmp = (decoded_x_tilde == torch.ones(decoded_x_tilde.shape[0], decoded_x_tilde.shape[1]).type_as(decoded_x_tilde)).type(
            torch.cuda.BoolTensor)
        new_masks = ~tmp

        embedded_x_tiled = self.word_embedder(decoded_x_tilde, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs_2, enc_hidden_2 = self.encoder.forward(embedded_x_tiled, new_masks, enc_hidden)
        # labels
        jobs_embedded = self.word_embedder(jobs, return_dict=True)["last_hidden_state"]
        jobs_embedded[:, 0, :] = atts_as_first_token[:, 0, :]

        resized_conditioning = self.decoder.lin_att_in_para(jobs_embedded[:, :-1, :])
        tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
        attn_weights = masked_softmax(tmp, masks, self.max_len)
        attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs_2, attn_weights)
        decoder_inputs = torch.cat((jobs_embedded[:, :-1, :], attn_applied), -1)
        decoder_output = self.decoder.forward_para(decoder_inputs, atts_as_bias)

        loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                 jobs[:, 1:], reduction="sum", ignore_index=1)
        if torch.isnan(loss):
            ipdb.set_trace()
        return loss

    def inference_para(self, jobs, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices):
        sample_len = len(jobs)
        atts = self.get_y_and_y_tilde(exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias = atts[0], atts[1], atts[2], \
                                                                                           atts[3]
        jobs_embedded = self.word_embedder(jobs, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs, enc_hidden = self.encoder.forward(jobs_embedded, masks, enc_hidden)

        previous_token = atts_tilde_as_first_token
        prev_hidden = self.decoder.init_hidden()
        decoded_tokens = []
        for di in range(self.max_len - 1):
            resized_in_token = self.decoder.lin_att_in_para(previous_token)
            tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, masks, self.max_len)
            attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)

            output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

            output_lstm, hidden = self.decoder.LSTM(output, prev_hidden)
            output_lin = self.decoder.lin_lstm_out(output_lstm)
            decoder_output = output_lin + atts_as_bias
            # you cannot teacher force it, you don't have the label
            decoder_tok = torch.argmax(decoder_output, dim=-1)
            decoded_tokens.append(decoder_tok)
            prev_hidden = hidden
            previous_token = self.word_embedder(decoder_tok, return_dict=True)["last_hidden_state"]

        # artificially add an "end of sequence" token
        decoded_tokens.append((torch.ones(sample_len, 1).type(torch.LongTensor) * 6))
        decoded_x_tilde = torch.stack(decoded_tokens).view(sample_len, self.max_len).type(
            torch.LongTensor)
        # computing new masks
        tmp = (decoded_x_tilde == torch.ones(decoded_x_tilde.shape[0], decoded_x_tilde.shape[1])).type(
            torch.BoolTensor)
        new_masks = ~tmp

        embedded_x_tiled = self.word_embedder(decoded_x_tilde, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs_2, enc_hidden_2 = self.encoder.forward(embedded_x_tiled, new_masks, enc_hidden)

        decoded_tokens = []
        previous_token = atts_as_first_token
        prev_hidden = self.decoder.init_hidden()
        for di in range(self.max_len - 2):
            resized_in_token = self.decoder.lin_att_in_para(previous_token)
            tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, masks, self.max_len)
            attn_applied = torch.einsum("beh,bed->bh", encoder_outputs_2, attn_weights)

            output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

            output_lstm, hidden = self.decoder.LSTM(output, prev_hidden)
            output_lin = self.decoder.lin_lstm_out(output_lstm)
            decoder_output = output_lin + atts_as_bias
            decoded_tok = torch.argmax(decoder_output, dim=-1)
            previous_token = self.word_embedder(decoded_tok, return_dict=True)["last_hidden_state"]
            prev_hidden = hidden
            decoded_tokens.append(decoded_tok)

        return torch.stack(decoded_tokens).view(sample_len, self.hp.max_len - 2)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)

    def training_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        sample_len = len(indices)
        exp_tilde_indices, ind_tilde_indices = self.sample_y_tilde(exp_indices, ind_indices)
        if self.hp.para == "True":
            loss_bt_total = self.forward_para(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        else:
            loss_bt_total = self.forward(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        loss_bt_per_sample = loss_bt_total / sample_len
        self.log('train_loss_ep', loss_bt_per_sample, on_step=False, on_epoch=True)
        self.log('train_loss_st', loss_bt_per_sample, on_step=True, on_epoch=False)
        self.log('train_full_loss', loss_bt_total)
        return {"loss": loss_bt_per_sample}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}


    def validation_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        sample_len = len(indices)
        exp_tilde_indices, ind_tilde_indices = self.sample_y_tilde(exp_indices, ind_indices)
        if self.hp.para == "True":
            loss_bt_total = self.forward_para(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        else:
            loss_bt_total = self.forward(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        loss_bt_per_sample = loss_bt_total / sample_len
        self.log('val_loss', loss_bt_per_sample, prog_bar=True)
        self.log('full_val_loss', loss_bt_total)
        return {"val_loss": loss_bt_per_sample}

    def validation_step_end(self, validation_step_outputs):
        return {'val_loss': validation_step_outputs['val_loss'].sum()}

    def test_epoch_start(self):
        lab_file = os.path.join(self.datadir, self.desc + '_LABELS.txt')
        pred_file = os.path.join(self.datadir, self.desc + '_PREDS.txt')
        if os.path.isfile(lab_file):
            os.system('rm ' + lab_file)
        if os.path.isfile(pred_file):
            os.system('rm ' + pred_file)

    def test_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        exp_tilde_indices, ind_tilde_indices = self.sample_y_tilde(exp_indices, ind_indices)
        if self.hp.para == "True":
            decoded_tokens = self.inference_para(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        else:
            decoded_tokens = self.inference(indices, masks, exp_indices, ind_indices, exp_tilde_indices, ind_tilde_indices)
        str_preds = self.word_tokenizer.decode(decoded_tokens[0])
        labels = self.word_tokenizer.decode(indices[0][1:])

        lab_file = os.path.join(self.datadir, self.desc + '_LABELS.txt')
        pred_file = os.path.join(self.datadir, self.desc + '_PREDS.txt')
        preds = word_tokenize(str_preds)
        labs = word_tokenize(labels.split("</s>")[0])
        eos_reached = False
        num_tok = 0

        with open(pred_file, "a+") as f:
            while not eos_reached and num_tok < len(preds):
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
            f.write("</s>")
            f.write("\n")

    def test_epoch_end(self, outputs):
        ref = os.path.join(self.datadir, self.desc + '_LABELS.txt')
        pred = os.path.join(self.datadir, self.desc + '_PREDS.txt')
        print("Computing BLEU score for model " + self.desc)
        compute_bleu_score(pred, ref)
        ipdb.set_trace()

    def sample_y_tilde(self, exp_index, ind_index):
        sample_len = len(exp_index)
        exp_tilde_index = torch.zeros(sample_len).type(torch.LongTensor)
        ind_tilde_index = torch.zeros(sample_len).type(torch.LongTensor)
        for num in range(sample_len):
            rnd_exp = exp_index[num]
            rnd_ind = ind_index[num]
            while rnd_exp == exp_index[num]:
                rnd_exp = random.randrange(0, 5)
            while rnd_ind == ind_index[num]:
                rnd_ind = random.randrange(5, 152)
            exp_tilde_index[num] = rnd_exp
            ind_tilde_index[num] = rnd_ind
        return exp_tilde_index.type_as(exp_index), ind_tilde_index.type_as(exp_index)

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

