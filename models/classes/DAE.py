import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import random
from line_profiler import LineProfiler
from nltk.tokenize import word_tokenize
from transformers import CamembertModel, CamembertTokenizer

from models.classes import BiLSTMEncoder, LSTMAttentionDecoder
from utils.models import compute_bleu_score, embed_and_avg_attributes, masked_softmax


class DAE(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, fine_tune_word_emb, vocab_size, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_size
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.num_ind = 147
        self.num_exp_level = 5

        self.word_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.word_embedder = CamembertModel.from_pretrained('camembert-base')
        if not fine_tune_word_emb:
            self.word_embedder.requires_grad = False
        with open(os.path.join(self.datadir, "embs_exp_ind.t"), "rb") as f:
            emb_attributes = torch.load(f)
        self.attribute_embedder_as_tokens = torch.nn.Embedding(self.num_ind + self.num_exp_level,
                                                               emb_attributes.shape[-1])
        self.attribute_embedder_as_tokens.from_pretrained(emb_attributes, freeze=False)
        self.attribute_embedder_as_bias = torch.nn.Embedding(self.num_ind + self.num_exp_level, self.voc_size)
        self.encoder = BiLSTMEncoder(self.emb_dim, self.hp.hidden_size, hp.pooling_window, hp)
        self.decoder = LSTMAttentionDecoder(self.hp.hidden_size, self.emb_dim, self.voc_size,
                                            hp)

    def forward_para(self, jobs, masks, exp_indices, ind_indices, batch_nb):
        torch.autograd.set_detect_anomaly(True)
        attributes_as_first_token = embed_and_avg_attributes(exp_indices,
                                                                  ind_indices,
                                                                  self.attribute_embedder_as_tokens.weight.shape[-1],
                                                                  self.attribute_embedder_as_tokens)
        attributes_as_bias = embed_and_avg_attributes(exp_indices,
                                                           ind_indices,
                                                           self.voc_size,
                                                           self.attribute_embedder_as_bias)
        jobs_corrupted = self.add_noise(jobs, masks)
        # jobs_corrupted = jobs

        jobs_corrupted_embedded = self.word_embedder(jobs_corrupted, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs, enc_hidden = self.encoder.forward(jobs_corrupted_embedded, masks, enc_hidden)

        # labels
        jobs_embedded = self.word_embedder(jobs, return_dict=True)["last_hidden_state"]
        jobs_embedded[:, 0, :] = attributes_as_first_token[:, 0, :]

        resized_conditioning = self.decoder.lin_att_in_para(jobs_corrupted_embedded[:, :-1, :])
        tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
        attn_weights = masked_softmax(tmp, masks, self.max_len)
        attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs, attn_weights)
        decoder_inputs = torch.cat((jobs_embedded[:, :-1, :], attn_applied), -1)
        decoder_output = self.decoder.forward_para(decoder_inputs, attributes_as_bias)

        decoded_tokens = torch.argmax(decoder_output, dim=-1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                 jobs[:, 1:], reduction="sum", ignore_index=1)
        if torch.isnan(loss):
            ipdb.set_trace()
        return loss, decoded_tokens

    def inference_para(self, jobs, masks, exp_indices, ind_indices, batch_nb):
        torch.autograd.set_detect_anomaly(True)
        sample_len = len(jobs)
        attributes_as_first_token = embed_and_avg_attributes(exp_indices,
                                                                  ind_indices,
                                                                  self.attribute_embedder_as_tokens.weight.shape[-1],
                                                                  self.attribute_embedder_as_tokens)
        attributes_as_bias = embed_and_avg_attributes(exp_indices,
                                                           ind_indices,
                                                           self.voc_size,
                                                           self.attribute_embedder_as_bias)
        jobs_corrupted = self.add_noise(jobs, masks)

        jobs_corrupted_embedded = self.word_embedder(jobs_corrupted, return_dict=True)["last_hidden_state"]
        enc_hidden = self.encoder.init_hidden()
        encoder_outputs, enc_hidden = self.encoder.forward(jobs_corrupted_embedded, masks, enc_hidden)

        previous_token = attributes_as_first_token
        prev_hidden = self.decoder.init_hidden()
        decoded_tokens = []
        for di in range(self.max_len - 2):
            resized_in_token = self.decoder.lin_att_in_para(previous_token)
            tmp = torch.bmm(encoder_outputs, resized_in_token.transpose(-1, 1))
            attn_weights = masked_softmax(tmp, masks, self.max_len)
            attn_applied = torch.einsum("beh,bed->bh", encoder_outputs, attn_weights)

            output = torch.cat((previous_token, attn_applied.unsqueeze(1)), -1)

            output_lstm, hidden = self.decoder.LSTM(output, prev_hidden)
            output_lin = self.decoder.lin_lstm_out(output_lstm)
            decoder_output = output_lin + attributes_as_bias
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
        if self.hp.para == "True":
            loss_dae_total, decoded_tokens = self.forward_para(indices, masks, exp_indices, ind_indices, batch_nb)
        else:
            loss_dae_total, decoded_tokens = self.forward(indices, masks, exp_indices, ind_indices, batch_nb)
        loss_dae_per_sample = loss_dae_total / sample_len
        self.log('train_loss_ep', loss_dae_per_sample, on_step=False, on_epoch=True)
        self.log('train_loss_st', loss_dae_per_sample, on_step=True, on_epoch=False)
        self.log('train_full_loss', loss_dae_total)
        if batch_nb % 200 == 0:
            self.save_at_step(batch_nb)
        return {"loss": loss_dae_per_sample}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def validation_step(self, batch, batch_nb):
        indices, masks, exp_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        sample_len = len(indices)
        if self.hp.para == "True":
            loss_dae_total, decoded_tokens = self.forward_para(indices, masks, exp_indices, ind_indices, batch_nb)
        else:
            loss_dae_total, decoded_tokens = self.forward(indices, masks, exp_indices, ind_indices, batch_nb)
        val_loss_dae_per_sample = loss_dae_total / sample_len
        self.log('val_loss', val_loss_dae_per_sample, prog_bar=True)
        self.log('full_val_loss', loss_dae_total)
        return {"val_loss": val_loss_dae_per_sample}

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
        if self.hp.para == "True":
            decoded_tokens = self.inference_para(indices, masks, exp_indices, ind_indices, batch_nb)
        else:
            decoded_tokens = self.inference(indices, masks, exp_indices, ind_indices, batch_nb)
        str_preds = self.word_tokenizer.decode(decoded_tokens[0])
        labels = self.word_tokenizer.decode(indices[0][1:])

        lab_file = os.path.join(self.datadir, self.desc + '_LABELS.txt')
        pred_file = os.path.join(self.datadir, self.desc + '_PREDS.txt')
        preds = word_tokenize(str_preds)
        labs = word_tokenize(labels.split("</s>")[0])
        eos_reached = False; num_tok = 0

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




