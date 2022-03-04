import ipdb
import torch
import os
import pytorch_lightning as pl
import pickle as pkl
import numpy as np
import random
from line_profiler import LineProfiler
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.classes import LSTMAttentionDecoderToy
from utils.models import indices_to_words, get_metrics, get_metrics_at_k, masked_softmax, compute_bleu_score, \
    prettify_bleu_score
from transformers import CamembertTokenizer, CamembertModel


class ToyTextGenerator(pl.LightningModule):
    def __init__(self, datadir, desc, vocab_size, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.emb_dim = 768
        self.test_preds = []
        self.test_labels = []
        with open(os.path.join(self.datadir, "vocab_dict.pkl"), "rb") as f:
            self.vocab_dict = pkl.load(f)
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.encoder = CamembertModel.from_pretrained('camembert-base')
        self.decoder = LSTMAttentionDecoderToy(self.emb_dim, self.hp.dec_hs, self.emb_dim, self.voc_size, hp)

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        max_seq_len = input_tokenized.shape[-1]
        encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']
        input_embedded = self.encoder.embeddings(input_tokenized)
        resized_conditioning = self.decoder.lin_att_in_para(input_embedded[:, :-1, :])
        tmp = torch.bmm(encoder_outputs, resized_conditioning.transpose(-1, 1))
        attn_weights = masked_softmax(tmp, mask, max_seq_len)
        attn_applied = torch.einsum("beh,bed->bdh", encoder_outputs, attn_weights)
        decoder_inputs = torch.cat((input_embedded[:, :-1, :], attn_applied), -1)
        predicted_tokens = self.decoder(decoder_inputs, torch.zeros(max_seq_len-1, self.voc_size).cuda())
        return predicted_tokens, inputs["input_ids"], inputs["attention_mask"]

    def inference(self, sentences):
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        outputs = self.ppl_model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda())
        return outputs.logits, inputs["input_ids"], inputs["attention_mask"]

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr)

    def training_step(self, batch, batch_nb):
        sentences = batch
        outputs, tokenized_sentence, mask = self.forward(sentences)
        sample_len = sum(sum(mask))
        loss = torch.nn.functional.cross_entropy(outputs.transpose(-1, 1),
                                                 tokenized_sentence[:, 1:].cuda(), reduction="sum", ignore_index=1)
        if torch.isnan(loss):
            ipdb.set_trace()
        self.log('train_loss', loss / sample_len)

        return {'loss': loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def validation_step(self, batch, batch_nb):
        sentences = batch
        outputs, tokenized_sentence, mask = self.forward(sentences)
        sample_len = sum(sum(mask))
        val_loss = torch.nn.functional.cross_entropy(outputs.transpose(-1, 1),
                                                 tokenized_sentence[:, 1:].cuda(), reduction="sum", ignore_index=1)
        if torch.isnan(val_loss):
            ipdb.set_trace()
        if batch_nb == 0:
            out_tokens = torch.argmax(outputs, dim=-1)
            print("PRED")
            print(self.tokenizer.decode(out_tokens[0], skip_special_tokens=True))
        self.log('val_loss', val_loss / sample_len)
        return {'val_loss': val_loss}

    def validation_step_end(self, validation_step_outputs):
        return {'val_loss': validation_step_outputs['val_loss'].sum()}

    def test_step(self, batch, batch_nb):
        sentences = batch
        outputs, tokenized_sentence, mask = self.forward(sentences)
        self.test_labels.append(sentences)
        self.test_preds.append(pad_sequence(torch.argmax(outputs, dim=-1), batch_first=True))

    def test_epoch_end(self, outputs):
        stacked_outs = torch.stack(self.test_preds)
        labels = self.test_labels
        bleu = prettify_bleu_score((self.get_bleu_score(stacked_outs, labels)))
        return bleu

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
            str_preds = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
            labels = test_labels[num][0]
            with open(pred_file, "a+") as f:
                f.write(str_preds + "\n")

            with open(lab_file, "a+") as f:
                f.write(labels + "\n")
            # get score
        print(f"Computing BLEU score for model {self.desc}")
        print(f"Pred file: {pred_file}")
        return compute_bleu_score(pred_file, lab_file)