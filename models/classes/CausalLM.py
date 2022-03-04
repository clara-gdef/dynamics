import math

import ipdb
import torch
import os
import pytorch_lightning as pl
import pickle as pkl
import numpy as np
import random
from line_profiler import LineProfiler
from utils.models import indices_to_words, get_metrics, get_metrics_at_k
from transformers import CamembertForCausalLM, CamembertTokenizer


class CausalLM(pl.LightningModule):
    def __init__(self, datadir, desc, vocab_size, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.test_preds = []
        self.test_labels = []
        with open(os.path.join(self.datadir, "vocab_dict.pkl"), "rb") as f:
            self.vocab_dict = pkl.load(f)
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.ppl_model = CamembertForCausalLM.from_pretrained('camembert-base')
        self.ppl_model.is_decoder = True

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        outputs = self.ppl_model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda(), labels=inputs["input_ids"].cuda())
        return outputs, inputs["input_ids"], inputs["attention_mask"]

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
        sentences = batch[0]
        outputs, tokenized_sentence, mask = self.forward(sentences)
        ce = get_ce(outputs.logits, tokenized_sentence.cuda(), mask.cuda())
        try:
            ppl = 2 ** (ce.item() / math.log(2))
        except OverflowError:
            ppl = math.inf
        self.log('train_loss', outputs.loss)
        self.log('train_ppl', ppl)
        return {'loss': outputs.loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def validation_step(self, batch, batch_nb):
        sentences = batch[0]
        outputs, tokenized_sentence, mask = self.forward(sentences)
        ce = get_ce(outputs.logits, tokenized_sentence.cuda(), mask.cuda())
        try:
            ppl = 2 ** (ce.item() / math.log(2))
        except OverflowError:
            ppl = math.inf
        self.log('val_loss', outputs.loss)
        self.log('val_ppl', ppl)
        return {'val_loss': outputs.loss}

    def validation_step_end(self, validation_step_outputs):
        return {'val_loss': validation_step_outputs['val_loss'].sum()}

    def test_step(self, batch, batch_nb):
        indices, att_indices = batch[0], batch[1]
        sentences = indices_to_words(indices, self.vocab_dict)
        self.test_labels.append(att_indices)
        self.test_preds.append(self.inference(sentences, att_indices))

    def test_epoch_end(self, outputs):
        att_labels = torch.stack(self.test_labels).cpu().numpy()
        preds_on_cpu = np.array(torch.argsort(torch.stack(self.test_preds), dim=-1, descending=True).squeeze(1).cpu())
        metrics = get_metrics(preds_on_cpu[:, 0], att_labels, self.num_classes,
                              self.att_type)
        metrics_atk = get_metrics_at_k(preds_on_cpu[:, :self.k], att_labels,
                                       self.num_classes,
                                       f"{self.att_type}@{self.k} ")
        return {**metrics, **metrics_atk}


def get_ce(logits, labels, mask):
    tmp2 = torch.nn.functional.cross_entropy(logits.transpose(-1, 1)[:, :, :-1], labels[:, 1:],
                                             reduction='none')
    return (tmp2 * mask[:, 1:]).sum() / mask[:, 1:].sum()
