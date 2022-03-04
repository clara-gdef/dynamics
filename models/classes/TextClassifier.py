import ipdb
import torch
import os
import pytorch_lightning as pl
import pickle as pkl
import numpy as np
import random
from line_profiler import LineProfiler
from nltk.tokenize import word_tokenize
from utils.models import indices_to_words, get_metrics, get_metrics_at_k
from transformers import CamembertForSequenceClassification, CamembertTokenizer


class TextClassifier(pl.LightningModule):
    def __init__(self, datadir, desc, vocab_size, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len
        self.att_type = hp.att_type
        if self.att_type == "ind":
            self.num_classes = 147
            self.k = 10
        elif self.att_type == "exp":
            self.num_classes = 5
            self.k = 2
        else:
            raise Exception("Wrong att type specified! Can be \"ind\" or \"exp\" got: " + self.att_type)
        self.test_preds = []
        self.test_labels = []
        with open(os.path.join(self.datadir, "vocab_dict.pkl"), "rb") as f:
            self.vocab_dict = pkl.load(f)

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model = CamembertForSequenceClassification.from_pretrained('camembert-base')
        # adapt model to our task
        self.model.num_labels = self.num_classes
        self.model.classifier.out_proj = torch.nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

    def forward(self, sentences, att_indices):
        self.model.classifier.num_labels = self.num_classes
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        outputs = self.model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda(), labels=att_indices.unsqueeze(0).cuda())
        return outputs.loss

    def inference(self, sentences, att_indices):
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        outputs = self.model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda())
        return outputs.logits

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr)

    def training_step(self, batch, batch_nb):
        indices, att_indices = batch[0], batch[1]
        sentences = indices_to_words(indices[:, 1:], self.vocab_dict)
        loss = self.forward(sentences, att_indices)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def validation_step(self, batch, batch_nb):
        indices, att_indices = batch[0], batch[1]
        sentences = indices_to_words(indices[:, 1:], self.vocab_dict)
        val_loss = self.forward(sentences, att_indices)
        self.log('val_loss', val_loss)
        return {'val_loss': val_loss}

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
