import pytorch_lightning as pl
import torch
import math
import ipdb


class LSTMAttentionDecoder(pl.LightningModule):
    def __init__(self, hidden_size, emb_dim, voc_size, hp):
        super().__init__()
        self.hp = hp
        self.hs = hidden_size
        self.num_layer = self.hp.num_layer
        self.dpo = hp.dpo
        self.max_len = hp.max_len
        self.emb_dim = emb_dim
        if self.hp.feed_z == "True":
            self.input_size = math.ceil(self.emb_dim / self.hp.pooling_window) * 2
        else:
            self.input_size = self.emb_dim

        self.LSTM = torch.nn.LSTM(self.input_size + self.emb_dim,
                                  self.hs,
                                  dropout=self.dpo,
                                  num_layers=self.num_layer,
                                  batch_first=True,
                                  bidirectional=False)
        self.lin_att_in_para = torch.nn.Linear(emb_dim, math.ceil(self.emb_dim / self.hp.pooling_window))
        self.lin_lstm_out = torch.nn.Linear(self.hs, voc_size)

    def forward(self, embedded_conditionning, emb_attributes, hs):
        output_lstm, hidden = self.LSTM(embedded_conditionning, hs)
        output_lin = self.lin_lstm_out(output_lstm)
        output_all = output_lin + emb_attributes
        return output_all, hidden

    def masked_softmax(self, logits, mask):
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        mask = mask.type(dtype=logits.dtype)
        weights = torch.exp(logits) * mask.cuda()
        if torch.isnan(weights).any():
            ipdb.set_trace()
        denominator = torch.finfo(torch.get_default_dtype()).min + torch.sum(weights, dim=1, keepdim=True)
        return weights / denominator
    #
    # def init_hidden(self):
    #     return (torch.zeros(self.num_layer, int(self.hp.b_size/self.hp.gpus), self.hs).cuda(),
    #             torch.zeros(self.num_layer, int(self.hp.b_size/self.hp.gpus), self.hs).cuda())
