import pytorch_lightning as pl
import torch
import ipdb


class BiLSTMEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, pooling_window, hp):
        super().__init__()
        self.hs = hidden_size
        self.w_s = pooling_window
        self.dpo = hp.dpo
        self.max_len = hp.max_len
        self.num_layer = 2
        self.hp = hp

        self.biLSTM = torch.nn.LSTM(input_size,
                                    self.hs,
                                    dropout=self.dpo,
                                    num_layers=self.num_layer,
                                    batch_first=True,
                                    bidirectional=True)
        # for name, param in self.named_parameters():
        #     torch.nn.init.normal(param)

        # max pool is non overlapping so window_size == striding.
        # ceil_mode – If True, will use ceil instead of floor to compute the output shape.
        # This ensures that every element in the input tensor is covered by a sliding window.
        self.max_pool = torch.nn.MaxPool1d(self.w_s, self.w_s, ceil_mode=True)

    def forward(self, embedded_tokens):
        output_lstm, hidden = self.biLSTM(embedded_tokens)
        return self.max_pool(output_lstm), hidden

    # def init_hidden(self):
    #     if self.hp.b_size > 1:
    #         return (torch.zeros(self.num_layer*2, int(self.hp.b_size/self.hp.gpus), self.hs).cuda(),
    #                 torch.zeros(self.num_layer*2, int(self.hp.b_size/self.hp.gpus), self.hs).cuda())
    #     else:
    #         return (torch.zeros(self.num_layer*2, 1, self.hs).cuda(),
    #                 torch.zeros(self.num_layer*2, 1, self.hs).cuda())
