import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.models.recognition.las import LASEncoderDecoder


class MTLEncoderDecoder(LASEncoderDecoder):

    def __init__(
            self,
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=256,
            listener_num_layers=2,
            listener_bidirectional=True,
            speller_num_layers=1,
            mlp_hidden_dim=128,
            multi_head=1,
            sos_index=0,
        ):
        super().__init__(
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=listener_hidden_dim,
            listener_num_layers=listener_num_layers,
            listener_bidirectional=listener_bidirectional,
            speller_num_layers=speller_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            multi_head=multi_head,
            sos_index=sos_index,
        )
        self.ctc_decoder = CTCDecoder(listener_hidden_dim * 2, num_class)

    def forward(
            self,
            inputs,
            input_lengths,
            ground_truth=None,
            teacher_force_prob=0.9,
        ):
        listener_feats, (listener_h, listener_c) = self.listener(
            inputs, input_lengths)
        listener_hc = self.combine_h_and_c(listener_h, listener_c)
        las_log_probs = self.speller(
            listener_feats,
            ground_truth=ground_truth,
            teacher_force_prob=teacher_force_prob,
        )
        ctc_log_probs = self.ctc_decoder(listener_feats)
        return ctc_log_probs, las_log_probs, listener_hc

    def get_loss(
            self,
            ctc_log_probs,
            las_log_probs,
            input_lengths,
            labels,
            label_lengths,
            num_labels,
            pad_index=0,
            blank_index=0,
            label_smooth=0.1,
        ):
        ctc_loss = self.ctc_decoder.get_loss(
            ctc_log_probs,
            input_lengths // 4,
            labels,
            label_lengths,
            blank=blank_index,
        )
        las_loss = super().get_loss(
            las_log_probs,
            labels,
            num_labels,
            pad_index=pad_index,
            label_smooth=label_smooth,
        )
        return ctc_loss, las_loss


class CTCDecoder(nn.Module):
    """
    This is a small decoder (just on linear layer) that takes 
    the listener embedding from LAS and imposes a CTC 
    objective on the decoding.

    NOTE: This is only to be used for the JOint CTC-Attention model.
    """
    def __init__(self, listener_hidden_dim, num_class):
        super().__init__()
        self.fc = nn.Linear(listener_hidden_dim, num_class)
        self.dropout = nn.Dropout()
        self.listener_hidden_dim = listener_hidden_dim
        self.num_class = num_class

    def forward(self, listener_outputs):
        batch_size, maxlen, _ = listener_outputs.size()
        logits = self.fc(self.dropout(listener_outputs))
        logits = logits.view(batch_size, maxlen, self.num_class)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def get_loss(
            self,
            log_probs,
            input_lengths,
            labels,
            label_lengths,
            blank=0,
        ):
        log_probs = log_probs.permute(1, 0, 2)
        ctc_loss = F.ctc_loss(
            log_probs.contiguous(), 
            labels.long(), 
            input_lengths.long(), 
            label_lengths.long(), 
            blank=blank,
            zero_infinity=True,
        )
        return ctc_loss
