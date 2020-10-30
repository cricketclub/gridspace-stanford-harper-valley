import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.models.las import ListenAttendSpell
from src.models.ctc import CTCDecoder


class MultiTaskLearning(ListenAttendSpell):

    def __init__(
            self,
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=256,
            num_pyramid_layers=3,
            dropout=0,
            speller_hidden_dim=512,
            speller_num_layers=1,
            mlp_hidden_dim=128,
            multi_head=1,
            sos_index=0,
            sample_decode=False,
        ):
        super().__init__(
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=listener_hidden_dim,
            num_pyramid_layers=num_pyramid_layers,
            dropout=dropout,
            speller_hidden_dim=speller_hidden_dim,
            speller_num_layers=speller_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            multi_head=multi_head,
            sos_index=sos_index,
            sample_decode=sample_decode,
        )
        self.ctc_decoder = CTCDecoder(
            listener_hidden_dim * 2,
            num_class,
        )
        self.num_pyramid_layers = num_pyramid_layers
        self.embedding_dim = listener_hidden_dim * 4

    def forward(
            self,
            inputs,
            ground_truth=None,
            teacher_force_prob=0.9,
        ):
        listener_feats, (listener_h, listener_c) = self.listener(inputs)
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
            targets,
            num_labels,
            input_lengths,
            target_lengths,
            pad_index=0,
            blank_index=0,
            label_smooth=0.1,
        ):
        ctc_loss = self.ctc_decoder.get_loss(
            ctc_log_probs,
            targets,
            # pyramid encode cuts timesteps in 1/2 each way
            input_lengths // (2**self.num_pyramid_layers),
            target_lengths,
            blank=blank_index,
        )
        las_loss = super().get_loss(
            las_log_probs,
            targets,
            num_labels,
            pad_index=pad_index,
            label_smooth=label_smooth,
        )

        return ctc_loss, las_loss
