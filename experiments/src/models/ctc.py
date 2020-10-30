import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils import edit_distance, levenshtein


class ConnectionistTemporalClassification(nn.Module):

    def __init__(
            self,
            input_dim,
            num_class,
            num_layers=2,
            hidden_dim=128,
            bidirectional=True,
        ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.word_fc = nn.Linear(hidden_dim * 2, num_class)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim * num_layers * 2 * \
                             (2 if bidirectional else 1)

    def forward(self, inputs, input_lengths):
        batch_size, maxlen, _ = inputs.size()
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        # outputs : batch_size x maxlen x hidden_dim
        # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
        # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
        outputs, (rnn_h, rnn_c) = self.rnn(inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=maxlen,
        )
        # outputs : batch_size * maxlen x hidden_dim*2
        outputs = outputs.view(-1, self.hidden_dim*2)
        # logits : batch_size * maxlen x num_class
        logits = self.word_fc(F.dropout(outputs, p=0.5))
        logits = logits.view(batch_size, maxlen, self.num_class)
        log_probs = F.log_softmax(logits, dim=2)
        # this embedding will be used for other transfer tasks
        embedding = self.combine_h_and_c(rnn_h, rnn_c)

        return log_probs, embedding

    def combine_h_and_c(self, h, c):
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)

    def get_loss(
            self,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
        ):
        log_probs = log_probs.permute(1, 0, 2)
        ctc_loss = F.ctc_loss(
            log_probs.contiguous(), 
            targets.long(), 
            input_lengths.long(), 
            target_lengths.long(), 
            blank=blank,
            zero_infinity=True,
        )
        return ctc_loss


class CTCDecoder(nn.Module):
    """Used for the joint CTC-Attention model."""

    def __init__(self, listener_hidden_dim, num_class):
        super().__init__()
        self.fc = nn.Linear(listener_hidden_dim, num_class)
        self.listener_hidden_dim = listener_hidden_dim
        self.num_class = num_class

    def forward(self, listener_outputs):
        batch_size, maxlen, _ = listener_outputs.size()
        logits = self.fc(F.dropout(listener_outputs, p=0.5))
        logits = logits.view(batch_size, maxlen, self.num_class)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def get_loss(
            self,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
        ):
        log_probs = log_probs.permute(1, 0, 2)
        loss = F.ctc_loss(
            log_probs.contiguous(), 
            targets.long(), 
            input_lengths.long(), 
            target_lengths.long(), 
            blank=blank,
            zero_infinity=True,
        )
        return loss


class GreedyDecoder(object):

    def __init__(self, space_index=-1, blank_index=0):
        self.blank_index = blank_index
        self.space_index = space_index  # -1 means space index is not used.

    def decode(self, log_probs):
        # log_probs : batch_size x maxlen x labels
        return torch.argmax(log_probs, dim=2)

    def _process_sequence(self, seqs, seq_lens, remove_rep=False):
        result = []
        assert seqs.size(0) == len(seq_lens)
        for i in range(seqs.size(0)):
            results_i = self._process_sequence_i(
                seqs[i], seq_lens[i].item(), remove_rep)
            result.append(results_i)
        return result

    def _process_sequence_i(self, seq, seq_len, remove_rep=False):
        result = []

        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != self.blank_index:
                if remove_rep and i != 0 and tok.item() == seq[i-1].item():  # duplicate
                    pass
                else:
                    result.append(tok.item())

        if not result: return result

        if self.space_index > 0:
            while True:
                if not result: break

                if result[0] == self.space_index: 
                    result.pop(0)
                elif result[-1] == self.space_index:
                    result.pop()
                else:
                    break
        
        return result

    def _process_labels(self, labels, label_lens):
        result = []
        assert labels.size(0) == len(label_lens)
        for i in range(labels.size(0)):
            results_i = labels[i][:label_lens[i].item()]
            results_i = list(results_i.cpu().numpy())
            result.append(results_i)
        return result

    def get_wer(self, log_probs, input_lengths, labels, label_lengths):
        decoded = self.decode(log_probs)
        decoded = self._process_sequence(
            decoded.long(), 
            input_lengths.long(), 
            remove_rep=True,
        )
        labels = self._process_labels(labels.long(), label_lengths.long())
        dist = 0
        batch_size = len(decoded)
        for i in range(batch_size):
            true_len = float(label_lengths[i].item())
            if len(decoded[i]) > 0: 
                dist_i = edit_distance(decoded[i], labels[i])
                dist += (dist_i[-1, -1] / float(true_len))
        wer = dist / float(batch_size)
        return wer

