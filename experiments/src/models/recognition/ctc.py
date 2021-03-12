import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils import edit_distance


class CTCEncoderDecoder(nn.Module):
    """
    Connectionist Temporal Classification Model. Use a bidirectional
    LSTM as the encoder and a linear layer to decode to class probabilities.

    Args:
        input_dim: integer
                    number of input features
        num_class: integer
                    size of transcription vocabulary
        num_layers: integer (default: 2)
                    number of layers in encoder LSTM
        hidden_dim: integer (default: 128)
                    number of hidden dimensions for encoder LSTM
        bidirectional: boolean (default: True)
                        is the encoder LSTM bidirectional?
    """

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
        self.dropout = nn.Dropout()
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
            input_lengths.cpu(), 
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
        logits = self.word_fc(self.dropout(outputs))
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

    def decode(
            self, 
            log_probs, 
            input_lengths, 
            labels, 
            label_lengths,
            sos_index,
            eos_index, 
            pad_index, 
            eps_index,
        ):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        for i in range(batch_size):
            hypotheses_i = self.ctc_collapse(
                decoded[i], 
                input_lengths[i].item(),
                blank_index=eps_index,
            )
            hypotheses.append(hypotheses_i)

        hypothesis_lengths = input_lengths.cpu().numpy().tolist()
        if labels is None: # Run at inference time.
            references, reference_lengths = None, None
        else:
            references = labels.cpu().numpy().tolist()
            reference_lengths = label_lengths.cpu().numpy().tolist()

        return hypotheses, hypothesis_lengths, references, reference_lengths

    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i-1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result

