import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Categorical

from src.utils import edit_distance


class LASEncoderDecoder(nn.Module):

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
        super().__init__()

        self.listener = Listener(
            input_dim,
            hidden_dim=listener_hidden_dim,
            num_pyramid_layers=listener_num_layers,
            bidirectional=listener_bidirectional,
        )
        self.speller = Speller(
            num_class,
            label_maxlen,
            listener_hidden_dim,
            mlp_hidden_dim,
            num_layers=speller_num_layers,
            multi_head=multi_head,
            sos_index=sos_index,
        )
        self.embedding_dim = listener_hidden_dim * 4

    def combine_h_and_c(self, h, c):
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)

    def forward(
            self,
            inputs,
            input_lengths,
            ground_truth=None,
            teacher_force_prob=0.9,
        ):
        listener_feats, (listener_h, listener_c) = self.listener(
            inputs, input_lengths)
        embedding = self.combine_h_and_c(listener_h, listener_c)
        log_probs = self.speller(
            listener_feats,
            ground_truth=ground_truth,
            teacher_force_prob=teacher_force_prob,
        )
        return log_probs, embedding

    def get_loss(
            self,
            log_probs,
            labels,
            num_labels,
            pad_index=0,
            label_smooth=0.1,
        ):
        batch_size = log_probs.size(0)
        labels_maxlen = labels.size(1)

        if label_smooth == 0.0:
            log_probs = log_probs.view(batch_size * labels_maxlen, -1)
            labels = labels.long().view(batch_size * labels_maxlen)
            loss = F.nll_loss(log_probs, labels, ignore_index=pad_index)
        else:
            loss = label_smooth_loss(
                log_probs, labels.float(), num_labels, smooth_param=label_smooth)

        return loss

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
        hypothesis_lengths = []
        references = []
        reference_lengths = []
        for i in range(batch_size):
            decoded_i = decoded[i]
            hypothesis_i = []
            for tok in decoded_i:
                if tok.item() == sos_index:
                    continue
                if tok.item() == pad_index:
                    continue
                if tok.item() == eos_index:
                    # once we reach an EOS token, we are done generating.
                    break
                hypothesis_i.append(tok.item())
            hypotheses.append(hypothesis_i)
            hypothesis_lengths.append(len(hypothesis_i))

            if labels is not None:
                reference_i = [tok.item() for tok in labels[i]
                                if tok.item() != sos_index and 
                                tok.item() != eos_index and 
                                tok.item() != pad_index]
                references.append(reference_i)
                reference_lengths.append(len(reference_i))
        
        if labels is None: # Run at inference time.
            references, reference_lengths = None, None

        return hypotheses, hypothesis_lengths, references, reference_lengths


class Listener(nn.Module):
    """Listener (encoder for LAS model). Use a bidirectional 
    LSTM as the encoder. 

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
            hidden_dim=128,
            num_pyramid_layers=2,
            bidirectional=True,
            dropout_rate=0.,
        ):
        super().__init__()
        self.rnn_layer0 = PyramidLSTMLayer(
            input_dim, hidden_dim, num_layers=1, bidirectional=bidirectional, dropout=dropout_rate)
        for i in range(1, num_pyramid_layers):
            setattr(
                self, f'rnn_layer{i}',
                PyramidLSTMLayer(
                    hidden_dim * 2, hidden_dim, num_layers=1,
                    bidirectional=bidirectional, dropout=dropout_rate),
            )
        self.num_pyramid_layers = num_pyramid_layers

    def forward(self, inputs, input_lengths):
        outputs, hiddens = self.rnn_layer0(inputs)
        for i in range(1, self.num_pyramid_layers):
            outputs, hiddens = getattr(self, f'rnn_layer{i}')(outputs)
        return outputs, hiddens


class Speller(nn.Module):

    def __init__(
            self,
            num_labels,
            label_maxlen,
            listener_hidden_dim,
            mlp_hidden_dim,
            num_layers=1,
            multi_head=1,
            sos_index=0,
        ):
        super().__init__()
        speller_hidden_dim = listener_hidden_dim * 2

        self.rnn = nn.LSTM(
            num_labels + speller_hidden_dim,
            speller_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = AttentionLayer(
            speller_hidden_dim,
            mlp_hidden_dim, 
            multi_head=multi_head,
        )
        self.fc_out = nn.Linear(speller_hidden_dim*2, num_labels)
        self.num_labels = num_labels
        self.label_maxlen = label_maxlen
        self.sos_index = sos_index

    def step(self, inputs, last_hiddens, listener_feats):
        outputs, cur_hiddens = self.rnn(inputs, last_hiddens)
        attention_score, context = self.attention(outputs, listener_feats)
        features = torch.cat((outputs.squeeze(1), context), dim=-1)
        logits = self.fc_out(features)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs, cur_hiddens, context, attention_score

    def forward(
            self, 
            listener_feats, 
            ground_truth=None, 
            teacher_force_prob=0.9,
        ):
        device = listener_feats.device
        if ground_truth is None: teacher_force_prob = 0
        teacher_force = np.random.random_sample() < teacher_force_prob
        
        batch_size = listener_feats.size(0)
        with torch.no_grad():
            output_toks = torch.zeros((batch_size, 1, self.num_labels), device=device)
            output_toks[:, 0, self.sos_index] = 1

        rnn_inputs = torch.cat([output_toks, listener_feats[:, 0:1, :]], dim=-1)

        hidden_state = None
        log_probs_seq = []

        if (ground_truth is None) or (not teacher_force_prob):
            max_step = int(self.label_maxlen)
        else:
            max_step = int(ground_truth.size(1))

        for step in range(max_step):
            log_probs, hidden_state, context, _ = self.step(
                rnn_inputs, hidden_state, listener_feats)
            log_probs_seq.append(log_probs.unsqueeze(1))

            if teacher_force:
                gt_tok = ground_truth[:, step:step+1].float()
                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(gt_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)
            else:
                # Pick max probability
                output_tok = torch.zeros_like(log_probs)
                sampled_tok = log_probs.topk(1)[1]

                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(sampled_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)

            rnn_inputs = torch.cat([output_tok, context.unsqueeze(1)], dim=-1)

        # batch_size x maxlen x num_labels
        log_probs_seq = torch.cat(log_probs_seq, dim=1)

        return log_probs_seq.contiguous()


class PyramidLSTMLayer(nn.Module):
    """A Pyramid LSTM layer is a standard LSTM layer that halves the size 
    of the input in its hidden embeddings.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 bidirectional=True, dropout=0.):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim * 2, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout,
            batch_first=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    def forward(self, inputs):
        batch_size, maxlen, input_dim = inputs.size()
        # reduce time resolution?
        inputs = inputs.contiguous().view(batch_size, maxlen // 2, input_dim * 2)
        outputs, hiddens = self.rnn(inputs)
        return outputs, hiddens


class AttentionLayer(nn.Module):
    """
    Attention module as in http://www.aclweb.org/anthology/D15-1166.
    Trains an MLP to get attention weights.
    """

    def __init__(self, input_dim, hidden_dim, multi_head=1):
        super().__init__()

        self.phi = nn.Linear(input_dim, hidden_dim*multi_head)
        self.psi = nn.Linear(input_dim, hidden_dim)

        if multi_head > 1:
            self.fc_reduce = nn.Linear(input_dim*multi_head, input_dim)

        self.multi_head = multi_head
        self.hidden_dim = hidden_dim
    
    def forward(self, decoder_state, listener_feat):
        input_dim = listener_feat.size(2)
        # decoder_state: batch_size x 1 x decoder_hidden_dim
        # listener_feat: batch_size x maxlen x input_dim
        comp_decoder_state = F.relu(self.phi(decoder_state))
        comp_listener_feat = F.relu(reshape_and_apply(self.psi, listener_feat))

        if self.multi_head == 1:
            energy = torch.bmm(
                comp_decoder_state,
                comp_listener_feat.transpose(1, 2)
            ).squeeze(1)
            attention_score = [F.softmax(energy, dim=-1)]
            weights = attention_score[0].unsqueeze(2).repeat(1, 1, input_dim)
            context = torch.sum(listener_feat * weights, dim=1)
        else:
            attention_score = []
            for att_query in torch.split(
                comp_decoder_state, 
                self.hidden_dim,
                dim=-1,
            ):
                score = torch.softmax(
                    torch.bmm(
                        att_query,
                        comp_listener_feat.transpose(1, 2),
                    ).squeeze(dim=1),
                )
                attention_score.append(score)
            
            projected_src = []
            for att_s in attention_score:
                weights = att_s.unsqueeze(2).repeat(1, 1, input_dim)
                proj = torch.sum(listener_feat * weights, dim=1)
                projected_src.append(proj)
            
            context = self.fc_reduce(torch.cat(projected_src, dim=-1))

        # context is the entries of listener input weighted by attention
        return attention_score, context


def reshape_and_apply(Module, inputs):
    batch_size, maxlen, input_dim = inputs.size()
    reshaped = inputs.contiguous().view(-1, input_dim)
    outputs = Module(reshaped)
    return outputs.view(batch_size, maxlen, -1)


def label_smooth_loss(log_probs, labels, num_labels, smooth_param=0.1):
    # convert labels to one_hotted
    with torch.no_grad():
        batch_size, maxlen = labels.size()
        labels_onehotted = torch.zeros(
            (batch_size, maxlen, num_labels), 
            device=labels.device,
        ).long()
        labels_onehotted = labels_onehotted.scatter_(
            -1, labels.long().unsqueeze(2), 1)
        labels = labels_onehotted
    
    assert log_probs.size() == labels.size()
    label_lengths = torch.sum(torch.sum(labels, dim=-1), dim=-1, keepdim=True)

    smooth_labels = ((1.0 - smooth_param) * labels + (smooth_param / num_labels)) * \
        torch.sum(labels, dim=-1, keepdim=True)
   
    loss = torch.sum(smooth_labels * log_probs, dim=-1)
    loss = torch.sum(loss / label_lengths, dim=-1)
    return -loss.mean()
