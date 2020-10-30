import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils import edit_distance


class ListenAttendSpell(nn.Module):

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
        super().__init__()

        self.listener = Listener(
            input_dim,
            listener_hidden_dim,
            num_pyramid_layers=num_pyramid_layers,
            dropout=dropout,
        )
        self.speller = Speller(
            num_class,
            label_maxlen,
            speller_hidden_dim,
            listener_hidden_dim,
            mlp_hidden_dim,
            num_layers=speller_num_layers,
            multi_head=multi_head,
            sos_index=sos_index,
            sample_decode=sample_decode,
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
            ground_truth=None,
            teacher_force_prob=0.9,
        ):
        listener_feats, (listener_h, listener_c) = self.listener(inputs)
        listener_hc = self.combine_h_and_c(listener_h, listener_c)
        log_probs = self.speller(
            listener_feats,
            ground_truth=ground_truth,
            teacher_force_prob=teacher_force_prob,
        )
        return log_probs, listener_hc

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
            loss = F.nll_loss(
                log_probs.view(batch_size * labels_maxlen, -1),
                labels.long().view(batch_size * labels_maxlen),
                ignore_index=pad_index,
            )
        else:
            loss = label_smooth_loss(
                log_probs,
                labels.float(),
                num_labels,
                smooth_param=label_smooth,
            )

        return loss


class Listener(nn.Module):
    """
    Stack three layers of PyramidLSTMLayers to reduce
    resolution 8 times.
    """
    
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_pyramid_layers=3,
            dropout=0.,
        ):
        super().__init__()
        self.rnn_layer0 = PyramidLSTMLayer(
            input_dim, 
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=dropout,
        )

        for i in range(1, num_pyramid_layers):
            setattr(
                self, 
                f'rnn_layer{i}',
                PyramidLSTMLayer(
                    hidden_dim * 2, 
                    hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    dropout=dropout,
                ),
            )
        
        self.num_pyramid_layers = num_pyramid_layers

    def forward(self, inputs):
        outputs, hiddens = self.rnn_layer0(inputs)
        for i in range(1, self.num_pyramid_layers):
            outputs, hiddens = getattr(
                self,
                f'rnn_layer{i}',
            )(outputs)
        
        return outputs, hiddens


class Speller(nn.Module):

    def __init__(
            self,
            num_labels,
            label_maxlen,
            speller_hidden_dim,
            listener_hidden_dim,
            mlp_hidden_dim,
            num_layers=1,
            multi_head=1,
            sos_index=0,
            sample_decode=False,
        ):
        super().__init__()

        self.rnn = nn.LSTM(
            num_labels + speller_hidden_dim,
            speller_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = AttentionLayer(
            listener_hidden_dim * 2, 
            mlp_hidden_dim, 
            multi_head=multi_head,
        )
        self.fc_out = nn.Linear(speller_hidden_dim*2, num_labels)
        self.num_labels = num_labels
        self.label_maxlen = label_maxlen
        self.sample_decode = sample_decode
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
                with torch.no_grad():
                    if self.sample_decode:
                        probs = torch.exp(log_probs)
                        sampled_tok = Categorical(probs).sample()
                    else:  # Pick max probability
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

    def __init__(
            self, 
            input_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=0.,
        ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim * 2, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    def forward(self, inputs):
        batch_size, maxlen, input_dim = inputs.size()
       
        # reduce time resolution?
        inputs = inputs.contiguous().view(
            batch_size,
            maxlen // 2, 
            input_dim * 2,
        )
        
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


def compute_wer_for_las(
        pred_labels, 
        true_labels,
        sos_index=0,
        eos_index=1,
        filler_index=0,
    ):
    """
    Compute WER for LAS. We ignore start and end tokens.
    """
    wers = []
    assert pred_labels.size(0) == true_labels.size(0)

    for i in range(pred_labels.size(0)):
        pred = pred_labels[i]
        true = true_labels[i]

        compressed_true = [tok.item() for tok in true 
                           if tok.item() != sos_index and 
                           tok.item() != eos_index and 
                           tok.item() != filler_index]
        compressed_pred = []
        for tok in pred:
            if tok.item() == sos_index:
                continue
            if tok.item() == filler_index:
                continue
            if tok.item() == eos_index:
                break
            compressed_pred.append(tok.item())

        if len(compressed_pred) > 0:
            dist = edit_distance(compressed_pred, compressed_true)
            wer = dist[-1, -1] / float(dist.shape[1])
            wers.append(wer)

    wers = np.array(wers)
    return float(np.mean(wers))
