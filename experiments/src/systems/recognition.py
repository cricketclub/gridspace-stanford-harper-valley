import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import (
    AverageMeter,
    save_checkpoint as save_snapshot,
    copy_checkpoint as copy_snapshot,
)
from src.datasets.harper_valley import HarperValley
from src.models.ctc import (
    ConnectionistTemporalClassification,
    GreedyDecoder,
)
from src.models.las import (
    ListenAttendSpell,
    label_smooth_loss,
    compute_wer_for_las,
)
from src.models.mtl import MultiTaskLearning
from src.models.tasks import (
    TaskTypePredictor,
    DialogActsPredictor,
    SentimentPredictor,
    SpeakerIdPredictor,
)
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)


class CTC_System(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.create_model()

    def create_datasets(self):
        wav_maxlen = self.config.data_params.wav_maxlen
        transcript_maxlen = self.config.data_params.transcript_maxlen
        root = self.config.data_params.harpervalley_root
        train_dataset = HarperValley(
            root,
            split='train',
            add_eps_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            split_by_speaker=self.config.data_params.speaker_split,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        val_dataset = HarperValley(
            root,
            split='val',
            add_eps_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            split_by_speaker=self.config.data_params.speaker_split,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        return train_dataset, val_dataset

    def create_asr_model(self):
        asr_model = ConnectionistTemporalClassification(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            num_layers=self.config.model_params.num_layers,
            hidden_dim=self.config.model_params.hidden_dim,
            bidirectional=self.config.model_params.bidirectional,
        )
        self.asr_model = asr_model.to(self.device)
        self.embedding_dim = asr_model.embedding_dim

    def create_auxiliary_models(self):
        task_type_model = TaskTypePredictor(
            self.embedding_dim,
            self.train_dataset.task_type_num_class,
        )
        dialog_acts_model = DialogActsPredictor(
            self.embedding_dim,
            self.train_dataset.dialog_acts_num_class,
        )
        sentiment_model = SentimentPredictor(
            self.embedding_dim,
            self.train_dataset.sentiment_num_class,
        )
        self.task_type_model = task_type_model.to(self.device)
        self.dialog_acts_model = dialog_acts_model.to(self.device)
        self.sentiment_model = sentiment_model.to(self.device)

        if not self.config.data_params.speaker_split:
            speaker_id_model = SpeakerIdPredictor(
                self.embedding_dim,
                self.train_dataset.speaker_id_num_class,
            )
            self.speaker_id_model = speaker_id_model.to(self.device)

    def create_model(self):
        self.create_asr_model()
        self.create_auxiliary_models()

    def configure_optimizers(self):
        if self.config.data_params.speaker_split:
            parameters = chain(
                self.asr_model.parameters(),
                self.task_type_model.parameters(),
                self.dialog_acts_model.parameters(),
                self.sentiment_model.parameters(),
            )
        else:
            parameters = chain(
                self.asr_model.parameters(),
                self.task_type_model.parameters(),
                self.dialog_acts_model.parameters(),
                self.sentiment_model.parameters(),
                self.speaker_id_model.parameters(),
            )
        optim = torch.optim.AdamW(
            parameters,
            lr=self.config.optim_params.learning_rate,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def get_asr_loss(self, log_probs, labels, input_lengths, label_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=self.train_dataset.eps_index,
        )
        return loss

    def get_asr_decode_error(self, log_probs, input_lengths, labels, label_lengths):
        ctc_decoder = GreedyDecoder(
            blank_index=self.train_dataset.eps_index,
            space_index=-1,  # no space label
        )
        wer = ctc_decoder.get_wer(
            log_probs,
            input_lengths,
            labels,
            label_lengths,
        )
        return wer

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.asr_model(inputs, input_lengths)
        return log_probs, embedding

    def get_losses_and_metrics_for_batch(self, batch, train=True):
        indices = batch[0].to(self.device)
        char_inputs = batch[1].to(self.device)
        char_input_lengths = batch[2].to(self.device)
        char_labels = batch[3].to(self.device)
        char_label_lengths = batch[4].to(self.device)
        task_type_labels = batch[5].to(self.device)
        dialog_acts_labels = batch[6].to(self.device)
        sentiment_labels = batch[7].to(self.device)
        speaker_id_labels = batch[8].to(self.device)
        batch_size = indices.size(0)

        char_log_probs, embedding = self.forward(char_inputs, char_input_lengths,
                                                 char_labels, char_label_lengths)
        task_type_log_probs = self.task_type_model(embedding)
        dialog_acts_probs = self.dialog_acts_model(embedding)
        sentiment_log_probs = self.sentiment_model(embedding)

        asr_loss = self.get_asr_loss(
            char_log_probs,
            char_labels,
            char_input_lengths,
            char_label_lengths,
        )
        task_type_loss = self.task_type_model.get_loss(
            task_type_log_probs,
            task_type_labels,
        )
        dialog_acts_loss = self.dialog_acts_model.get_loss(
            dialog_acts_probs,
            dialog_acts_labels,
        )
        sentiment_loss = self.sentiment_model.get_loss(
            sentiment_log_probs,
            sentiment_labels,
        )
        loss = (
            asr_loss * self.config.loss_params.asr_weight + 
            task_type_loss * self.config.loss_params.task_type_weight + 
            dialog_acts_loss * self.config.loss_params.dialog_acts_weight + 
            sentiment_loss * self.config.loss_params.sentiment_weight
        )

        if not self.config.data_params.speaker_split:
            speaker_id_log_probs = self.speaker_id_model(embedding)
            speaker_id_loss = self.speaker_id_model.get_loss(
                speaker_id_log_probs,
                speaker_id_labels,
            )
            loss = loss + speaker_id_loss * self.config.loss_params.speaker_id_weight

        with torch.no_grad():
            asr_wer = self.get_asr_decode_error(
                char_log_probs,
                char_input_lengths,
                char_labels,
                char_label_lengths,
            )
            task_type_preds = torch.argmax(task_type_log_probs, dim=1)
            num_task_type_correct = (task_type_preds == task_type_labels).sum().item()
            num_task_type_total = batch_size

            dialog_acts_preds = torch.round(dialog_acts_probs)
            num_dialog_acts_correct = (dialog_acts_preds == dialog_acts_labels).sum().item()
            num_dialog_acts_total = batch_size
            dialog_acts_preds_npy = dialog_acts_preds.cpu().numpy()     # batch_size x num_dialog_actions
            dialog_acts_labels_npy  = dialog_acts_labels.cpu().numpy()  # batch_size x num_dialog_actions

            sentiment_preds = torch.argmax(sentiment_log_probs, dim=1)
            sentiment_labels = torch.argmax(sentiment_labels, dim=1)
            num_sentiment_correct = (sentiment_preds == sentiment_labels).sum().item()
            num_sentiment_total = batch_size

            if not self.config.data_params.speaker_split:
                speaker_id_preds = torch.argmax(speaker_id_log_probs, dim=1)
                num_speaker_id_correct = (speaker_id_preds == speaker_id_labels).sum().item()
                num_speaker_id_total = batch_size

            prefix = 'train' if train else 'val'

            metrics = {
                # -- all losses 
                f'{prefix}_asr_loss': asr_loss,
                f'{prefix}_task_type_loss': task_type_loss,
                f'{prefix}_dialog_acts_loss': dialog_acts_loss,
                f'{prefix}_sentiment_loss': sentiment_loss,
                # -- asr metrics
                f'{prefix}_asr_wer': asr_wer,
                # -- task metrics
                f'{prefix}_num_task_type_correct': num_task_type_correct,
                f'{prefix}_num_task_type_total': num_task_type_total,
                # -- dialog_acts metrics
                f'{prefix}_num_dialog_acts_correct': num_dialog_acts_correct,
                f'{prefix}_num_dialog_acts_total': num_dialog_acts_total,
                # -- sentiment metrics
                f'{prefix}_num_sentiment_correct': num_sentiment_correct,
                f'{prefix}_num_sentiment_total': num_sentiment_total,
            }
            if not self.config.data_params.speaker_split:
                # -- speaker id metrics
                metrics[f'{prefix}_speaker_id_loss'] = speaker_id_loss
                metrics[f'{prefix}_num_speaker_id_correct'] = num_speaker_id_correct
                metrics[f'{prefix}_num_speaker_id_total'] = num_speaker_id_total
            
            if not train:
                # ---- store the full numpy so that we can
                metrics['dialog_acts_preds_npy'] = dialog_acts_preds_npy
                metrics['dialog_acts_labels_npy'] = dialog_acts_labels_npy

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=True)
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=False)
        metrics['val_loss'] = loss
        return OrderedDict(metrics)

    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            if key not in ['dialog_acts_preds_npy', 'dialog_acts_labels_npy']:
                metrics[key] = torch.tensor([elem[key]
                                            for elem in outputs]).float().mean()
        metric_keys = ['task_type', 'dialog_acts', 'sentiment']
        if not self.config.data_params.speaker_split:
            metric_keys.append('speaker_id')

        for name in metric_keys:
            num_correct = sum([out[f'val_num_{name}_correct'] for out in outputs])
            num_total = sum([out[f'val_num_{name}_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            metrics[f'val_{name}_acc'] = val_acc

        dialog_acts_preds_npy = np.concatenate([out['dialog_acts_preds_npy'] 
                                               for out in outputs], axis=0)
        dialog_acts_labels_npy = np.concatenate([out['dialog_acts_labels_npy'] 
                                                for out in outputs], axis=0)
        f1_scores = []
        for i in range(self.train_dataset.dialog_acts_num_class):
            f1_score_i = f1_score(dialog_acts_labels_npy[:, i], dialog_acts_preds_npy[:, i])
            f1_scores.append(f1_score_i)
        f1_scores = np.array(f1_scores)
        metrics[f'val_dialog_acts_f1'] = np.mean(f1_scores)

        return {'val_loss': metrics['val_loss'], 'log': metrics}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class LAS_System(CTC_System):

    def create_datasets(self):
        root = self.config.data_params.harpervalley_root
        wav_maxlen = self.config.data_params.wav_maxlen
        transcript_maxlen = self.config.data_params.transcript_maxlen
        train_dataset = HarperValley(
            root,
            split='train',
            add_sos_and_eos_tok=True, 
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        val_dataset = HarperValley(
            root,
            split='val',
            add_sos_and_eos_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        return train_dataset, val_dataset

    def create_asr_model(self):
        asr_model = ListenAttendSpell(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            self.train_dataset.transcript_maxlen,
            listener_hidden_dim=self.config.model_params.listener_hidden_dim,
            num_pyramid_layers=self.config.model_params.num_pyramid_layers,
            dropout=self.config.model_params.dropout,
            speller_hidden_dim=self.config.model_params.speller_hidden_dim,
            speller_num_layers=self.config.model_params.speller_num_layers,
            mlp_hidden_dim=self.config.model_params.mlp_hidden_dim,
            multi_head=self.config.model_params.multi_head,
            sos_index=self.train_dataset.sos_index,
            sample_decode=self.config.model_params.sample_decode,
        )
        self.asr_model = asr_model.to(self.device)
        self.embedding_dim = asr_model.embedding_dim

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.asr_model(
            inputs,
            ground_truth=labels,
            teacher_force_prob=self.config.loss_params.teacher_force_prob,
        )
        return log_probs, embedding

    def get_asr_loss(self, log_probs, labels, input_lengths, label_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            labels,
            self.train_dataset.num_labels,
            pad_index=self.train_dataset.pad_index,
            label_smooth=self.config.loss_params.label_smooth,
        )
        return loss

    def get_asr_decode_error(self, log_probs, input_lengths, labels, label_lengths):
        pred_labels = torch.argmax(log_probs, dim=2)
        wer = compute_wer_for_las(
            pred_labels, 
            labels, 
            sos_index=self.train_dataset.sos_index,
            eos_index=self.train_dataset.eos_index,
        )
        return wer


class MTL_System(LAS_System):

    def create_datasets(self):
        root = self.config.data_params.harpervalley_root
        wav_maxlen = self.config.data_params.wav_maxlen
        transcript_maxlen = self.config.data_params.transcript_maxlen
        train_dataset = HarperValley(
            root,
            split='train',
            add_eps_tok=True,
            add_sos_and_eos_tok=True, 
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        val_dataset = HarperValley(
            root,
            split='val',
            add_eps_tok=True,
            add_sos_and_eos_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        return train_dataset, val_dataset

    def create_asr_model(self):
        asr_model = MultiTaskLearning(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            self.train_dataset.transcript_maxlen,
            listener_hidden_dim=self.config.model_params.listener_hidden_dim,
            num_pyramid_layers=self.config.model_params.num_pyramid_layers,
            dropout=self.config.model_params.dropout,
            speller_hidden_dim=self.config.model_params.speller_hidden_dim,
            speller_num_layers=self.config.model_params.speller_num_layers,
            mlp_hidden_dim=self.config.model_params.mlp_hidden_dim,
            multi_head=self.config.model_params.multi_head,
            sos_index=self.train_dataset.sos_index,
            sample_decode=self.config.model_params.sample_decode,
        )
        self.asr_model = asr_model.to(self.device)
        self.embedding_dim = asr_model.embedding_dim

    def forward(self, inputs, input_lengths, labels, label_lengths):
        ctc_log_probs, las_log_probs, embedding = self.asr_model(
            inputs,
            ground_truth=labels,
            teacher_force_prob=self.config.loss_params.teacher_force_prob,
        )
        return (ctc_log_probs, las_log_probs), embedding

    def get_asr_loss(self, log_probs, labels, input_lengths, label_lengths):
        (ctc_log_probs, las_log_probs) = log_probs
        ctc_loss, las_loss = self.asr_model.get_loss(
            ctc_log_probs,
            las_log_probs,
            labels,
            self.train_dataset.num_labels,
            input_lengths,
            label_lengths,
            pad_index=self.train_dataset.pad_index,
            blank_index=self.train_dataset.eps_index,
            label_smooth=self.config.loss_params.label_smooth,
        )
        loss = self.config.loss_params.alpha * ctc_loss + \
               (1 - self.config.loss_params.alpha) * las_loss
        return loss

    def get_asr_decode_error(self, log_probs, input_lengths, labels, label_lengths):
        # only decode using LAS: https://arxiv.org/pdf/1609.06773.pdf
        (_, las_log_probs) = log_probs
        pred_labels = torch.argmax(las_log_probs, dim=2)
        wer = compute_wer_for_las(
            pred_labels, 
            labels, 
            sos_index=self.train_dataset.sos_index,
            eos_index=self.train_dataset.eos_index,
            filler_index=0,
        )
        return wer


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        num_workers=config.data_loader_workers,
    )
    return loader
