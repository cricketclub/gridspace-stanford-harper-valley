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
from src.models.recognition.ctc import CTCEncoderDecoder
from src.models.recognition.las import LASEncoderDecoder, label_smooth_loss
from src.models.recognition.mtl import MTLEncoderDecoder
from src.models.recognition.tasks import (
    TaskTypePredictor,
    DialogActsPredictor,
    SentimentPredictor,
)
from src.utils import get_cer
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)


class CTC_System(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()
        self.create_model()

    def create_datasets(self):
        wav_maxlen = self.config.data_params.wav_maxlen
        transcript_maxlen = self.config.data_params.transcript_maxlen
        root = self.config.data_params.harpervalley_root
        train_dataset = HarperValley(
            root,
            split='train',
            append_eos_token=self.config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            n_fft=self.config.data_params.n_fft,
            hop_length=self.config.data_params.hop_length,
            win_length=self.config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        val_dataset = HarperValley(
            root,
            split='val',
            append_eos_token=self.config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            n_fft=self.config.data_params.n_fft,
            hop_length=self.config.data_params.hop_length,
            win_length=self.config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        test_dataset = HarperValley(
            root,
            split='test',
            append_eos_token=self.config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            n_fft=self.config.data_params.n_fft,
            hop_length=self.config.data_params.hop_length,
            win_length=self.config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        return train_dataset, val_dataset, test_dataset

    def create_asr_model(self):
        asr_model = CTCEncoderDecoder(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            num_layers=self.config.model_params.num_layers,
            hidden_dim=self.config.model_params.hidden_dim,
            bidirectional=self.config.model_params.bidirectional,
        )
        self.asr_model = asr_model
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
        self.task_type_model = task_type_model
        self.dialog_acts_model = dialog_acts_model
        self.sentiment_model = sentiment_model

    def create_model(self):
        self.create_asr_model()
        self.create_auxiliary_models()

    def configure_optimizers(self):
        parameters = chain(
            self.asr_model.parameters(),
            self.task_type_model.parameters(),
            self.dialog_acts_model.parameters(),
            self.sentiment_model.parameters(),
        )
        optim = torch.optim.AdamW(
            parameters,
            lr=self.config.optim_params.learning_rate,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def get_asr_loss(self, log_probs, input_lengths, labels, label_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            input_lengths,
            labels,
            label_lengths,
            blank=self.train_dataset.eps_index,
        )
        return loss

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.asr_model(inputs, input_lengths)
        return log_probs, embedding

    def get_losses_and_metrics_for_batch(self, batch, split='train'):
        indices = batch['indices']
        inputs = batch['inputs']
        input_lengths = batch['input_lengths']
        labels = batch['labels']
        label_lengths = batch['label_lengths']
        task_type_labels = batch['task_types']
        dialog_acts_labels = batch['dialog_acts']
        sentiment_labels = batch['sentiments']
        batch_size = indices.size(0)

        if split == 'train':
            log_probs, embedding = self.forward(
                inputs, input_lengths, labels, label_lengths)
        else:
            log_probs, embedding = self.forward(inputs, input_lengths, None, None)

        task_type_log_probs = self.task_type_model(embedding)
        dialog_acts_probs = self.dialog_acts_model(embedding)
        sentiment_log_probs = self.sentiment_model(embedding)

        asr_loss = self.get_asr_loss(
            log_probs, input_lengths, labels, label_lengths)
        task_type_loss = self.task_type_model.get_loss(
            task_type_log_probs, task_type_labels)
        dialog_acts_loss = self.dialog_acts_model.get_loss(
            dialog_acts_probs, dialog_acts_labels)
        sentiment_loss = self.sentiment_model.get_loss(
            sentiment_log_probs, sentiment_labels)

        loss = (
            asr_loss * self.config.loss_params.asr_weight + 
            task_type_loss * self.config.loss_params.task_type_weight + 
            dialog_acts_loss * self.config.loss_params.dialog_acts_weight + 
            sentiment_loss * self.config.loss_params.sentiment_weight
        )

        with torch.no_grad():
            if isinstance(log_probs, tuple):
                log_probs = log_probs[1]

            hypotheses, hypothesis_lengths, references, reference_lengths = \
                self.asr_model.decode(
                    log_probs, input_lengths, 
                    labels, label_lengths,
                    self.train_dataset.sos_index,
                    self.train_dataset.eos_index,
                    self.train_dataset.pad_index,
                    self.train_dataset.eps_index,
                )
            asr_cer = get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)

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

            metrics = {
                f'{split}_loss': loss,
                # -- all losses 
                f'{split}_asr_loss': asr_loss,
                f'{split}_task_type_loss': task_type_loss,
                f'{split}_dialog_acts_loss': dialog_acts_loss,
                f'{split}_sentiment_loss': sentiment_loss,
                # -- asr metrics
                f'{split}_asr_cer': asr_cer,
                # -- task metrics
                f'{split}_num_task_type_correct': num_task_type_correct,
                f'{split}_num_task_type_total': num_task_type_total,
                # -- dialog_acts metrics
                f'{split}_num_dialog_acts_correct': num_dialog_acts_correct,
                f'{split}_num_dialog_acts_total': num_dialog_acts_total,
                # -- sentiment metrics
                f'{split}_num_sentiment_correct': num_sentiment_correct,
                f'{split}_num_sentiment_total': num_sentiment_total,
            }
            
            if split != 'train':
                # ---- store the full numpy so that we can
                metrics['dialog_acts_preds_npy'] = dialog_acts_preds_npy
                metrics['dialog_acts_labels_npy'] = dialog_acts_labels_npy

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, split='train')
        self.log_dict(metrics)
        self.log('train_loss', metrics['train_loss'], prog_bar=True, on_step=True)
        self.log('train_cer', metrics['train_asr_cer'], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, metrics = self.get_losses_and_metrics_for_batch(batch, split='val')
        return metrics

    def test_step(self, batch, batch_idx):
        _, metrics = self.get_losses_and_metrics_for_batch(batch, split='test')
        return metrics

    def process_outputs(self, outputs, split='val'):
        metrics = {}
        for key in outputs[0].keys():
            if key not in ['dialog_acts_preds_npy', 'dialog_acts_labels_npy']:
                metrics[key] = torch.tensor([elem[key]
                                            for elem in outputs]).float().mean()
        metric_keys = ['task_type', 'dialog_acts', 'sentiment']

        for name in metric_keys:
            num_correct = sum([out[f'{split}_num_{name}_correct'] for out in outputs])
            num_total = sum([out[f'{split}_num_{name}_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            metrics[f'{split}_{name}_acc'] = val_acc

        dialog_acts_preds_npy = np.concatenate([out['dialog_acts_preds_npy'] 
                                               for out in outputs], axis=0)
        dialog_acts_labels_npy = np.concatenate([out['dialog_acts_labels_npy'] 
                                                for out in outputs], axis=0)
        f1_scores = []
        for i in range(self.train_dataset.dialog_acts_num_class):
            f1_score_i = f1_score(dialog_acts_labels_npy[:, i], dialog_acts_preds_npy[:, i])
            f1_scores.append(f1_score_i)
        f1_scores = np.array(f1_scores)
        metrics[f'{split}_dialog_acts_f1'] = np.mean(f1_scores)
        return metrics

    def validation_epoch_end(self, outputs):
        metrics = self.process_outputs(outputs, split='val')
        self.log('val_loss', metrics['val_loss'], prog_bar=True)
        self.log('val_cer', metrics['val_asr_cer'], prog_bar=True)
        self.log_dict(metrics)

    def test_epoch_end(self, outputs):
        metrics = self.process_outputs(outputs, split='test')
        self.log('test_loss', metrics['test_loss'], prog_bar=True)
        self.log('test_cer', metrics['test_asr_cer'], prog_bar=True)
        self.log_dict(metrics)

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)


class LAS_System(CTC_System):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.add_sos_and_eos_tok = True
        self.add_eps_tok = False
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()
        self.create_model()

    def create_asr_model(self):
        asr_model = LASEncoderDecoder(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            self.train_dataset.transcript_maxlen,
            listener_hidden_dim=self.config.model_params.listener_hidden_dim,
            listener_num_layers=self.config.model_params.listener_num_layers,
            listener_bidirectional=self.config.model_params.listener_bidirectional,
            speller_num_layers=self.config.model_params.speller_num_layers,
            mlp_hidden_dim=self.config.model_params.mlp_hidden_dim,
            multi_head=self.config.model_params.multi_head,
            sos_index=self.train_dataset.sos_index,
        )
        self.asr_model = asr_model
        self.embedding_dim = asr_model.embedding_dim

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.asr_model(
            inputs,
            input_lengths,
            ground_truth=labels,
            teacher_force_prob=self.config.loss_params.teacher_force_prob,
        )
        return log_probs, embedding

    def get_asr_loss(self, log_probs, input_lengths, labels, label_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            labels,
            self.train_dataset.num_labels,
            pad_index=self.train_dataset.pad_index,
            label_smooth=self.config.loss_params.label_smooth,
        )
        return loss


class MTL_System(LAS_System):

    def create_asr_model(self):
        asr_model = MTLEncoderDecoder(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            self.train_dataset.transcript_maxlen,
            listener_hidden_dim=self.config.model_params.listener_hidden_dim,
            listener_num_layers=self.config.model_params.listener_num_layers,
            listener_bidirectional=self.config.model_params.listener_bidirectional,
            speller_num_layers=self.config.model_params.speller_num_layers,
            mlp_hidden_dim=self.config.model_params.mlp_hidden_dim,
            multi_head=self.config.model_params.multi_head,
            sos_index=self.train_dataset.sos_index,
        )
        self.asr_model = asr_model
        self.embedding_dim = asr_model.embedding_dim

    def forward(self, inputs, input_lengths, labels, label_lengths):
        ctc_log_probs, las_log_probs, embedding = self.asr_model(
            inputs,
            input_lengths,
            ground_truth=labels,
            teacher_force_prob=self.config.loss_params.teacher_force_prob,
        )
        return (ctc_log_probs, las_log_probs), embedding

    def get_asr_loss(self, log_probs, input_lengths, labels, label_lengths):
        (ctc_log_probs, las_log_probs) = log_probs
        ctc_loss, las_loss = self.asr_model.get_loss(
            ctc_log_probs,
            las_log_probs,
            input_lengths,
            labels,
            label_lengths,
            self.train_dataset.num_labels,
            pad_index=self.train_dataset.pad_index,
            blank_index=self.train_dataset.eps_index,
            label_smooth=self.config.loss_params.label_smooth,
        )
        loss = self.config.loss_params.alpha * ctc_loss + \
               (1 - self.config.loss_params.alpha) * las_loss
        return loss


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        num_workers=config.data_loader_workers,
    )
    return loader
