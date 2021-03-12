import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

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
from src.datasets.harper_valley import HarperValleyWav2VecTransfer
from src.models.tasks import (
    TaskTypePredictor,
    DialogActsPredictor,
    SentimentPredictor,
    SpeakerIdPredictor,
)
from src.systems.recognition import create_dataloader
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)


class Wav2Vec_System(pl.LightningModule):
    """
    Try a pre-trained Wav2Vec model on the transfer tasks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.wav2vec = self.create_wav2vec(
            self.config.data_params.wav2vec_path,
        )
        self.model = self.create_model()

    def create_wav2vec(self, weight_path):
        cp = torch.load(weight_path)
        wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
        wav2vec.load_state_dict(cp['model'])
        wav2vec.eval()
        for param in wav2vec.parameters():
            param.requires_grad = False
        return wav2vec

    def create_datasets(self):
        root = self.config.data_params.harpervalley_root
        train_dataset = HarperValleyWav2VecTransfer(
            root,
            self.config.data_params.caller_intent,
            split='train',
            prune_speakers=self.config.data_params.prune_speakers,
        )
        val_dataset = HarperValleyWav2VecTransfer(
            root,
            self.config.data_params.caller_intent,
            split='test',
            prune_speakers=self.config.data_params.prune_speakers,
        )
        return train_dataset, val_dataset

    def create_model(self):
        caller_intent = self.config.data_params.caller_intent
        if caller_intent == 'task_type':
            model = TaskTypePredictor(
                self.config.model_params.embedding_dim,
                self.train_dataset.num_class,
            )
        elif caller_intent == 'dialog_acts':
            model = DialogActsPredictor(
                self.config.model_params.embedding_dim,
                self.train_dataset.num_class,
            )
        elif caller_intent == 'sentiment':
            model = SentimentPredictor(
                self.config.model_params.embedding_dim,
                self.train_dataset.num_class,
            )
        elif caller_intent == 'speaker_id':
            model = SpeakerIdPredictor(
                self.config.model_params.embedding_dim,
                self.train_dataset.num_class,
            )
        else:
            raise Exception(f'Caller intent {caller_intent} not supported.')
        return model

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )        
        return [optim], []

    def forward(self, inputs):
        with torch.no_grad():
            embeddings = self.wav2vec.feature_extractor(inputs)
            embeddings = self.wav2vec.feature_aggregator(embeddings)
            embeddings = torch.mean(embeddings, dim=2)
        return self.model(embeddings)

    def get_losses_and_metrics_for_batch(self, batch, train=True):
        indices, inputs, labels = batch
        batch_size = indices.size(0)

        log_probs = self.forward(inputs)
        loss = self.model.get_loss(log_probs, labels)

        caller_intent = self.config.data_params.caller_intent

        with torch.no_grad():
            if caller_intent == 'task_type':
                preds = torch.argmax(log_probs, dim=1)
                num_correct = (preds == labels).sum().item()
                num_total = batch_size
            elif caller_intent == 'dialog_acts':
                preds = torch.round(log_probs)
                num_correct = (preds == labels).sum().item()
                num_total = batch_size * self.train_dataset.num_class
                preds_npy = preds.cpu().numpy()     # batch_size x num_dialog_actions
                labels_npy  = labels.cpu().numpy()  # batch_size x num_dialog_actions
            elif caller_intent == 'sentiment':
                preds = torch.argmax(log_probs, dim=1)
                labels = torch.argmax(labels, dim=1)
                num_correct = (preds == labels).sum().item()
                num_total = batch_size
            elif caller_intent == 'speaker_id':
                preds = torch.argmax(log_probs, dim=1)
                num_correct = (preds == labels).sum().item()
                num_total = batch_size

            prefix = 'train' if train else 'val'

            metrics = {
                f'{prefix}_loss': loss,
                f'{prefix}_num_correct': num_correct,
                f'{prefix}_num_total': num_total,
                f'{prefix}_acc': num_correct / float(num_total),
            }
            if not train:
                if caller_intent == 'dialogacts':
                    # ---- store the full numpy so that we can
                    metrics['preds_npy'] = preds_npy
                    metrics['labels_npy'] = labels_npy

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=True)
        self.log_dict(metrics)
        self.log('train_loss', metrics['train_loss'], prog_bar=True, on_step=True)
        self.log('train_acc', metrics['train_acc'], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, metrics = self.get_losses_and_metrics_for_batch(batch, train=False)
        return metrics

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            if key not in ['preds_npy', 'labels_npy']:
                metrics[key] = torch.tensor([elem[key]
                                            for elem in outputs]).float().mean()

        num_correct = sum([out[f'val_num_correct'] for out in outputs])
        num_total = sum([out[f'val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics[f'val_acc'] = val_acc

        caller_intent = self.config.data_params.caller_intent
        if caller_intent == 'dialogacts':
            preds_npy = np.concatenate([out['preds_npy'] for out in outputs], axis=0)
            labels_npy = np.concatenate([out['labels_npy'] for out in outputs], axis=0)
            f1_scores = []
            for i in range(self.train_dataset.num_class):
                f1_score_i = f1_score(labels_npy[:, i], preds_npy[:, i])
                f1_scores.append(f1_score_i)
            f1_scores = np.array(f1_scores)
            metrics[f'val_f1'] = np.mean(f1_scores)

        self.log_dict(metrics)
        self.log('val_loss', metrics['val_loss'], prog_bar=True)
        self.log('val_acc', metrics['val_acc'], prog_bar=True)

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class Wav2Vec2_System(Wav2Vec_System):
    
    def create_wav2vec(self, weight_path):
        cp = torch.load(weight_path)
        new_state_dict = {}
        for name, value in cp['model'].items():
            if 'w2v_encoder.w2v_model.' in name:
                name = name.replace('w2v_encoder.w2v_model.', '')
            new_state_dict[name] = value
        wav2vec = Wav2Vec2Model.build_model(cp['args'].w2v_args, task=None)
        wav2vec.load_state_dict(new_state_dict, strict=False)
        wav2vec.eval()
        for param in wav2vec.parameters():
            param.requires_grad = False
        return wav2vec

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.wav2vec.forward(inputs, features_only=True)
            embeddings = outputs['x']
            embeddings = embeddings.mean(1)
        return self.model(embeddings)
