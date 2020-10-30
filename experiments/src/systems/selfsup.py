"""
Try some simple SimCLR inspired audio adaptations. Audio augmentations
include cropping, noise, pitch, and speed. We should fit this on Librispeech.
"""

import os
import math
import faiss
import random
import librosa
import numpy as np
from dotmap import DotMap
from itertools import chain
from sklearn.metrics import f1_score
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision

from src.datasets.harper_valley import HarperValleyContrastiveTransfer
from src.datasets.librispeech import (
    LibriSpeech,
    LibriSpeechTransfer,
    LibriSpeechTwoViews,
)
from src.models.tasks import (
    TaskTypePredictor,
    DialogActsPredictor,
    SentimentPredictor,
    SpeakerIdPredictor,
)
from src.models.logreg import LogisticRegression
from src.models.resnet import resnet18
from src.models import resnet_small
from src.utils import l2_normalize, frozen_params, load_json
from src.systems.recognition import create_dataloader
from src.models.las import Listener
from src.utils import concat_all_gather

import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)

DEFAULT_KMEANS_SEED = 1234


class InstDiscSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.device = f'cuda:{config.gpu_device}' if config.cuda else 'cpu'
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_encoder()
        self.memory_bank = MemoryBank(len(self.train_dataset), 
                                      self.config.model_params.out_dim, 
                                      device=self.device)
        self.train_ordered_labels = self.train_dataset.all_speaker_ids

    def create_datasets(self):
        root = self.config.data_params.librispeech_root
        print('Initializing train dataset.')
        train_dataset = LibriSpeech(root, train=True, 
                                    spectral_transforms=self.config.data_params.spectral_transforms,
                                    wavform_transforms=self.config.data_params.wavform_transforms,
                                    train_urls=self.config.data_params.train_urls)
        print('Initializing validation dataset.')
        val_dataset = LibriSpeech(root, train=False, 
                                  spectral_transforms=self.config.data_params.spectral_transforms,
                                  wavform_transforms=self.config.data_params.wavform_transforms,
                                  # use the same split so same speakers
                                  test_url=self.config.data_params.test_url)
        return train_dataset, val_dataset

    def create_encoder(self):
        if self.config.model_params.resnet_small:
            model = resnet_small.ResNet18(
                self.config.model_params.out_dim,
                num_channels=1,
                input_size=self.config.data_params.input_size,
            )
        else:
            resnet_class = getattr(
                torchvision.models, 
                self.config.model_params.resnet_version,
            )
            model = resnet_class(
                pretrained=False,
                num_classes=self.config.model_params.out_dim,
            )
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                    padding=3, bias=False)

            if self.config.model_params.projection_head:
                mlp_dim = model.fc.weight.size(1)
                model.fc = nn.Sequential(
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.ReLU(),
                    model.fc,
                )

        return model.to(self.device)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=self.config.optim_params.learning_rate,
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        return [optim], []

    def forward(self, inputs):
        return self.model(inputs)

    def get_losses_for_batch(self, batch):
        indices, inputs, _ = batch
        outputs = self.forward(inputs)
        loss_fn = NCE(indices, outputs, self.memory_bank,
                      k=self.config.loss_params.k,
                      t=self.config.loss_params.t,
                      m=self.config.loss_params.m)
        loss = loss_fn.get_loss()

        with torch.no_grad():
            new_data_memory = loss_fn.updated_new_data_memory()
            self.memory_bank.update(indices, new_data_memory)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        metrics = {'loss': loss}
        return {'loss': loss, 'log': metrics}

    def get_nearest_neighbor_label(self, embs, labels):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.
        
        For each example in validation, find the nearest example in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        all_dps = self.memory_bank.get_all_dot_products(embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()

        return num_correct, embs.size(0)
    
    def validation_step(self, batch, batch_idx):
        _, inputs, speaker_ids = batch
        outputs = self.model(inputs)
        num_correct, batch_size = self.get_nearest_neighbor_label(outputs, speaker_ids)
        num_correct = torch.tensor(num_correct, dtype=float, device=self.device)
        batch_size = torch.tensor(batch_size, dtype=float, device=self.device)
        return OrderedDict({'val_num_correct': num_correct,
                            'val_num_total': batch_size})

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.stack([elem[key] for elem in outputs]).mean()
        num_correct = torch.stack([out['val_num_correct'] for out in outputs]).sum()
        num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'log': metrics, 'val_acc': val_acc} 

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)
    
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class LocalAggSystem(InstDiscSystem):

    def __init__(self, config):
        super().__init__(config)

        self._init_cluster_labels()
        self.km = None  # will be populated by a kmeans model

        if self.config.loss_params.kmeans_freq is None:
            data_size = len(self.train_dataset)
            batch_size = self.config.optim_params.batch_size
            self.config.loss_params.kmeans_freq = data_size // batch_size

    def _init_cluster_labels(self, attr_name='cluster_labels'):
        no_kmeans_k = self.config.loss_params.n_kmeans
        data_len = len(self.train_dataset)
        # initialize cluster labels
        cluster_labels = torch.arange(data_len).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans_k, 1)
        setattr(self, attr_name, cluster_labels)

    def get_losses_for_batch(self, batch):
        indices, inputs, _ = batch
        outputs = self.forward(inputs)
        loss_fn = LocalAgg(indices, outputs, self.memory_bank,
                           k=self.config.loss_params.k,
                           t=self.config.loss_params.t,
                           m=self.config.loss_params.m,
                           close_nei_combine='union')
        loss = loss_fn.get_loss(self.cluster_labels)

        with torch.no_grad():
            new_data_memory = loss_fn.updated_new_data_memory()
            self.memory_bank.update(indices, new_data_memory)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)

        if (self.global_step % self.config.loss_params.kmeans_freq == 0) or \
            self.global_step == 0:

            k = [self.config.loss_params.kmeans_k for _ in
                 range(self.config.loss_params.n_kmeans)]
            self.km = Kmeans(k, self.memory_bank, 
                             gpu_device=self.config.faiss_gpu_device)
            self.cluster_labels = self.km.compute_clusters()

        metrics = {'loss': loss}
        return {'loss': loss, 'log': metrics}


class SimCLRSystem(InstDiscSystem):

    def create_datasets(self):
        root = self.config.data_params.librispeech_root
        print('Initializing train dataset.')
        train_dataset = LibriSpeechTwoViews(
            root, 
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=self.config.data_params.wavform_transforms,
            train_urls=self.config.data_params.train_urls,
        )
        print('Initializing validation dataset.')
        val_dataset = LibriSpeech(root, train=False, 
                                  spectral_transforms=self.config.data_params.spectral_transforms,
                                  wavform_transforms=self.config.data_params.wavform_transforms,
                                  # use the same split so same speakers
                                  test_url=self.config.data_params.test_url)
        return train_dataset, val_dataset

    def get_losses_for_batch(self, batch):
        indices, inputs1, inputs2, _ = batch
        outputs1 = self.forward(inputs1)
        outputs2 = self.forward(inputs2)
        loss_fn = SimCLR(outputs1, outputs2, 
                         t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        with torch.no_grad():  # for nearest neighbor
            new_data_memory = (l2_normalize(outputs1, dim=1) + 
                               l2_normalize(outputs2, dim=1)) / 2.
            self.memory_bank.update(indices, new_data_memory)

        return loss


class MoCoSystem(SimCLRSystem):

    def __init__(self, config):
        super().__init__(config)

        self.model_k = self.create_encoder()

        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # do not update

        # create queue (k x out_dim)
        moco_queue = torch.randn(
            self.config.loss_params.k,
            self.config.model_params.out_dim, 
        )
        self.register_buffer("moco_queue", moco_queue)
        self.moco_queue = l2_normalize(moco_queue, dim=1)
        self.register_buffer("moco_queue_ptr", torch.zeros(1, dtype=torch.long))

    def get_losses_for_batch(self, batch):
        indices, inputs1, inputs2, _ = batch
        outputs1 = self.forward(inputs1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            if self.use_ddp or self.use_ddp2:
                inputs2, idx_unshuffle = self._batch_shuffle_ddp(inputs2)
            outputs2 = self.model_k(inputs2)
            if self.use_ddp or self.use_ddp2:
                outputs2 = self._batch_unshuffle_ddp(outputs2, idx_unshuffle)

        loss_fn = MoCo(outputs1, outputs2, 
                       self.moco_queue.clone().detach(),
                       t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        with torch.no_grad():
            outputs2 = l2_normalize(outputs2, dim=1)
            self._dequeue_and_enqueue(outputs2)
            
            outputs1 = l2_normalize(outputs1, dim=1)
            self.memory_bank.update(indices, outputs1)

        return loss

    # NOTE: MoCo specific function
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.config.loss_params.m
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    # NOTE: MoCo specific function
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        config_batch_size = self.config.optim_params.batch_size
        batch_size = keys.size(0)

        k = self.config.loss_params.k
        ptr = int(self.moco_queue_ptr)
        assert k % batch_size == 0  # why?

        # replace keys at ptr
        self.moco_queue[ptr:ptr+batch_size] = keys
        # move config by full batch size even if current batch is smaller
        ptr = (ptr + config_batch_size) % k

        self.moco_queue_ptr[0] = ptr

    # NOTE: MoCo specific function
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    # NOTE: MoCo specific function
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


class TransferBaseSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder, self.pretrain_config = self.load_pretrained_model()
        self.encoder = self.encoder.eval()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        frozen_params(self.encoder)
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_model()

    def load_embedding_size(self):
        resnet = self.pretrain_config.model_params.resnet_version
        base_feature_dict = {'resnet18': 512, 'resnet50': 2048}
        if self.pretrain_config.model_params.resnet_small:
            num_features = base_feature_dict['resnet18'] * 4 * 4
        else:
            num_features = base_feature_dict[resnet] * 4 * 4
        return num_features

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = load_json(config_path)
        config = DotMap(config_json)

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'])

        encoder = system.model.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder, config

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class TransferLibriSpeechSystem(TransferBaseSystem):

    def create_datasets(self):
        root = self.config.data_params.librispeech_root
        train_dataset = LibriSpeechTransfer(
            root,
            train=True,
            spectral_transforms=self.pretrain_config.data_params.spectral_transforms,
            wavform_transforms=self.pretrain_config.data_params.wavform_transforms,
        )
        val_dataset = LibriSpeechTransfer(
            root,
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
        )
        return train_dataset, val_dataset

    def create_model(self):
        num_features = self.load_embedding_size()
        model = LogisticRegression(num_features, self.train_dataset.num_unique_speakers)
        return model.to(self.device)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def forward(self, inputs):
        batch_size = inputs.size(0)
        embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch):
        _, inputs, label = batch
        logits = self.forward(inputs)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, inputs, label = batch
        logits = self.forward(inputs)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = inputs.size(0)
        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        num_correct, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}


class TransferHarperValleySystem(TransferBaseSystem):

    def create_datasets(self):
        root = self.config.data_params.harpervalley_root
        train_dataset = HarperValleyContrastiveTransfer(
            root,
            caller_intent=self.config.data_params.caller_intent,
            split='trainval',
            spectral_transforms=self.pretrain_config.data_params.spectral_transforms,
            wavform_transforms=self.pretrain_config.data_params.wavform_transforms,
            prune_speakers=self.config.data_params.prune_speakers,
        )
        val_dataset = HarperValleyContrastiveTransfer(
            root,
            caller_intent=self.config.data_params.caller_intent,
            split='test',
            spectral_transforms=False,
            wavform_transforms=False,
            prune_speakers=self.config.data_params.prune_speakers,
        )
        return train_dataset, val_dataset

    def create_model(self):
        num_features = self.load_embedding_size()
        caller_intent = self.config.data_params.caller_intent
        if caller_intent == 'task_type':
            model = TaskTypePredictor(num_features, self.train_dataset.num_class)
        elif caller_intent == 'dialog_acts':
            model = DialogActsPredictor(num_features, self.train_dataset.num_class)
        elif caller_intent == 'sentiment':
            model = SentimentPredictor(num_features, self.train_dataset.num_class)
        elif caller_intent == 'speaker_id':
            model = SpeakerIdPredictor(num_features, self.train_dataset.num_class)
        return model.to(self.device)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def forward(self, inputs):
        batch_size = inputs.size(0)
        with torch.no_grad():
            embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        log_probs = self.model(embs)
        return log_probs

    def get_losses_and_metrics_for_batch(self, batch, train=True):
        indices = batch[0].to(self.device)
        inputs = batch[1].to(self.device)
        labels = batch[2].to(self.device)

        batch_size = indices.size(0)
        caller_intent = self.config.data_params.caller_intent

        log_probs = self.forward(inputs)
        loss = self.model.get_loss(log_probs, labels)

        with torch.no_grad():
            if caller_intent == 'task_type':
                preds = torch.argmax(log_probs, dim=1)
                num_correct = (preds == labels).sum().item()
                num_total = batch_size
            elif caller_intent == 'dialog_acts':
                probs = log_probs  # actual log_probs
                preds = torch.round(probs)
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
            }
            if not train:
                if caller_intent == 'dialog_acts':
                    # ---- store the full numpy so that we can
                    metrics['preds_npy'] = preds_npy
                    metrics['labels_npy'] = labels_npy

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=True)
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=False)
        metrics['val_loss'] = loss
        return OrderedDict(metrics)

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
        if caller_intent == 'dialog_acts':
            preds_npy = np.concatenate([out['preds_npy'] for out in outputs], axis=0)
            labels_npy = np.concatenate([out['labels_npy'] for out in outputs], axis=0)
            f1_scores = []
            for i in range(self.train_dataset.num_class):
                f1_score_i = f1_score(labels_npy[:, i], preds_npy[:, i])
                f1_scores.append(f1_score_i)
            f1_scores = np.array(f1_scores)
            metrics[f'val_f1'] = np.mean(f1_scores)
            print(f'\nF1:{metrics[f"val_f1"]}')

        return {'val_loss': metrics['val_loss'], 'log': metrics}


class MemoryBank(nn.Module):
    """For efficiently computing the background vectors."""

    def __init__(self, size, dim, device):
        super().__init__()
        self.size = size
        self.dim = dim
        self.device = device
        self.register_buffer('_bank', self._create())

    def _create(self):
        # initialize random weights
        mb_init = torch.rand(self.size, self.dim, device=self.device)
        std_dev = 1. / np.sqrt(self.dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalise so that the norm is 1
        mb_init = l2_normalize(mb_init, dim=1)
        return mb_init.detach()  # detach so its not trainable

    def as_tensor(self):
        return self._bank

    def at_idxs(self, idxs):
        return torch.index_select(self._bank, 0, idxs)

    def get_all_distances(self, emb_batch):
        '''Returns a tensor of L2-distances between each given embedding and all the embeddings in the bank
        
        Args:
            emb_batch: [batch_size, emb_dim] Tensor of embeddings
        
        Returns:
            [batch_size, memory_bank_size] Tensor of L2-norm distances
        '''
        assert len(emb_batch.shape) == 2

        differences = self._bank.unsqueeze(0) - emb_batch.unsqueeze(1)
        # Broadcasted elementwise dot product.
        distances = torch.sqrt(torch.einsum('abc,abc->ab', differences, differences))
        return distances

    def get_all_dot_products(self, vec):
        # [bs, dim]
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self._bank, 1, 0))

    def get_dot_products(self, vec, idxs):
        vec_shape = list(vec.size())    # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def update(self, indices, data_memory):
        # in lieu of scatter-update operation
        data_dim = data_memory.size(1)
        self._bank = self._bank.scatter(0, indices.unsqueeze(1).repeat(1, data_dim),
                                        data_memory.detach())


def run_kmeans(x, nmb_clusters, verbose=False,
               seed=DEFAULT_KMEANS_SEED, gpu_device=0):
    """
    Runs kmeans on 1 GPU.
    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters
    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = gpu_device
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]


class Kmeans(object):
    """
    Train <k> different k-means clusterings with different
    random seeds. These will be used to compute close neighbors
    for a given encoding.
    """
    def __init__(self, k, memory_bank, gpu_device=0):
        super().__init__()
        self.k = k
        self.memory_bank = memory_bank
        self.gpu_device = gpu_device

    def compute_clusters(self):
        """
        Performs many k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
        """
        data = self.memory_bank.as_tensor()
        data_npy = data.cpu().detach().numpy()
        clusters = self._compute_clusters(data_npy)
        return clusters

    def _compute_clusters(self, data):
        pred_labels = []
        for k_idx, each_k in enumerate(self.k):
            # cluster the data
            I = run_kmeans(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                           gpu_device=self.gpu_device)
            clust_labels = np.asarray(I)
            pred_labels.append(clust_labels)
        pred_labels = np.stack(pred_labels, axis=0)
        pred_labels = torch.from_numpy(pred_labels).long()

        return pred_labels


class NCE(object):

    def __init__(self, indices, outputs, memory_bank, k=4096, t=0.07, m=0.5):
        super().__init__()
        self.k, self.t, self.m = k, t, m

        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size

    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k-1), device=self.device).long()
        noise_indx = torch.cat([self.indices.unsqueeze(1), noise_indx], dim=1)
        witness_norm = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(self.k)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss


class MoCo(object):

    def __init__(self, outputs1, outputs2, queue, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.queue = queue.detach()
        self.t = t
        self.k = queue.size(0)
        self.device = self.outputs1.device

    def get_loss(self):
        witness_pos = torch.sum(self.outputs1 * self.outputs2, dim=1, keepdim=True)
        witness_neg = self.outputs1 @ self.queue.T
        # batch_size x (k + 1)
        witness_logits = torch.cat([witness_pos, witness_neg], dim=1) / self.t

        labels = torch.zeros(witness_logits.size(0), device=self.device).long()
        loss = F.cross_entropy(witness_logits, labels.long()) 
        return loss


class LocalAgg(NCE):

    def __init__(self, indices, outputs, memory_bank, k=4096, t=0.07, m=0.5,
                 close_nei_combine='union'):
        super().__init__(indices, outputs, memory_bank, k=k, t=t, m=m)
        self.close_nei_combine = close_nei_combine

    def __get_close_nei_in_back(self, each_k_idx, cluster_labels,
                                back_nei_idxs, k):
        # get which neighbors are close in the background set
        # NOTE: cast CPU() bc this operation is expensive
        batch_labels = cluster_labels[each_k_idx][self.indices.cpu()]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei.byte()

    def get_loss(self, cluster_labels):
        batch_size = self.outputs.size(0)
        all_dps = self.memory_bank.get_all_dot_products(self.outputs)
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=self.k, sorted=False, dim=1)

        all_close_nei = None
        no_kmeans = cluster_labels.size(0)
        with torch.no_grad():
            for each_k_idx in range(no_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(
                    each_k_idx,
                    cluster_labels,
                    back_nei_idxs.cpu(),
                    self.k,
                )
                if all_close_nei is None:
                    all_close_nei = curr_close_nei
                else:
                    if self.close_nei_combine == 'union':
                        all_close_nei = all_close_nei | curr_close_nei
                    elif self.close_nei_combine == 'intersection':
                        all_close_nei = all_close_nei & curr_close_nei
                    else:
                        raise Exception('Combine strategy close_nei_combine not supported.')

            all_close_nei = all_close_nei.to(self.device)

        close_nei_probs = []  # # maybe theres a scatter operation
        for i in range(batch_size):
            num_nonzero = torch.sum(all_close_nei[i]).item()
            if num_nonzero == 0:  # all 0's
                close_nei_probs_i = torch.Tensor([1e-7]).to(self.outputs.device)
                close_nei_probs_i = torch.log(close_nei_probs_i)
            else:
                ix_i = torch.where(all_close_nei[i] > 0)[0]
                close_nei_dps_i = torch.index_select(back_nei_dps[i], 0, ix_i)
                close_nei_probs_i = torch.logsumexp(close_nei_dps_i / self.t, dim=0, keepdim=True)
            close_nei_probs.append(close_nei_probs_i)
        close_nei_probs = torch.stack(close_nei_probs).squeeze(1)

        back_nei_probs = torch.logsumexp(back_nei_dps / self.t, dim=1)
        loss = -torch.mean(close_nei_probs - back_nei_probs)
        return loss


class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # overcounts a constant
        witness_norm = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)
