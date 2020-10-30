import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskTypePredictor(nn.Module):

    def __init__(self, input_dim, task_type_num_classes):
        super().__init__()
        self.task_fc = nn.Linear(input_dim, task_type_num_classes)
        self.task_type_num_classes = task_type_num_classes

    def forward(self, inputs):
        task_logits = self.task_fc(inputs)
        task_log_probs = F.log_softmax(task_logits, dim=1)
        return task_log_probs
    
    def get_loss(self, log_probs, targets):
        return F.nll_loss(log_probs, targets)


class DialogActsPredictor(nn.Module):

    def __init__(self, input_dim, num_dialog_acts):
        super().__init__()
        self.dialogacts_fc = nn.Linear(input_dim, num_dialog_acts)
        self.num_dialog_acts = num_dialog_acts

    def forward(self, inputs):
        dialogacts_logits = self.dialogacts_fc(inputs)
        # one person can have multiple dialog actions
        dialogacts_probs = torch.sigmoid(dialogacts_logits)
        return dialogacts_probs

    def get_loss(self, probs, targets):
        # probs   : batch_size x num_dialog_acts
        # targets : batch_size x num_dialog_acts
        return F.binary_cross_entropy(probs.view(-1), targets.view(-1).float())


class SentimentPredictor(nn.Module):

    def __init__(self, input_dim, sentiment_num_classes):
        super().__init__()
        self.sentiment_fc = nn.Linear(input_dim, sentiment_num_classes)
        self.sentiment_num_classes = sentiment_num_classes

    def forward(self, inputs):
        sentiment_logits = self.sentiment_fc(inputs)
        sentiment_log_probs = F.log_softmax(sentiment_logits, dim=1)
        return sentiment_log_probs

    def get_loss(self, pred_log_probs, target_probs):
        # pred_logits   : batch_size x num_sentiment_class
        # target_logits : batch_size x num_sentiment_class
        xentropy = -torch.sum(target_probs * pred_log_probs, dim=1)
        return torch.mean(xentropy)


class SpeakerIdPredictor(nn.Module):

    def __init__(self, input_dim, num_speaker_ids):
        super().__init__()
        self.speaker_id_fc = nn.Linear(input_dim, num_speaker_ids)
        self.num_speaker_ids = num_speaker_ids

    def forward(self, inputs):
        speaker_id_logits = self.speaker_id_fc(inputs)
        speaker_id_log_probs = F.log_softmax(speaker_id_logits, dim=1)
        return speaker_id_log_probs

    def get_loss(self, log_probs, targets):
        return F.nll_loss(log_probs, targets)
