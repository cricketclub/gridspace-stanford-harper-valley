import os
import json
import torch
import pickle
import string
import random
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from glob import glob
import nlpaug.flow as naf
from collections import Counter
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize

from torch.utils.data import Dataset
from src.datasets.librispeech import (
    WavformAugmentation,
    SpectrumAugmentation,
)

HARPER_VALLEY_MEAN = [-29.436176]
HARPER_VALLEY_STDEV = [14.90793]
HARPER_VALLEY_HOP_LENGTH_DICT = {
    224: 672, 
    112: 1344, 
    64: 2360, 
    32: 4800,
}

# I fetched from the dataset itself (all lowered cased)
VOCAB = [' ',"'",'~','-','.','<','>','[',']','a','b','c','d','e','f','g',
         'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
         'w','x','y','z']


class BaseHarperValley(Dataset):
    """Base class for loading HarperValley datasets that will handle the data
    loading and preprocessing.

    @param root: string 
                 path to the directory where harpervalley data is stored.
    @param min_utterance_length: integer (default: 4)
                                 minimum number of tokens to compose an utterance. 
                                 Utterances with fewer tokens will be ignored.
    @param prune_speakers: boolean (default: True)
                           prune speakers who make less than min_utterance_length speakers      
    """
    def __init__(self, root, min_utterance_length=4, min_speaker_utterances=10, prune_speakers=True):
        super().__init__()
        agent_wavpaths, caller_wavpaths, transcript_paths, metadata_paths = self.load_paths(root)
        metadata = self.load_metadata(metadata_paths)
        transcripts = self.load_transcripts(transcript_paths)
        if prune_speakers:
            transcripts = self.prune_transcripts(transcripts, min_utterance_length)
        # A list of entries, each with the following keys:
        #   - wavpath
        #   - wav_start_ix
        #   - wav_end_ix
        #   - speaker_type (caller | agent)
        #   - speaker_id
        #   - human_transcript
        #   - task_type
        #   - sentiment
        #   - dialog_acts
        _, invalid_speaker_ids = self.aggregate_speaker_ids(
            transcripts,
            metadata,
            min_speaker_utterances=min_speaker_utterances,
        )
        data = self.aggregate_raw_data(
            agent_wavpaths,
            caller_wavpaths,
            transcripts,
            metadata,
            invalid_speaker_ids=invalid_speaker_ids,
        )
        self.data = data
        self.root = root
        self.min_utterance_length = min_utterance_length
        self.min_speaker_utterances = min_speaker_utterances
        self.prune_speakers = prune_speakers

    def load_paths(self, root):
        agent_bases  = glob(os.path.join(root, 'audio', 'agent', '*.wav'))
        caller_bases = glob(os.path.join(root, 'audio', 'agent', '*.wav'))
        agent_bases  = [os.path.basename(base) for base in agent_bases]
        caller_bases = [os.path.basename(base) for base in caller_bases]
        assert len(agent_bases) == len(caller_bases)

        transcript_bases = os.listdir(os.path.join(root, 'transcript'))
        metadata_bases = os.listdir(os.path.join(root, 'metadata'))

        def intersection(lst1, lst2): 
            return list(set(lst1) & set(lst2))

        basenames_1 = []
        for transcript_base in transcript_bases:
            basename = transcript_base.replace('.json', '')
            basenames_1.append(basename)
        basenames_2 = []
        for agent_base in agent_bases:
            basename = agent_base.replace('.wav', '')
            basenames_2.append(basename)
        basenames_3 = []
        for metadata_base in metadata_bases:
            basename = metadata_base.replace('.json', '')
            basenames_3.append(basename)
        all_basenames = intersection(intersection(basenames_1, basenames_2), basenames_3) 
        
        agent_dir = os.path.join(root, 'audio', 'agent')
        caller_dir = os.path.join(root, 'audio', 'caller')
        transcript_dir = os.path.join(root, 'transcript')
        metadata_dir = os.path.join(root, 'metadata')

        agent_paths = [os.path.join(agent_dir, basename + '.wav') 
                        for basename in all_basenames]
        caller_paths = [os.path.join(caller_dir, basename + '.wav')
                        for basename in all_basenames]
        transcript_paths = [os.path.join(transcript_dir, basename + '.json') 
                            for basename in all_basenames]
        metadata_paths = [os.path.join(metadata_dir, basename + '.json')
                            for basename in all_basenames]

        return agent_paths, caller_paths, transcript_paths, metadata_paths

    @staticmethod
    def aggregate_speaker_ids(transcripts, metadata, min_speaker_utterances=10):
        all_speaker_ids = []
        for transcript, metadatum in zip(transcripts, metadata):
            num_utterances = len(transcript)
            for i in range(num_utterances):
                row = transcript[i]
                if row['channel_index'] == 2:  # agent is speaking
                    row_speaker_id = metadatum['agent']['speaker_id']
                elif row['channel_index'] == 1:
                    row_speaker_id = metadatum['caller']['speaker_id']
                else:
                    index = row['channel_index']
                    raise Exception(f'channel_index {index} unsupported.')
                all_speaker_ids.append(row_speaker_id)
        
        speaker_freq = dict(Counter(all_speaker_ids))
        valid_speaker_ids = []
        invalid_speaker_ids = []

        for speaker_id, freq in speaker_freq.items():
            if freq <= min_speaker_utterances:
                invalid_speaker_ids.append(speaker_id)
            else:
                valid_speaker_ids.append(speaker_id)

        valid_speaker_ids = sorted(valid_speaker_ids)
        invalid_speaker_ids = sorted(invalid_speaker_ids)

        return valid_speaker_ids, invalid_speaker_ids

    @staticmethod
    def aggregate_raw_data(agent_wavpaths, caller_wavpaths, transcripts, metadata, invalid_speaker_ids=[]):
        # make sure everything is the same length
        assert len(agent_wavpaths) == len(caller_wavpaths)
        assert len(agent_wavpaths) == len(transcripts)
        assert len(agent_wavpaths) == len(metadata)

        dataset = []  # store entries here

        pbar = tqdm(total=len(agent_wavpaths))
        for batch in zip(agent_wavpaths, caller_wavpaths, transcripts, metadata):
            agent_wavpath = batch[0]
            caller_wavpath = batch[1]
            transcript = batch[2]
            metadatum = batch[3]

            tasks = metadatum['tasks']
            # all task_types in a single transcript are the same
            task_type = tasks[0]['task_type']

            num_utterances = len(transcript)

            for i in range(num_utterances):
                row = transcript[i]
                row_human_transcript = row['human_transcript']
                row_sentiment = row['emotion']
                row_dialog_acts = row['dialog_acts']

                if row['channel_index'] == 2:  # agent is speaking
                    row_speaker = 'agent'
                    row_speaker_id = metadatum['agent']['speaker_id']
                    row_wavpath = agent_wavpath
                elif row['channel_index'] == 1:
                    row_speaker = 'caller'
                    row_speaker_id = metadatum['caller']['speaker_id']
                    row_wavpath = caller_wavpath
                else:
                    index = row['channel_index']
                    raise Exception(f'channel_index {index} unsupported.')

                if row_speaker_id in invalid_speaker_ids:
                    continue  # skip! we dont want to include any utterances by this speaker

                entry = {
                    'wavpath': row_wavpath,
                    'speaker': row_speaker,
                    'speaker_id': row_speaker_id,
                    'human_transcript': row_human_transcript,
                    'task_type': task_type,
                    'sentiment': row_sentiment,
                    'dialog_acts': row_dialog_acts,
                    # ---
                    'crop_start_ms': row['start_ms'],
                    'crop_duration_ms': row['duration_ms'],
                }
                dataset.append(entry)

            pbar.update()
        pbar.close()
        
        return dataset

    @staticmethod
    def load_transcripts(transcript_paths):
        transcripts = []
        for path in transcript_paths:
            with open(path) as fp:
                transcript = json.load(fp)
            transcripts.append(transcript)
        
        return transcripts

    @staticmethod
    def load_metadata(metadata_paths):
        metadata = []
        for path in metadata_paths:
            with open(path) as fp:
                metadata.append(json.load(fp))
        return metadata

    def prune_data_by_speaker(self, data, min_freq=10):
        pruned_data = []
        valid_speaker_ids, _ = self.process_speaker_ids(data, min_freq=min_freq)
        for row in data:
            if row['speaker_id'] in valid_speaker_ids:
                pruned_data.append(row)
        return pruned_data

    def process_speaker_ids(self, data, min_freq=10):
        # speaker ids arent balanced... collapse everything 
        # that speaks less than 10 times
        speaker_ids = [row['speaker_id'] for row in data]
        speaker_freq = dict(Counter(speaker_ids))
        valid_speaker_ids = []
        invalid_speaker_ids = []

        for speaker_id, freq in speaker_freq.items():
            if freq <= min_freq:
                invalid_speaker_ids.append(speaker_id)
            else:
                valid_speaker_ids.append(speaker_id)

        valid_speaker_ids = sorted(list(set(valid_speaker_ids)))
        invalid_speaker_ids = sorted(list(set(invalid_speaker_ids)))

        return valid_speaker_ids, invalid_speaker_ids

    @staticmethod
    def prune_transcripts(transcripts, min_length=4):
        """Throw away sentences with < min_length words"""
        
        def prune_transcript(transcript):
            num = len(transcript)   # num utterances in transcript
            transcript_ = []
            for i in range(num):
                transcript_text_i = transcript[i].get(
                    'human_transcript',
                    transcript[i]['transcript'],
                )
                words_i = transcript_text_i.split()
                if len(words_i) >= min_length:
                    transcript_.append(transcript[i])
            return transcript_

        return [prune_transcript(transcript) for transcript in transcripts]

    def get_vocabs(self, data):
        tasks = [row['task_type'] for row in data]
        dialogacts = [row['dialog_acts'] for row in data]
        dialogact_vocab = sorted(set([item for sublist in dialogacts
                                      for item in sublist]))
        task_vocab = sorted(set(tasks))
        valid_speaker_ids, invalid_speaker_ids = self.process_speaker_ids(data)
        assert len(invalid_speaker_ids) == 0
        return task_vocab, dialogact_vocab, valid_speaker_ids

    def transcripts_to_labels(self, texts):
        # Convert text strings to a list of indices representing characters.
        labels = []
        for text in texts:
            chars = list(text)
            label = [VOCAB.index(ch) for ch in chars]
            labels.append(label)
        return labels

    def task_type_to_labels(self, tasks, vocab):
        return [vocab.index(task) for task in tasks]

    def dialog_acts_to_labels(self, dialogacts, vocab):
        actions = []
        for acts in dialogacts:
            onehot = [0 for _ in range(len(vocab))]
            for act in acts:
                onehot[vocab.index(act)] = 1
            actions.append(onehot)
        return actions

    def speaker_id_to_labels(self, speaker_ids, valid_ids):
        labels = []
        for speaker_id in speaker_ids:
            if speaker_id in valid_ids:
                labels.append(valid_ids.index(speaker_id))
            else:
                raise Exception(f'speaker_id {speaker_id} unexpected.')
        return labels

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class HarperValley(BaseHarperValley):
    """Dataset to be used to train CTC, LAS, and MTL. 

    @param inputs_maxlen (default: None)
                         Maximum number of tokens in the wav input.
    @param labels_maxlen (default: None)
                         Maximum number of tokens in the labels output.
    @param add_sos_and_eos_tok: boolean (default: False)
                                Whether to prepend SOS token and append with EOS token.
                                Required for LAS and MTL.
    @param add_eps_token: boolean (default: False)
                          Whether to add blank / epsilon tokens.
                          Required for CTC and MTL.
    @param split_by_speaker: boolean (default: False)
                             Whether to train/test split randomly or by speaker,
                             where entries in the training and test sets have
                             disjoint speakers.
    """
    def __init__(
            self,
            root, 
            split='train', 
            n_mels=128,
            wav_maxlen=200,
            transcript_maxlen=200,
            add_sos_and_eos_tok=False,
            add_eps_tok=False,
            split_by_speaker=False,
            min_utterance_length=4,
            min_speaker_utterances=10,
            prune_speakers=True,
        ):
        super().__init__(root, min_utterance_length, min_speaker_utterances, prune_speakers)

        if not split_by_speaker:
            train_data, val_data, test_data = self.train_test_split(self.data)
        else:
            train_data, val_data, test_data = self.train_test_split_by_speaker(self.data)

        # do some processing to pull out labels
        task_type_vocab, dialog_acts_vocab, speaker_id_set = self.get_vocabs(self.data)

        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        elif split == 'test':
            data = test_data
        else:
            raise Exception(f'Split {split} not supported.')

        wavpaths = [d['wavpath'] for d in data]
        human_transcripts = [d['human_transcript'] for d in data]
        task_types = [d['task_type'] for d in data]
        dialog_acts = [d['dialog_acts'] for d in data]
        sentiments = [d['sentiment'] for d in data]
        speaker_ids = [d['speaker_id'] for d in data]
        crop_start_ms_list = [d['crop_start_ms'] for d in data]
        crop_duration_ms_list = [d['crop_duration_ms'] for d in data]
        human_transcript_labels = self.transcripts_to_labels(human_transcripts)
        task_type_labels = self.task_type_to_labels(task_types, task_type_vocab)
        dialog_acts_labels = self.dialog_acts_to_labels(dialog_acts, dialog_acts_vocab)
        speaker_id_labels = self.speaker_id_to_labels(speaker_ids, speaker_id_set)
        sentiment_labels = sentiments

        if add_eps_tok and add_sos_and_eos_tok:
            # add 3 because reserve 0 for epsilon, 1 for sos and 2 for eos
            human_transcript_labels = [list(np.array(lab) + 3) 
                                       for lab in human_transcript_labels]
            eps_index, sos_index, eos_index = 0, 1, 2
        elif add_eps_tok:
            # add 1 bc we reserve 0 for epsilon
            human_transcript_labels = [list(np.array(lab) + 1) 
                                       for lab in human_transcript_labels]
            eps_index = 0
        elif add_sos_and_eos_tok:
            # add 2 bc we reserve 0 for sos and 1 for eos
            human_transcript_labels = [list(np.array(lab) + 2) 
                                       for lab in human_transcript_labels]
            sos_index, eos_index = 0, 1

        if add_sos_and_eos_tok:
            # Add a EOS token to the end of all the labels
            # This is important for the sequential decoding objective
            print("Adding EOS token to all labels.")
            human_transcript_labels_ = []
            for i in range(len(human_transcript_labels)):
                new_label_i = human_transcript_labels[i] + [eos_index]  # eos_token
                human_transcript_labels_.append(new_label_i)
            human_transcript_labels = human_transcript_labels_

        if add_eps_tok and add_sos_and_eos_tok:
            self.num_class = len(VOCAB) + 3  # eps/sos/eos
            self.eps_index = eps_index
            self.sos_index = sos_index
            self.eos_index = eos_index
            self.pad_index = sos_index
        elif add_eps_tok:
            self.num_class = len(VOCAB) + 1  # epsilon
            self.eps_index = eps_index
        elif add_sos_and_eos_tok:
            self.num_class = len(VOCAB) + 2  # sos/eos
            self.sos_index = sos_index
            self.eos_index = eos_index
            self.pad_index = sos_index

        self.root = root
        self.add_eps_tok = add_eps_tok
        self.add_sos_and_eos_tok = add_sos_and_eos_tok
        self.wavpaths = wavpaths
        self.human_transcript_labels = human_transcript_labels
        self.task_type_labels = task_type_labels
        self.dialog_acts_labels = dialog_acts_labels
        self.speaker_id_labels = speaker_id_labels
        self.sentiment_labels = sentiment_labels
        self.task_type_num_class = len(task_type_vocab)
        self.dialog_acts_num_class = len(dialog_acts_vocab)
        self.sentiment_num_class = len(sentiment_labels[0])
        self.speaker_id_num_class = len(speaker_id_set)
        self.speaker_id_set = speaker_id_set
        self.split_by_speaker = split_by_speaker
        self.crop_start_ms_list = crop_start_ms_list
        self.crop_duration_ms_list = crop_duration_ms_list
        self.wav_maxlen = wav_maxlen
        self.transcript_maxlen = transcript_maxlen
        self.input_dim = n_mels
        self.num_labels = self.num_class
        self.n_mels = n_mels

    def train_test_split(
            self,
            data,
            train_frac = 0.8,
            val_frac = 0.1,
            return_indices = False,
        ):
        """
        Randomly split. No guarantees!
        """
        num_rows = len(data)
        num_train = int(num_rows * train_frac)
        num_val = int(num_rows * val_frac)
        indices = np.arange(num_rows)

        rs = np.random.RandomState(42)  # fix seed so reproducible splitting
        rs.shuffle(indices)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train+num_val]
        test_indices = indices[num_train+num_val:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        if return_indices:
            return (train_data, val_data, test_data), \
                   (train_indices, val_indices, test_indices)

        return train_data, val_data, test_data

    def train_test_split_by_speaker(
            self,
            data,
            train_frac = 0.8,
            val_frac = 0.1,
            return_indices = False,
        ):
        """
        Split such that the training and test sets have disjoint speakers.
        """
        # data here should also have bad speaker ids pruned out
        unique_speaker_ids = sorted(list(set([row['speaker_id'] for row in data])))
        unique_speaker_ids = np.array(unique_speaker_ids)
        rs = np.random.RandomState(42)  # fix seed so reproducible splitting
        rs.shuffle(unique_speaker_ids)

        num_speaker = len(unique_speaker_ids)
        num_train = int(train_frac * num_speaker)
        num_val = int(val_frac * num_speaker)
        num_test = num_speaker - num_train - num_val

        train_speaker_ids = unique_speaker_ids[:num_train]
        val_speaker_ids = unique_speaker_ids[num_train:num_train+num_val]
        test_speaker_ids = unique_speaker_ids[num_train+num_val:]

        train_speaker_dict = dict(zip(train_speaker_ids, ['train'] * num_train))
        val_speaker_dict = dict(zip(val_speaker_ids, ['val'] * num_val))
        test_speaker_dict = dict(zip(test_speaker_ids, ['test'] * num_test))
        speaker_dict = {**train_speaker_dict, **val_speaker_dict, **test_speaker_dict} 

        train_data, val_data, test_data = [], [], []
        train_indices, val_indices, test_indices = [], [], []
        for i, row in enumerate(data):
            if speaker_dict[row['speaker_id']] == 'train':
                train_data.append(row)
                train_indices.append(i)
            elif speaker_dict[row['speaker_id']] == 'val':
                val_data.append(row)
                val_indices.append(i)
            elif speaker_dict[row['speaker_id']] == 'test':
                test_data.append(row)
                test_indices.append(i)
            else:
                raise Exception('split not recognized.')

        if return_indices:
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)
            return (train_data, val_data, test_data), \
                   (train_indices, val_indices, test_indices)

        return train_data, val_data, test_data

    @staticmethod
    def pad_wav(wav, maxlen):
        dim = wav.shape[1]
        padded = np.zeros((maxlen, dim))
        if len(wav) > maxlen:
            wav = wav[:maxlen]
        length = len(wav)
        padded[:length, :] = wav
        return padded, length
    
    @staticmethod
    def pad_transcript_labels(transcript_labels, maxlen):
        padded = np.zeros(maxlen) - 1
        if len(transcript_labels) > maxlen:
            transcript_labels = transcript_labels[:maxlen]
        length = len(transcript_labels)
        padded[:length] = transcript_labels
        return padded, length

    def __getitem__(self, index):
        wavpath = self.wavpaths[index]
        wav, sr = torchaudio.load(wavpath)
        wav = wav.numpy()[0]

        crop_start_ms = self.crop_start_ms_list[index]
        crop_duration_ms = self.crop_duration_ms_list[index]
        crop_start_ix = librosa.core.time_to_samples(
            crop_start_ms / 1000.,
            sr=sr
        )
        crop_end_ix = librosa.core.time_to_samples(
            (crop_start_ms + crop_duration_ms) / 1000.,
            sr=sr,
        )
        wav_crop = wav[crop_start_ix:crop_end_ix]

        wav_mel = librosa.feature.melspectrogram(
            y=wav_crop, 
            sr=sr,
            n_mels=self.n_mels,
        )
        wav_logmel = np.log(wav_mel + 1e-9)
        wav_logmel = librosa.util.normalize(wav_logmel).T

        # pad to a fixed length
        wav_logmel, wav_length = self.pad_wav(wav_logmel, self.wav_maxlen)

        human_transcript_label = self.human_transcript_labels[index]
        # pad transcript to a fixed length
        human_transcript_label, human_transcript_length = self.pad_transcript_labels(
            human_transcript_label, 
            self.transcript_maxlen,
        )
        
        task_type_label = self.task_type_labels[index]
        dialog_acts_label = self.dialog_acts_labels[index]
        sentiment_label = self.sentiment_labels[index]
        sentiment_label = np.array([
            sentiment_label['positive'],
            sentiment_label['neutral'],
            sentiment_label['negative'],
        ])

        if self.split_by_speaker:
            # hack: we cant predict speaker if val/test have
            #       all unseen speakers. In this case, we just 
            #       trivially return -1.
            speaker_id_label = -1
        else:
            speaker_id_label = self.speaker_id_labels[index]

        wav_logmel = torch.from_numpy(wav_logmel).float()
        dialog_acts_label = torch.LongTensor(dialog_acts_label)
        sentiment_label = torch.FloatTensor(sentiment_label)

        return (index, wav_logmel, wav_length, 
                human_transcript_label, human_transcript_length,
                task_type_label, dialog_acts_label, 
                sentiment_label, speaker_id_label)

    def __len__(self):
        return len(self.wavpaths)


class BaseHarperValleyTransfer(BaseHarperValley):

    def train_test_split_by_transfer(
            self,
            data,
            caller_intent,
            task_type_vocab, 
            dialog_acts_vocab, 
            valid_speaker_ids,
            train_frac = 0.8,
            val_frac = 0.1,
            return_indices = False,
        ):
        """
        Split for balance for a single transfer task. We ensure an 80/20
        split for every class. For dialog action, which is a multi-label
        classification task, we remove very rare actions. 

        This function is not used in this class but will be required for 
        a child class.
        """
        assert caller_intent in [
            'speaker_id',
            'task_type',
            'dialog_acts',
            'sentiment',
        ]

        rs = np.random.RandomState(42)  # fix seed so reproducible splitting

        if caller_intent == 'speaker_id':
            speaker_ids = [row['speaker_id'] for row in data]
            speaker_ids = self.speaker_id_to_labels(speaker_ids, valid_speaker_ids)
            unique_speaker_ids = np.array(sorted(set(speaker_ids)))
            speaker_ids = np.array(speaker_ids)

            # train test split to ensure the 80/10/10 splits such that each speaker
            # appears in all the splits
            train_indices, val_indices, test_indices = [], [], []
            for speaker_id in unique_speaker_ids:
                speaker_indices = np.where(speaker_ids == speaker_id)[0]
                size = len(speaker_indices)
                rs.shuffle(speaker_indices)
                train_size = int(train_frac * size)
                test_size = int(val_frac * size)
                val_size = size - train_size - test_size
                train_indices.extend(speaker_indices[:train_size].tolist())
                val_indices.extend(speaker_indices[train_size:train_size+val_size].tolist())
                test_indices.extend(speaker_indices[-test_size:].tolist())

        elif caller_intent == 'task_type':
            task_types = [row['task_type'] for row in data]
            task_types = self.task_type_to_labels(task_types, task_type_vocab)
            unique_task_types = np.array(sorted(set(task_types)))
            task_types = np.array(task_types)

            # train test split to ensure the 80/10/10 splits
            train_indices, val_indices, test_indices = [], [], []
            for task_type in unique_task_types:
                task_indices = np.where(task_types == task_type)[0]
                size = len(task_indices)
                rs.shuffle(task_indices)
                train_size = int(train_frac * size)
                test_size = int(val_frac * size)
                val_size = size - train_size - test_size
                train_indices.extend(task_indices[:train_size].tolist())
                val_indices.extend(task_indices[train_size:train_size+val_size].tolist())
                test_indices.extend(task_indices[-test_size:].tolist())

        elif caller_intent == 'sentiment':
            sentiment_dicts = [row['sentiment'] for row in data]
            sentiment_logits = np.array([[
                sentiment_dict['positive'],
                sentiment_dict['neutral'],
                sentiment_dict['negative'],
            ] for sentiment_dict in sentiment_dicts])
            sentiments = np.argmax(sentiment_logits, axis=1)  # take biggest probability
            unique_sentiments = np.array(sorted(set(sentiments)))

            # train test split to ensure the 80/10/10 splits
            train_indices, val_indices, test_indices = [], [], []
            for sentiment in unique_sentiments:
                sentiment_indices = np.where(sentiments == sentiment)[0]
                size = len(sentiment_indices)
                rs.shuffle(sentiment_indices)
                train_size = int(train_frac * size)
                test_size = int(val_frac * size)
                val_size = size - train_size - test_size
                train_indices.extend(sentiment_indices[:train_size].tolist())
                val_indices.extend(sentiment_indices[train_size:train_size+val_size].tolist())
                test_indices.extend(sentiment_indices[-test_size:].tolist())

        elif caller_intent == 'dialog_acts':
            # randomly split!
            indices = np.arange(len(data))
            rs.shuffle(indices)
            size = len(indices)
            train_size = int(size * train_frac)
            val_size = int(size * val_frac)

            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:train_size+val_size].tolist()
            test_indices = indices[train_size+val_size:].tolist()

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        if return_indices:
            return (train_data, val_data, test_data), \
                    (train_indices, val_indices, test_indices)

        return train_data, val_data, test_data


class HarperValleyWav2VecTransfer(BaseHarperValleyTransfer):
    """Simplified dataset for Wav2Vec stuff"""

    def __init__(
            self,
            root,
            caller_intent,
            split='train',
            max_length=150526,
            min_utterance_length=4,
            min_speaker_utterances=10,
            prune_speakers=True,
        ):
        super().__init__(root, min_utterance_length, min_speaker_utterances, prune_speakers)
        assert caller_intent in [
            'speaker_id',
            'task_type',
            'dialog_acts',
            'sentiment',
        ]

        task_type_vocab, dialog_acts_vocab, speaker_ids = self.get_vocabs(self.data)

        (train_data, val_data, test_data), \
        (train_ix, val_ix, test_ix) = self.train_test_split_by_transfer(
            self.data, 
            caller_intent, 
            task_type_vocab, 
            dialog_acts_vocab, 
            speaker_ids,
            return_indices=True,
        )

        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        elif split == 'trainval':
            data = train_data + val_data
        elif split == 'test':
            data = test_data
        else:
            raise Exception(f'Split {split} not supported.')

        wavpaths  = [d['wavpath'] for d in data]

        if caller_intent == 'task_type':
            task_types = [d['task_type'] for d in data]
            labels = self.task_type_to_labels(task_types, task_type_vocab)
            num_class = len(set(labels))
        elif caller_intent == 'dialog_acts':
            dialog_acts = [d['dialog_acts'] for d in data]
            labels = self.dialog_acts_to_labels(dialog_acts, dialog_acts_vocab)
            num_class = len(labels[0])
        elif caller_intent == 'sentiment':
            sentiments = [d['sentiment'] for d in data]
            labels = sentiments
            num_class = len(sentiments[0])
        elif caller_intent == 'speaker_id':
            speaker_id = [d['speaker_id'] for d in data]
            labels = self.speaker_id_to_labels(speaker_id, speaker_ids)
            num_class = len(speaker_ids)
        else:
            raise Exception(f'Caller intent {caller_intent} not supported.')

        self.split = split
        self.wavpaths = wavpaths
        self.labels = labels
        self.num_class = num_class
        self.max_length = max_length
        self.caller_intent = caller_intent
        self.min_utterance_length = min_utterance_length
        self.min_speaker_utterances = min_speaker_utterances

    def __getitem__(self, index):
        wavpath = self.wavpaths[index]
        wav, sr = torchaudio.load(wavpath)
        wav = wav.squeeze().numpy()

        if self.split == 'train':
            transforms = WavformAugmentation(sr)
            wav = transforms(wav)

        # pad to 150k frames
        if len(wav) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.split == 'train' else True)
            padded = (wav[:self.max_length] if flip else
                      wav[-self.max_length:])
        else:
            padded = np.zeros(self.max_length)
            padded[:len(wav)] = wav  # pad w/ silence        
        
        label = self.labels[index]
        if self.caller_intent == 'dialog_acts':
            label = torch.LongTensor(label)
        elif self.caller_intent == 'sentiment':
            label = [label['positive'], label['neutral'], label['negative']]
            label = torch.FloatTensor(label)

        padded = torch.from_numpy(padded).float()
        return index, padded, label

    def __len__(self):
        return len(self.wavpaths)


class HarperValleyContrastiveTransfer(BaseHarperValleyTransfer):
    """Return log-Mel features along with Wav2Vec stuff"""

    def __init__(
            self,
            root,
            caller_intent,
            split='train',
            spectral_transforms=False,
            wavform_transforms=True,
            max_length=150526,
            input_size=112,
            normalize_mean=HARPER_VALLEY_MEAN,
            normalize_stdev=HARPER_VALLEY_STDEV,
            min_utterance_length=4,
            min_speaker_utterances=10,
            prune_speakers=True,
        ):
        super().__init__(root, min_utterance_length, min_speaker_utterances, prune_speakers)
        # choose to either apply augmentation at wavform or at augmentation level
        assert not (spectral_transforms and wavform_transforms)
        assert caller_intent in [
            'speaker_id',
            'task_type',
            'dialog_acts',
            'sentiment',
        ]

        task_type_vocab, dialog_acts_vocab, speaker_ids = self.get_vocabs(self.data)

        train_data, val_data, test_data  = self.train_test_split_by_transfer(
            self.data, 
            caller_intent, 
            task_type_vocab, 
            dialog_acts_vocab, 
            speaker_ids,
        )

        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        elif split == 'trainval':
            data = train_data + val_data
        elif split == 'test':
            data = test_data
        else:
            raise Exception(f'Split {split} not supported.')

        wavpaths  = [d['wavpath'] for d in data]

        if caller_intent == 'task_type':
            task_types = [d['task_type'] for d in data]
            labels = self.task_type_to_labels(task_types, task_type_vocab)
            num_class = len(set(labels))
        elif caller_intent == 'dialog_acts':
            dialog_acts = [d['dialog_acts'] for d in data]
            labels = self.dialog_acts_to_labels(dialog_acts, dialog_acts_vocab)
            num_class = len(labels[0])
        elif caller_intent == 'sentiment':
            sentiments = [d['sentiment'] for d in data]
            labels = sentiments
            num_class = len(sentiments[0])
        elif caller_intent == 'speaker_id':
            speaker_id = [d['speaker_id'] for d in data]
            labels = self.speaker_id_to_labels(speaker_id, speaker_ids)
            num_class = len(speaker_ids)
        else:
            raise Exception(f'Caller intent {caller_intent} not supported.')

        self.split = split
        self.wavpaths = wavpaths
        self.labels = labels
        self.num_class = num_class
        self.input_size = input_size
        self.wavform_transforms = wavform_transforms
        self.spectral_transforms = spectral_transforms
        self.max_length = max_length
        self.caller_intent = caller_intent
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev
        self.min_utterance_length = min_utterance_length
        self.min_speaker_utterances = min_speaker_utterances

    def __getitem__(self, index):
        wavpath = self.wavpaths[index]
        wav, sr = torchaudio.load(wavpath)
        wav = wav.squeeze().numpy()

        if self.wavform_transforms and self.split == 'train':
            transforms = WavformAugmentation(sr)
            wav = transforms(wav)

        # pad to 150k frames
        if len(wav) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.split == 'train' else True)
            padded = (wav[:self.max_length] if flip else 
                      wav[-self.max_length:])
        else:
            padded = np.zeros(self.max_length)
            padded[:len(wav)] = wav  # pad w/ silence

        spectrum = librosa.feature.melspectrogram(
            padded,
            sr,
            hop_length=HARPER_VALLEY_HOP_LENGTH_DICT[self.input_size],
            n_mels=self.input_size,
        )

        if self.spectral_transforms:  # apply time and frequency masks
            transforms = SpectrumAugmentation()
            spectrum = transforms(spectrum)

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        if self.spectral_transforms:  # apply noise on spectral
            noise_stdev = 0.25 * self.normalize_stdev[0]
            noise = torch.randn_like(spectrum) * noise_stdev
            spectrum = spectrum + noise

        label = self.labels[index]
        if self.caller_intent == 'dialog_acts':
            label = torch.LongTensor(label)
        elif self.caller_intent == 'sentiment':
            label = [label['positive'], label['neutral'], label['negative']]
            label = torch.FloatTensor(label)

        normalize = Normalize(self.normalize_mean, self.normalize_stdev)
        spectrum = normalize(spectrum)

        return index, spectrum, label

    def __len__(self):
        return len(self.wavpaths)
