import random
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.vocabulary import GloveVocabulary
import json

class SST2Dataset(Dataset):
    def __init__(self, path: str, vocabulary: GloveVocabulary,
                 max_length: int = 256,min_length: int = 10,
                 skip_long_sentence: bool = False,
                 skip_short_sentence: bool = False,
                 sort_in_mini_batch: bool = True,
                #  convert_label_to_binary_sentiment: bool = False,
                 balanced: bool = False,
                 only_load_first_sentence: bool = False):
        # self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.path = path
        # self.aspect = aspect
        self.sort_in_mini_batch = sort_in_mini_batch
        self.vocabulary = vocabulary
        # load data
        self.reviews = []
        self.scores = []
        self.data_max = 0
        self.positive_indices = []
        self.negative_indices = []
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for idx, item in enumerate(data):
            review = item['sentence']
            score = item['label']
            if skip_long_sentence and len(review.split()) > max_length:
                continue
            if skip_short_sentence and len(review.split()) < min_length:
                continue
            if int(score) == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
            self.reviews.append(review)
            self.scores.append(int(score))
            self.data_max = max(self.data_max, len(review.split()))
        print(f'\nSST2Dataset load to cpu memory finished.\n'
        f'SST2Dataset.data_max:{self.data_max}\n'
        f'SST2Dataset.len:{len(self.reviews)}')
        if balanced:
            random.seed(20220531)
            print(f'Make the Training dataset class balanced. The Sample seed is 20220531!')
            min_examples = min(len(self.positive_indices), len(self.negative_indices))
            if len(self.positive_indices) > min_examples:
                samples_to_pop = random.sample(self.positive_indices, len(self.positive_indices) - min_examples)
                print(f'Drop {len(samples_to_pop)} positive reviews!')
            else:
                samples_to_pop = random.sample(self.negative_indices, len(self.negative_indices) - min_examples)
                print(f'Drop {len(samples_to_pop)} negative reviews!')

            samples_to_pop = sorted(samples_to_pop, reverse=True)
            for sample_idx in samples_to_pop:
                self.reviews.pop(sample_idx)
                self.scores.pop(sample_idx)
            print(f'balance finished.\n'
                    f'SST2Dataset.len:{len(self.reviews)}')

    def __getitem__(self, item):
        return self.reviews[item], self.scores[item]
    def __len__(self):
        return len(self.reviews)

    def collate_fn(self, batch):
        """
        this function is for torch.utils.Dataloader.
        """
        lengths = np.array([len(item[0]) for item in batch])
        max_length = lengths.max()
        input_ids, labels = [], []
        for item in batch:
            input_ids.append(self.vocabulary.encode(item[0], max_length.item(), return_mask=False))
            labels.append([item[1]])
        input_ids = np.array(input_ids)
        labels = np.array(labels)
        # if self.convert_label_to_binary_sentiment:
        #     labels = np.array(labels)
        # else:
        #     labels = np.array(labels, dtype=np.float32)

        if self.sort_in_mini_batch:  # required for LSTM
            sort_idx = np.argsort(lengths)[::-1]
            input_ids = input_ids[sort_idx]
            labels = labels[sort_idx]
        # input_ids, mask, labels
        return torch.from_numpy(input_ids), \
            torch.from_numpy(input_ids != self.vocabulary.pad_token_id), \
            torch.from_numpy(labels)

class SST2AnnotationDataset(Dataset):
    def __init__(self, path: str, vocabulary, sort_in_mini_batch: bool = True, 
        max_length: int = None,min_length: int = None):
        self.vocabulary = vocabulary
        self.path = path
        # self.aspect = aspect
        # load data
        self.reviews = []
        self.scores = []
        self.rationales = []
        # self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.sort_in_mini_batch = sort_in_mini_batch
        self.max_length = max_length
        self.min_length = min_length
        self.positive_nums = 0
        self.negative_nums = 0
        
    def __getitem__(self, item):
        return self.reviews[item], self.scores[item], self.rationales[item]
