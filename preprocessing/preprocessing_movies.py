import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
# from transformers import PreTrainedTokenizer
from preprocessing.vocabulary import GloveVocabulary


class MovieDataset(Dataset):
    def __init__(self, path: str, vocabulary: GloveVocabulary, aspect: int = 0,
                 max_length: int = 256, skip_long_sentence: bool = False,
                 sort_in_mini_batch: bool = True,
                 convert_label_to_binary_sentiment: bool = False,
                 balanced: bool = False,
                 only_load_first_sentence: bool = False
                 ):
        # self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.path = path
        # self.aspect = aspect
        self.sort_in_mini_batch = sort_in_mini_batch
        self.vocabulary = vocabulary
        # load data
        self.reviews = []
        self.labels = []
        self.data_max = 0
        # if self.convert_label_to_binary_sentiment:
        self.positive_indices = []
        self.negative_indices = []
        self.label_map = {
            'NEG': 0,
            'POS': 1,
        }
        with open(self.path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                _review = line['review']
                _label = line['label']
                label = self.label_map[_label]
                split_review = _review.split()
                # load a review
                if len(split_review) > max_length:
                    if skip_long_sentence:
                        continue
                    else:
                        temp_review_string = split_review[:max_length]
                else:
                    temp_review_string = split_review

                if only_load_first_sentence is True:
                    # Take the first sentence
                    temp_review_string = ' '.join(temp_review_string).split('.')[0].split(' ')

                self.data_max = max(self.data_max, len(temp_review_string))

                self.reviews.append(temp_review_string)
                self.labels.append(label)

        print(f'\nMovieDataset load to cpu memory finished.\n'
              f'MovieDataset.data_max:{self.data_max}\n'
              f'MovieDataset.len:{len(self.reviews)}')

        if balanced and convert_label_to_binary_sentiment:
            random.seed(20230815)
            print(f'Make the Training dataset class balanced. The Sample seed is 20230815(paper deadline)!')
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
                  f'BeerDataset.len:{len(self.reviews)}')

    def __getitem__(self, item):
        return self.reviews[item], self.scores[item]

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
        if self.convert_label_to_binary_sentiment:
            labels = np.array(labels)
        else:
            labels = np.array(labels, dtype=np.float32)

        if self.sort_in_mini_batch:  # required for LSTM
            sort_idx = np.argsort(lengths)[::-1]
            input_ids = input_ids[sort_idx]
            labels = labels[sort_idx]

        # input_ids, mask, labels
        return torch.from_numpy(input_ids), \
            torch.from_numpy(input_ids != self.vocabulary.pad_token_id), \
            torch.from_numpy(labels)

    def __len__(self):
        return len(self.reviews)


class BeerAnnotationDataset(Dataset):
    def __init__(self, path: str, vocabulary, aspect: int = -1, sort_in_mini_batch: bool = True,
                 convert_label_to_binary_sentiment: bool = False, max_length: int = None):
        """
        Loading Beer Advocate dataset with human-annotations, refer to the implementation:
        "Interpretable Neural Predictions with Differentiable Binary Variables"
        (https://aclanthology.org/P19-1284/)
        :param path: dataset path
        :param vocabulary: vocabulary of pretrained embedding
        :param aspect: specify an aspect in beer reviews (0-Appearance,1-Smell,2-Palate)
        :param max_length: maximum sentence length, which is not limited by default.
        :param sort_in_mini_batch: whether to sort by sentence length in mini_batch
        :param convert_label_to_binary_sentiment: whether to convert scores into binary label
        """
        self.vocabulary = vocabulary
        self.path = path
        self.aspect = aspect
        # load data
        self.reviews = []
        self.scores = []
        self.rationales = []
        self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.sort_in_mini_batch = sort_in_mini_batch
        self.max_length = max_length
        if convert_label_to_binary_sentiment:
            self.positive_nums = 0
            self.negative_nums = 0
        with open(self.path, 'rt', encoding='utf-8') as f:
            for line in f:
                data_description = json.loads(line)
                review = data_description['x']

                if max_length is not None:
                    if len(review) > max_length:
                        review = review[:max_length]

                scores = list(map(float, data_description['y']))
                annotations = data_description[f'{aspect}']
                if len(annotations) == 0:
                    continue

                if self.aspect > -1:
                    score = scores[self.aspect]
                    if convert_label_to_binary_sentiment:
                        if score <= 0.4:
                            score = 0
                            self.negative_nums += 1
                        elif score >= 0.6:
                            score = 1
                            self.positive_nums += 1
                        else:
                            continue

                self.reviews.append(review)
                self.scores.append(score)
                self.rationales.append(data_description[f'{aspect}'])
        print(f'\nBeerAnnotationDataset aspect{aspect} load finished.\n'
              f'BeerAnnotationDataset.len:{len(self.reviews)}')
        if convert_label_to_binary_sentiment:
            print(f'\n positive aspect{aspect} instance{self.positive_nums}.\n'
                  f'\n negative aspect{aspect} instance{self.negative_nums}.\n')

    def __getitem__(self, item):
        # input_ids, mask = self.vocabulary.encode(self.reviews[item], max_length=len(self.reviews[item]))
        # input_ids = torch.tensor(input_ids)
        # mask = torch.tensor(mask, dtype=torch.bool)
        # label = torch.tensor([self.scores[item]])
        return self.reviews[item], self.scores[item], self.rationales[item]

    def collate_fn(self, batch):
        """
        this function is for torch.utils.Dataloader.
        """
        lengths = np.array([len(item[0]) for item in batch])
        max_length = lengths.max()
        input_ids, labels = [], []
        rationales = torch.zeros((len(batch), max_length.item()), dtype=torch.long)
        for idx, item in enumerate(batch):
            input_ids.append(self.vocabulary.encode(item[0], max_length.item(), return_mask=False))
            labels.append([item[1]])
            for rationale_interval in item[2]:
                if max_length is not None:
                    if rationale_interval[0] >= max_length:
                        continue
                rationales[idx, rationale_interval[0]:rationale_interval[1]] = 1

        input_ids = np.array(input_ids)
        rationales = np.array(rationales)
        if self.convert_label_to_binary_sentiment:
            labels = np.array(labels)
        else:
            labels = np.array(labels, dtype=np.float32)

        if self.sort_in_mini_batch:  # required for LSTM
            sort_idx = np.argsort(lengths)[::-1]
            input_ids = input_ids[sort_idx]
            labels = labels[sort_idx]
            rationales = rationales[sort_idx]

        # input_ids, mask, labels, rationales
        return torch.from_numpy(input_ids), \
            torch.from_numpy(input_ids != self.vocabulary.pad_token_id), \
            torch.from_numpy(labels), torch.from_numpy(rationales)

    def __len__(self):
        return len(self.reviews)

    def get_rationale_average_ratio(self):
        """
        :return:  the average proportion of all rationales in corresponding sentences.
        """
        rationale_lens = []
        for idx, rationale_intervals in enumerate(self.rationales):
            dist = 0.
            for interval in rationale_intervals:
                dist += (interval[1] - interval[0])
            print(f'{idx}: dist is {dist},total len is {len(self.reviews[idx].split())}')
            rationale_lens.append(dist / len(self.reviews[idx].split()))
        return np.mean(rationale_lens)