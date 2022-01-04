# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from filelock import FileLock

from transformers.models.auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from transformers.data.datasets.squad import  SquadFeatures, SquadDataset
from transformers.data.processors.squad import squad_convert_examples_to_features
from transformers import PreTrainedTokenizerBase
from src.schema import InputExample

import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset

from src.schema import InputExample
logger = logging.get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TagDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            examples: List[InputExample] = None,
            label2idx: Dict = None,
            label_all_tokens: bool = True,
            mode: str = "train"
    ):
        super(TagDataset, self).__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens
        self.mode = mode
        self.label2idx = label2idx
        self.features = self.tokenize_and_align_labels(examples)

    def tokenize_and_align_labels(self, examples: List[InputExample]):
        tokens = [example.get_tokens() for example in examples]

        tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True)

        labels = []
        for i, example in enumerate(examples):
            # print(examples["tokens"][i], label)
            label = [self.label2idx[bio_tag] for bio_tag in example.get_bio_tags()]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Convert to Tensors and build dataset
        input_ids = self.features["input_ids"][i]
        attention_mask = self.features["attention_mask"][i]
        labels = self.features["labels"][i]
        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return feature
