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


from transformers import PreTrainedTokenizerBase
from src.schema import InputExample, Entity, Relation
from src.utils.EntityMarker import EntityMarker

import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset

from src.schema import InputExample
logger = logging.get_logger(__name__)


class RelDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            examples: List[InputExample] = None,
            label2idx: Dict = None,
            label_all_tokens: bool = True,
            mode: str = "train",
            marker_mode: str = "entity",
            max_length: int = 512,
            padding: str = "max_length"
    ):
        super(RelDataset, self).__init__()
        self.relations = self._get_relations(examples=examples)
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens
        self.mode = mode
        self.label2idx = label2idx
        self.marker = EntityMarker(marker_mode=marker_mode)
        self.max_length = max_length
        self.padding = padding

    def check_is_relation(self, src_entity, trg_entity, relations):
        if isinstance(src_entity, Entity):
            src_entity = src_entity.to_dict()

        if isinstance(trg_entity, Entity):
            trg_entity = trg_entity.to_dict()

        for rel in relations:
            if isinstance(rel, Relation):
                _src = rel.source_entity.to_dict()
                _trg = rel.target_entity.to_dict()
            else:
                _src = rel["source_entity"]
                _trg = rel["target_entity"]

            if src_entity["entity"] == _src["entity"] and trg_entity["entity"] == _trg["entity"]:
                if src_entity["start_token"] == _src["start_token"] or src_entity["end_token"] == _src["start_token"]:
                    return True

        return False

    def _get_relations(self, examples: List[InputExample]):
        if examples is None:
            return []
        relations = []
        for example in examples:
            tokens = example.get_tokens()
            entities = example.get_entities()
            _relations = example.get_relations()
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    src_e = entities[i]
                    trg_e = entities[j]
                    if self.check_is_relation(src_entity=src_e, trg_entity=trg_e, relations=_relations) is False:
                        relations.append({
                            "tokens": tokens,
                            "source_entity": src_e,
                            "target_entity": trg_e,
                            "relation": "no_relation"
                        })

            for relation in _relations:
                if isinstance(relation, Relation):
                    relation = relation.to_dict()
                relations.append({
                    "tokens": tokens,
                    "source_entity": relation["source_entity"],
                    "target_entity": relation["target_entity"],
                    "relation": relation["relation"]
                })

        return relations

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        relation = self.relations[i]
        # Convert to Tensors and build dataset
        tokens = relation["tokens"]
        src_entity = relation["source_entity"]
        trg_entity = relation["target_entity"]

        rel_label = self.label2idx[relation["relation"]]
        tokens = self.marker.mark(tokens=tokens, src_entity=src_entity, trg_entity=trg_entity)
        text = " ".join(tokens)
        tokenized_inputs = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=self.padding)
        tokenized_inputs["labels"] = rel_label
        return tokenized_inputs
