import os
from pathlib import Path
from typing import Union, Text, List
import logging
from collections import Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseReader:
    examples = {}

    def _load_examples(self, **kwargs):
        raise NotImplementedError()

    def get_examples(self, split: Text, **kwargs):
        return self.examples[split]

    def num_examples(self, split: Text, **kwargs):
        return len(self.get_examples(split=split))

    def count_entities(self, split: Text, **kwargs):
        examples = self.get_examples(split)
        counter = Counter()

        for example in examples:
            entities = example.get_entities()
            for e in entities:
                counter[e["entity"]] += 1

        return counter

    def count_relations(self, split: Text, **kwargs):
        examples = self.get_examples(split)
        r_counter = Counter()
        rdf_counter = Counter()
        for example in examples:
            relations = example.get_relations()
            for r in relations:
                head_entity = r["head_entity"]["entity"]
                tail_entity = r["tail_entity"]["entity"]
                relation = r["relation"]
                r_counter[relation] += 1
                rdf_counter[f'{head_entity}-{tail_entity}#{relation}'] += 1
        return r_counter, rdf_counter

    def get_dataset(self, split: Text = None, **kwargs):
        raise NotImplementedError()
