import os
import pandas as pd
from typing import Text, List

from src.data_reader import CoNLLReader
from src.evaluation.RelEvaluation import RelEvaluation
from tqdm import tqdm


class RelBase:
    def train(self, **kwargs):
        NotImplementedError()

    def evaluate(self, test_path, has_direction=False, **kwargs):
        reader = CoNLLReader(test_path=test_path)
        examples = reader.get_examples("test")
        evaluation = RelEvaluation(has_direction=has_direction)
        true_relations = []
        pred_relations = []
        for example in tqdm(examples):
            text = example.get_text()
            entities = example.get_entities()
            output = self.run(text=text, entities=entities)
            pre_rels = output["relations"]
            true_rels = example.get_relations()
            true_relations.append(true_rels)
            pred_relations.append(pre_rels)
        scores = evaluation.evaluate(true_relations=true_relations, pred_relations=pred_relations)
        return scores

    def run(self, text: Text, entities: List, **kwargs):
        """

        :param text:
        :param entities:
        :param kwargs:
        :return:
            {
                "text" : "",
                "entities": [],
                "relations": []
            }
        """
        NotImplementedError()

    def save(self, model_dir: Text, **kwargs):
        NotImplementedError()

    def load(self, model_dir: Text, **kwargs):
        NotImplementedError()

    @classmethod
    def from_pretrained(cls, model_name_or_path: Text, **kwargs):
        NotImplementedError()