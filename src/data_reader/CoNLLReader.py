import os
from typing import Union, Text, List
from pathlib import Path
from collections import Counter

from src.schema import InputExample, Token
from src.data_reader import BaseReader


class CoNLLReader(BaseReader):
    examples = {}

    def __init__(
            self,
            train_path: Union[Text, Path] = None,
            dev_path: Union[Text, Path] = None,
            test_path: Union[Text, Path] = None
    ):
        super(CoNLLReader, self).__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.examples['train'] = self._load_examples(train_path)
        self.examples['dev'] = self._load_examples(dev_path)
        self.examples['test'] = self._load_examples(test_path)

    def _load_examples(self, file_path: Union[Text, Path], **kwargs) -> List[InputExample]:
        if file_path and os.path.exists(file_path) is False:
            return []

        with open(file_path, 'r') as pf:
            lines = pf.readlines()
            examples = []
            tokens = []
            example_id = None
            for line in lines:
                if line[0] == "#" or line[0] == "\n":
                    if line[0] == "#":
                        example_id = line
                    continue
                else:
                    token = Token.from_text(line)
                    if token.index == 0:
                        if tokens:
                            example = InputExample(id=example_id, tokens=tokens)
                            examples.append(example)
                        tokens = []
                    tokens.append(token)

            example = InputExample(id=example_id, tokens=tokens)
            examples.append(example)
            return examples

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
