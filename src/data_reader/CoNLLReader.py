import os
import ast
import pathlib
from typing import Union, Text, List
from pathlib import Path
from collections import Counter

from src.schema import InputExample, Token
from src.data_reader import BaseReader
from src.utils.utils import load_json, write_json


class CoNLLReader(BaseReader):
    def __init__(
            self,
            train_path: Union[Text, Path] = None,
            dev_path: Union[Text, Path] = None,
            test_path: Union[Text, Path] = None,
            rel_in: Text = "end"
    ):
        super(CoNLLReader, self).__init__()
        self.rel_in = rel_in
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.examples['train'] = self._load_examples(train_path)
        self.examples['dev'] = self._load_examples(dev_path)
        self.examples['test'] = self._load_examples(test_path)

    @staticmethod
    def parse_text_to_token(line):
        tmp = line.split('\t')
        index = int(tmp[0])
        text = tmp[1]
        bio_tag = tmp[2]
        rel_tags = ast.literal_eval(tmp[3])
        rel_indies = ast.literal_eval(tmp[4])
        return Token(
            index=index,
            text=text,
            bio_tag=bio_tag,
            rel_tags=rel_tags,
            rel_indies=rel_indies
        )

    def read_txt(self, file_path):
        examples = []

        with open(file_path, 'r') as pf:
            lines = pf.readlines()

            tokens = []
            example_id = None
            for line in lines:
                if line[0] == "#" or line[0] == "\n":
                    if line[0] == "#":
                        example_id = line
                    continue
                else:
                    token = self.parse_text_to_token(line)
                    if token.index == 0:
                        if tokens:
                            example = InputExample(id=example_id, tokens=tokens, rel_in=self.rel_in)
                            examples.append(example)
                        tokens = []
                    tokens.append(token)

            example = InputExample(id=example_id, tokens=tokens, rel_in=self.rel_in)
            examples.append(example)
        return examples

    def read_json(self, file_path):
        data = load_json(file_path)
        examples = []
        for sample in data:
            _tokens = sample["tokens"]
            _entities = sample["entities"]
            _relations = sample["relations"]
            _sample_id = sample["orig_id"]

            entities = []
            for e in _entities:
                entity = {
                    "entity": e["type"],
                    "start_token": e["start"],
                    "end_token": e["end"],
                    "value": " ".join(_tokens[e["start"]: e["end"]])
                }
                entities.append(entity)

            relations = []
            for r in _relations:
                relation = {
                    "source_entity": entities[r["head"]],
                    "target_entity": entities[r["tail"]],
                    "relation": r["type"]
                }
                relations.append(relation)
            input_example = InputExample.from_dict(
                {
                    "id": _sample_id,
                    "tokens": _tokens,
                    "entities": entities,
                    "relations": relations
                }
            )
            examples.append(input_example)
        return examples

    def _load_examples(self, file_path: Union[Text, Path, List], **kwargs) -> List[InputExample]:
        if file_path is None:
            return []

        if isinstance(file_path, List) is False:
            file_path = [file_path]

        examples = []

        for file in file_path:
            if os.path.exists(file):
                if file.endswith(".txt"):
                    examples.extend(self.read_txt(file))
                elif file.endswith(".json"):
                    examples.extend(self.read_json(file))
        return examples

    def get_dataset(self, split: Text = None, **kwargs):
        pass
