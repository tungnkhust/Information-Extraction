import os
import ast
from typing import Union, Text, List
from pathlib import Path
from collections import Counter, defaultdict

from src.schema import InputExample, Token, Relation, Entity
from src.data_reader import BaseReader


class CoNLL2004Reader(BaseReader):
    def __init__(
            self,
            train_path: Union[Text, Path] = None,
            dev_path: Union[Text, Path] = None,
            test_path: Union[Text, Path] = None,
            rel_in: Text = "end",
            col_label: List = ["id", "ner", "index", "unk", "pos", "token"]
    ):
        super(CoNLL2004Reader, self).__init__()
        self.rel_in = rel_in
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.examples['train'] = self._load_examples(train_path)
        self.examples['dev'] = self._load_examples(dev_path)
        self.examples['test'] = self._load_examples(test_path)
        self.col_label = col_label

    @staticmethod
    def parse_text_to_token(line):
        tmp = line.split('\t')
        index = int(tmp[2])
        text = tmp[5]
        bio_tag = tmp[1]
        pos_tag = tmp[4]
        return Token(
            index=index,
            text=text,
            bio_tag=bio_tag,
            pos_tag=pos_tag
        )

    def format_bio_tag(self, tokens: List[Token]):
        if tokens[0].bio_tag != "O":
            tokens[0].bio_tag = f'B-{tokens[0].bio_tag}'
        for token in tokens[1:]:
            index = token.index
            bio = token.bio_tag
            if bio != "O":
                if bio == tokens[index-1].bio_tag:
                    token.bio_tag = f'I-{bio}'
                else:
                    token.bio_tag = f'B-{bio}'

        return tokens

    def _load_examples(self, file_path: Union[Text, Path], **kwargs) -> List[InputExample]:
        if file_path is None:
            return []

        if os.path.exists(file_path) is False:
            return []

        input_examples = []

        with open(file_path, 'r') as pf:
            lines = pf.readlines()

            examples = {}

            example_id = None
            _relations = []

            for line in lines:
                line = line.replace('\n', '')
                if line == '\n' or line == '':
                    continue
                elif len(line.split('\t')) == 3:
                    examples[example_id]["relations"].append(line.split('\t'))
                    continue
                else:
                    example_id = line.split('\t')[0]
                    if example_id not in examples:
                        examples[example_id] = {"tokens": [], "relations": []}
                    examples[example_id]["tokens"].append(self.parse_text_to_token(line))
            n_relation = 0
            for _id in examples:
                tokens = examples[_id]["tokens"]
                tokens = self.format_bio_tag(tokens)
                _relations = examples[_id]["relations"]
                input_example = InputExample(id=_id, tokens=tokens, rel_in_tag=False)
                relations = []
                for _rel in _relations:
                    n_relation += 1
                    src_e = input_example.get_entity(start_token=int(_rel[0]))[0]
                    trg_e = input_example.get_entity(start_token=int(_rel[1]))[0]
                    rel_label = _rel[2]
                    relation = Relation(source_entity=src_e, target_entity=trg_e, relation=rel_label)
                    relations.append(relation)

                input_example.relations = relations
                input_examples.append(input_example)
                if len(tokens) < 50 and len(relations) > 0:
                    print(input_example)
        return input_examples

    def get_bio_list(self):
        label_list = []
        for example in self.examples["train"]:
            bio_tags = example.get_bio_tags()
            for tag in bio_tags.split(" "):
                if tag not in label_list:
                    label_list.append(tag)

        for example in self.examples["dev"]:
            bio_tags = example.get_bio_tags()
            for tag in bio_tags.split(" "):
                if tag not in label_list:
                    label_list.append(tag)

        for example in self.examples["test"]:
            bio_tags = example.get_bio_tags()
            for tag in bio_tags.split(" "):
                if tag not in label_list:
                    label_list.append(tag)

        label_list.sort()

        return label_list

    def get_dataset(self, split: Text = None, **kwargs):
        pass
