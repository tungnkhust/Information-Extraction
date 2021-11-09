import os
import ast
from typing import Union, Text, List
from pathlib import Path
from collections import Counter

from src.schema import InputExample, Token
from src.data_reader import BaseReader


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

    def _load_examples(self, file_path: Union[Text, Path], **kwargs) -> List[InputExample]:
        if file_path is None:
            return []

        if os.path.exists(file_path) is False:
            return []

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

    def get_dataset(self, split: Text = None, **kwargs):
        pass
