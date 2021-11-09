import os
import re
import ast
from typing import Union, Text, List
from pathlib import Path
from collections import Counter

from src.schema import InputExample, Token
from src.data_reader import BaseReader


class VLSPReader(BaseReader):
    def __init__(
            self,
            train_dir: Union[Text, Path] = None,
            dev_dir: Union[Text, Path] = None,
            test_dir: Union[Text, Path] = None,
            rel_in: Text = "begin"
    ):
        super(VLSPReader, self).__init__()
        self.rel_in = rel_in
        self.train_dir = train_dir
        self.dev_dir = dev_dir
        self.test_dir = test_dir
        self.examples['train'] = self._load_examples(train_dir)
        self.examples['dev'] = self._load_examples(dev_dir)
        self.examples['test'] = self._load_examples(test_dir)

    @staticmethod
    def parse_text_to_token(line, pre_token: Token = None):
        line = line.replace('\n', '')
        tmp = line.split('\t')
        try:
            index = int(tmp[0].split('-')[1]) - 1
        except :
            return "substitute_token"

        [start, end] = tmp[1].split('-')
        start = int(start)
        end = int(end)
        text = tmp[2]
        if tmp[3] == '_':
            entity_index = None
        else:
            entity_index = tmp[3][2:-1]

        try:
            if tmp[4] == '_':
                bio_tag = 'O'
            else:
                entity = tmp[4].replace(f'[{entity_index}]', '')
                if pre_token:
                    if pre_token.entity_index == entity_index:
                        bio_tag = "I-" + entity
                    else:
                        bio_tag = 'B-' + entity
                else:
                    bio_tag = 'B-' + entity
        except:
            bio_tag = "O"

        try:
            if tmp[5] == '_':
                rel_tags = ['N']
            elif tmp[5] == '':
                rel_tags = ['N']
            else:
                rel_tags = tmp[5].split('|')
                rel_tags = [rel.replace(' ', '') for rel in rel_tags]
        except:
            rel_tags = ['N']

        try:
            if tmp[6] == '_':
                rel_indies = [index]
            else:
                rel_index = re.sub(r'\[[\d_]+\]', '', tmp[6])
                rel_indies = rel_index.split('|')
                rel_indies = [int(rel[2:]) for rel in rel_indies]
        except:
            rel_indies = [index]
        return Token(
            index=index,
            text=text,
            bio_tag=bio_tag,
            rel_tags=rel_tags,
            rel_indies=rel_indies,
            start=start,
            end=end,
            entity_index=entity_index
        )

    def _load_examples(self, dir_path: Union[Text, Path], **kwargs) -> List[InputExample]:
        if dir_path is None:
            return []
        if os.path.exists(dir_path) is False:
            return []
        counter = Counter()
        filenames = os.listdir(dir_path)

        examples = []
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            tmp_file = os.listdir(file_path)[0]
            file_path = os.path.join(dir_path, filename, tmp_file)
            example_id = filename[:-6]

            with open(file_path, 'r') as pf:
                lines = pf.readlines()
                tokens = []

                for line in lines:
                    if "LOCATED" in line:
                        counter["LOCATED"] += 1
                    elif "PART – WHOLE" in line:
                        counter["PART–WHOLE"] += 1
                    elif "PERSONAL - SOCIAL" in line:
                        counter["PERSONAL-SOCIAL"] += 1
                    elif "AFFILIATION" in line:
                        counter["AFFILIATION"] += 1

                    if line[0] == "#" or line[0] == "\n":
                        continue
                    else:
                        if tokens:
                            pre_token = tokens[-1]
                        else:
                            pre_token = None

                        token = self.parse_text_to_token(line, pre_token)
                        if token == "substitute_token":
                            print(f"substitute_token in {example_id}, line:", line)
                            continue
                        tokens.append(token)

                example = InputExample(id=example_id, tokens=tokens, rel_in=self.rel_in)
                examples.append(example)
        print(counter)
        return examples

    def get_dataset(self, split: Text = None, **kwargs):
        pass
