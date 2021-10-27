import os
from pathlib import Path
from typing import Union, Text, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Token:
    def __init__(
            self,
            index: int = None,
            token: Text = None,
            bio_tag: Text = None,
            rel_tag: Text = None,
            rel_index: int = None,
            pos_tag: Text = None,
            chunking_tag: Text = None,

    ):
        self.index = index
        self.token = token
        self.bio_tag = bio_tag
        self.rel_tag = rel_tag
        self.rel_index = rel_index
        self.pos_tag = pos_tag
        self.chunking_tag = chunking_tag


class InputExample:
    def __init__(
            self,
            tokens: List[Token]

    ):
        self.tokens = tokens


class CoNLLReader:
    def __init__(
            self,
            train_path: Union[Text, Path] = None,
            dev_path: Union[Text, Path] = None,
            test_path: Union[Text, Path] = None
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path


