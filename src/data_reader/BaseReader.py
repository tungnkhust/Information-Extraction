import os
from pathlib import Path
from typing import Union, Text, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseReader:
    def _load_examples(self, file_path: Union[Text, Path], **kwargs):
        raise NotImplementedError()

    def get_examples(self, **kwargs):
        raise NotImplementedError()

    def count_entities(self, **kwargs):
        raise NotImplementedError()

    def count_relations(self, **kwargs):
        raise NotImplementedError()

    def num_examples(self, **kwargs):
        raise NotImplementedError()

    def get_dataset(self, split: Text = None, **kwargs):
        raise NotImplementedError()