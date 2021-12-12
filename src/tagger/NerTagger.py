from typing import Text

from src.tagger import TaggerBase


class NerTagger(TaggerBase):
    def __init__(
            self,
            model=None,
            **kwargs
    ):
        self.model = model

    def run(self, text: Text, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def eval(self, **kwargs):
        pass