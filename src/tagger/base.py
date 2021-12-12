import os
from typing import Text


class TaggerBase:
    def train(self, **kwargs):
        NotImplementedError()

    def eval(self, **kwargs):
        NotImplementedError()

    def run(self, text: Text, **kwargs):
        NotImplementedError()