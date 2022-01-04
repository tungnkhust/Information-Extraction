import os


class RelBase:
    def train(self, **kwargs):
        NotImplementedError()

    def eval(self, **kwargs):
        NotImplementedError()

    def run(self, **kwargs):
        NotImplementedError()