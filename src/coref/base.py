from typing import Text


class CoreFBase:
    def train(self, **kwargs):
        NotImplementedError()

    def eval(self, **kwargs):
        NotImplementedError()

    def run(self, text: Text, **kwargs):
        """
        :param text: sentence or document
        :param kwargs:
        :return: text is coreference resolved
        """
        NotImplementedError()

