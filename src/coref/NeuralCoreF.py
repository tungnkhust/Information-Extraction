import spacy
import neuralcoref

from typing import Text
import os

from src.coref.base import CoreFBase


class NeuralCoreF(CoreFBase):
    def __init__(
            self,
            language: Text = "en"
    ):
        try:
            self.model = spacy.load(language)
        except:
            os.system(f"python -m spacy download {language}")
            self.model = spacy.load("en")

        neuralcoref.add_to_pipe(self.model)

    def run(self, text: Text, **kwargs):
        doc = self.model(text)

        if doc._.has_coref:
            return doc._.coref_resolved
        else:
            return text

