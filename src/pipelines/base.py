from src.relation_extraction import RelBase
from src.tagger import TaggerBase
from src.coref import CoreFBase
from src.data_reader.CoNLLReader import CoNLLReader


class InformationPipeline:
    def __init__(
            self,
            coref_model: CoreFBase = None,
            ner_model: TaggerBase = None,
            rel_model: RelBase = None
    ):
        self.coref_model = coref_model
        self.ner_model = ner_model
        self.rel_model = rel_model

    def run(self, text, **kwargs):
        if self.coref_model:
            text = self.coref_model.run(text)

        kwargs["text"] = text
        if self.ner_model:
            output = self.ner_model.run(**kwargs)
            kwargs.update(output)

        if self.rel_model:
            output = self.ner_model.run(**kwargs)
            kwargs.update(output)

        return kwargs

    def evaluate(self, file_path, **kwargs):
        predictions = []
        reader = CoNLLReader(test_path=file_path)
        examples = reader.get_examples("test")
        for example in examples:
            output = self.run(example.get_text())

