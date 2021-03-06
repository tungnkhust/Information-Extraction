from src.relation_extraction import RelBase
from src.tagger import TaggerBase
from src.coref import CoreFBase
from src.data_reader.CoNLLReader import CoNLLReader
from src.pipelines.base import PipelineBase
from src.utils.helpers import get_module
from typing import Text

from src.evaluation.RelEvaluation import RelEvaluation
from src.evaluation.TagEvaluation import TagEvaluation
from tqdm import tqdm


class InfoExPipeline(PipelineBase):
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
            output = self.rel_model.run(**kwargs)
            kwargs.update(output)

        return kwargs

    def run_ner(self, text: Text, **kwargs):
        if self.ner_model:
            output = self.ner_model.run(text, **kwargs)
            kwargs.update(output)
        return kwargs

    def evaluate(self, file_path, **kwargs):
        tag_evaluation = TagEvaluation()
        rel_evaluation = RelEvaluation()
        reader = CoNLLReader(test_path=file_path)
        examples = reader.get_examples("test")
        true_tags = []
        pred_tags = []

        true_relations = []
        pred_relations = []

        for example in tqdm(examples):
            output = self.run(example.get_text())
            true_tags.append(example.get_bio_tags())
            pred_tags.append(output["tags"])
            true_relations.append(example.get_relations())
            pred_relations.append(output["relations"])

        tag_result = tag_evaluation.evaluate(true_tags, pred_tags, result_dir=None)
        rel_result = rel_evaluation.evaluate(true_relations, pred_relations)
        results = dict()
        results["entity_result"] = tag_result
        results["relation_result"] = rel_result
        return results

    @classmethod
    def from_confg(cls, config):
        if "PIPELINE" in config:
            config = config["PIPELINE"]
        coref_cf = config["COREF"]
        if coref_cf is None:
            coref = None
        else:
            coref_class = get_module(coref_cf["package"], coref_cf["name"])
            coref = coref_class.from_pretrained(**coref_cf["params"])

        ner_cf = config["NER"]
        if ner_cf is None:
            ner = None
        else:
            ner_class = get_module(ner_cf["package"], ner_cf["name"])
            ner = ner_class.from_pretrained(**ner_cf["params"])

        rel_cf = config["REL"]
        if rel_cf is None:
            rel = None
        else:
            rel_class = get_module(rel_cf["package"], rel_cf["name"])
            rel = rel_class.from_pretrained(**rel_cf["params"])

        return cls(coref_model=coref, ner_model=ner, rel_model=rel)