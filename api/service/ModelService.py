from src.pipelines.InfoExPipeline import InfoExPipeline
from src.tagger import BertNER
from src.relation_extraction import BertRelCLF
from typing import List, Dict, Text
from src.utils.utils import load_yaml


class ModelService:
    def __init__(
            self,
            config: Dict,
            pipeline,
            **kwargs
    ):
        self.config = config
        self.pipeline = pipeline

    async def run(self, text: Text, **kwargs):
        output = self.pipeline.run(text)
        entities = output["entities"]
        e_index = {}
        for i, e in enumerate(entities):
            e_index[f'{e["start_token"]}-{e["end_token"]}'] = i

        relations = output["relations"]

        for rel in relations:
            src_e = rel["source_entity"].copy()
            trg_e = rel["target_entity"].copy()
            src_e["index"] = e_index[f'{src_e["start_token"]}-{src_e["end_token"]}']
            trg_e["index"] = e_index[f'{trg_e["start_token"]}-{trg_e["end_token"]}']
            rel["source_entity"] = src_e
            rel["target_entity"] = trg_e

        output["relations"] = relations
        return output

    async def run_ner(self, text: Text, **kwargs):
        output = self.pipeline.run_ner(text)
        return output

    def get_version(self):
        return self.config["VERSION"]

    async def evaluate(self, examples, **kwargs):
        pass

    @classmethod
    def from_config(cls, config_path):
        config = load_yaml(config_path)
        pipeline = InfoExPipeline.from_confg(config["PIPELINE"])
        return cls(config, pipeline)

