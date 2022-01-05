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

