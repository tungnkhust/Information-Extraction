from typing import Text


class PipelineBase:
    def run(self, text: Text, **kwargs):
        pass

    def run_ner(self, text: Text, **kwargs):
        pass

    def evaluate(self, file_path, **kwargs):
        pass

    @classmethod
    def from_confg(cls, config):
        pass

    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass