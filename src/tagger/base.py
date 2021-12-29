import os
from typing import Text
from src.data_reader import CoNLLReader
from src.evaluation.TagEvaluation import TagEvaluation
from tqdm import tqdm


class TaggerBase:
    def train(self, **kwargs):
        NotImplementedError()

    def evaluate(self, test_path=None, batch_size=None, result_dir=None, soft_eval=False, **kwargs):
        tag_evaluation = TagEvaluation()
        data_reader = CoNLLReader(test_path=test_path)
        examples = data_reader.get_examples("test")
        true_tags = []
        pred_tags = []
        for example in tqdm(examples):
            text = example.get_text()
            true_tag = example.get_bio_tags().split(' ')
            output = self.run(text)
            pred_tag = output["tags"]

            true_tags.append(true_tag)
            pred_tags.append(pred_tag)

        results = tag_evaluation.evaluate(
            true_tags=true_tags,
            pred_tags=pred_tags,
            result_dir=result_dir,
            soft_eval=soft_eval)
        return results

    def run(self, text: Text, **kwargs):
        NotImplementedError()