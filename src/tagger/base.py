import os
from typing import Text

import pandas as pd

from src.data_reader import CoNLLReader
from src.evaluation.TagEvaluation import TagEvaluation, compute_score
from src.utils.utils import write_json
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

        inc_examples = []
        mis_examples = []
        spu_examples = []
        fail_examples = []
        par_examples = []

        for example in tqdm(examples):
            text = example.get_text()
            true_tag = example.get_bio_tags()
            if isinstance(true_tag, str):
                true_tag = true_tag.split(' ')
            output = self.run(text)
            pred_tag = output["tags"]

            true_tags.append(true_tag)
            pred_tags.append(pred_tag)
            score = compute_score(y_true=true_tag, y_pred=pred_tag, tokens=text.split(' '))

            if score["incorrect"]:
                inc_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "incorrect_entities": score["incorrect"]
                })

            if score["missing"]:
                mis_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "missing_entities": score["missing"]
                })

            if score["spurius"]:
                spu_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "spurius_entities": score["spurius"]
                })

            if score["partial"]:
                par_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "partial_entities": score["partial"]
                })

            if true_tag != pred_tag:
                fail_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag
                })

        results = tag_evaluation.evaluate(
            true_tags=true_tags,
            pred_tags=pred_tags,
            result_dir=result_dir,
            soft_eval=soft_eval)

        if result_dir:
            inc_path = os.path.join(result_dir, "incorrect.json")
            miss_path = os.path.join(result_dir, "missing.json")
            spu_path = os.path.join(result_dir, "spurius.json")
            fail_path = os.path.join(result_dir, "fail.json")

            write_json(inc_examples, inc_path)
            write_json(mis_examples, miss_path)
            write_json(spu_examples, spu_path)
            write_json(fail_examples, fail_path)

        return results

    def run(self, text: Text, **kwargs):
        NotImplementedError()

    def save(self, model_dir):
        NotImplementedError()

    def load(self, model_dir):
        NotImplementedError()

    @classmethod
    def from_pretrained(cls, model_name_or_path: Text, **kwargs):
        NotImplementedError()