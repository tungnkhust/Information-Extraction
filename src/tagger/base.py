import os
from typing import Text

import pandas as pd

from src.data_reader import CoNLLReader
from src.evaluation.TagEvaluation import TagEvaluation, compute_score
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

        for example in tqdm(examples):
            text = example.get_text()
            true_tag = example.get_bio_tags().split(' ')
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
                    "fail_entities": str(score["incorrect"])
                })

            if score["missing"]:
                mis_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "fail_entities": str(score["missing"])
                })

            if score["spurius"]:
                spu_examples.append({
                    "text": text,
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "fail_entities": str(score["spurius"])
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
            inc_path = os.path.join(result_dir, "incorrect.csv")
            miss_path = os.path.join(result_dir, "missing.csv")
            spu_path = os.path.join(result_dir, "spurius.csv")
            fail_path = os.path.join(result_dir, "fail.csv")

            inc_df = pd.DataFrame(inc_examples)
            inc_df.to_csv(inc_path, index=False)

            miss_df = pd.DataFrame(mis_examples)
            miss_df.to_csv(miss_path, index=False)

            spu_df = pd.DataFrame(spu_examples)
            spu_df.to_csv(spu_path, index=False)

            fail_df = pd.DataFrame(fail_examples)
            fail_df.to_csv(fail_path, index=False)

        return results

    def run(self, text: Text, **kwargs):
        NotImplementedError()