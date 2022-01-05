from typing import Text, Dict, List, Union

import torch
from datasets import ClassLabel, load_dataset, load_metric
import datasets

import transformers
from transformers import AutoModel, AutoConfig, BertConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, DataCollator
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers import set_seed
from transformers import BertForTokenClassification
from transformers.pipelines import TextClassificationPipeline

import numpy as np

from src.relation_extraction import RelBase
from src.utils.EntityMarker import EntityMarker


class BertRelCLF(RelBase):
    def __init__(
            self,
            model_name_or_path=None,
            tokenizer=None,
            label2idx: Dict = None,
            max_seq_length: int = 512,
            device: str = None,
            marker_mode: Text = None
    ):
        self.label2idx = label2idx
        if model_name_or_path:
            if label2idx:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    num_labels=len(label2idx),
                    label2id=label2idx,
                    id2label={i: l for l, i in label2idx.items()}
                )

                if "marker_mode" in config.to_dict():
                    marker_mode = config.marker_mode
                else:
                    if marker_mode is None:
                        marker_mode = "entity"
                    config.update({"marker_mode": marker_mode})

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    config=config
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
                marker_mode = self.model.config.marker_mode

        self.entity_marker = EntityMarker(marker_mode=marker_mode)

        if tokenizer:
            self.tokenizer = tokenizer

        elif model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = None

        self.max_seq_length = max_seq_length

        self.label_list = list(self.model.config.label2id.keys())

        if device is None:
            self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.model:
            self.inferencer = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model)
        else:
            self.inferencer = None

    def run(self, text: Text, entities: List, **kwargs):
        kwargs["text"] = text
        kwargs["entities"] = entities
        kwargs["relations"] = []

        tokens = text.split(" ")
        if len(entities) < 2:
            return kwargs

        relations = []
        if len(entities) == 2:
            src_e = entities[0]
            trc_e = entities[1]
            token_marked = self.entity_marker.mark(src_entity=src_e, trg_entity=trc_e, tokens=tokens)
            text_marked = " ".join(token_marked)
            output = self.inferencer(text_marked)
            rel_label = output[0]["label"]
            score = output[0]["score"]
            if rel_label != "no_relation":
                relation = {
                    "source_entity": src_e,
                    "target_entity": trc_e,
                    "relation": rel_label,
                    "score": float(score)
                }
                relations.append(relation)
        else:
            for i in range(len(entities)-1):
                for j in range(i+1, len(entities)):
                    src_e = entities[i]
                    trc_e = entities[j]
                    token_marked = self.entity_marker.mark(src_entity=src_e, trg_entity=trc_e, tokens=tokens)
                    text_marked = " ".join(token_marked)
                    output = self.inferencer(text_marked)

                    rel_label = output[0]["label"]
                    score = output[0]["score"]
                    if rel_label != "no_relation":
                        relation = {
                            "source_entity": src_e,
                            "target_entity": trc_e,
                            "relation": rel_label,
                            "score": score
                        }
                        relations.append(relation)
        kwargs["relations"] = relations

        return kwargs

    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def train(
            self,
            train_dataset,
            eval_dataset,
            test_dataset=None,
            output_dir=None,
            learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            num_epochs=10,
            batch_size=8,
            evaluation_strategy="epoch",
            eval_steps=1000,
            save_strategy="no",
            fp16=False,
            **kwargs
    ):
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast)
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            fp16=fp16
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        train_result = trainer.train()

        if output_dir:
            train_metrics = train_result.metrics
            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)
            trainer.save_state()

            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            self.tokenizer.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)

        if test_dataset:
            test_result = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            print(test_result)

        return train_result

    def save(self, model_dir, **kwargs):
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path: Text, **kwargs):
        return cls(model_name_or_path=model_name_or_path, **kwargs)