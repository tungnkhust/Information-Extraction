from typing import Text, Dict, List, Union

import torch
from datasets import ClassLabel, load_dataset, load_metric
import datasets

import transformers
from transformers import AutoModel, AutoConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers import set_seed
from transformers import BertForTokenClassification
from transformers.pipelines import pipeline

import numpy as np

from src.tagger import TaggerBase
from src.data_reader import CoNLLReader
from src.datasets.ConllDataset import Conll2003Dataset
from src.utils.utils import convert_entities_to_bio


class BertNER(TaggerBase):
    def __init__(
            self,
            model_name_or_path=None,
            tokenizer=None,
            label2idx: Dict = None,
            max_seq_length: int = 512,
            device: str = None,
            task_name="ner"
    ):
        self.label2idx = label2idx
        if model_name_or_path:
            if label2idx:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    num_labels=len(label2idx),
                    label2id=label2idx,
                    id2label={i: l for l, i in label2idx.items()},
                    finetuning_task=task_name
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name_or_path,
                    config=config
                )
            else:
                self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

        if tokenizer:
            self.tokenizer = tokenizer
        elif model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = None

        self.max_seq_length = max_seq_length

        self.label_list = list(self.model.config.label2id.keys())

        self.metric = load_metric("seqeval")
        self.task = "ner"
        self.label_all_tokens = True

        if device is None:
            self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.inferencer = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def run(self, text: Text, **kwargs):
        output = self.inferencer(text)
        entity = output[0]["entity"]
        s = output[0]["start"]
        e = output[0]["end"]
        entities = []
        for i, token in enumerate(output[1:]):
            if 'B-' in token["entity"]:
                entities.append({"entity": entity[2:], "start": s, "end": e, "value": text[s:e]})
                s = token["start"]
                e = token["end"]
                entity = token["entity"]
            elif "I-" in token["entity"]:
                e = token["end"]
                if i == len(output) - 2:
                    entities.append({"entity": entity[2:], "start": s, "end": e, "value": text[s:e]})

        tags = convert_entities_to_bio(entities=entities, text=text)
        kwargs["text"] = text
        kwargs["tags"] = tags
        kwargs["entities"] = entities
        return kwargs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)

        return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

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

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

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
            return train_result

        if test_dataset:
            test_result = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            print(test_result)

    def save(self, model_dir):
        self.model.save_pretrained(model_dir)