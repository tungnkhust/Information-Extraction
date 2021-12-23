from typing import Dict, Optional, List, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as f

from allennlp.common.checks import check_dimensions_match

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


class CRFTagger(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            label_namespace: str = "labels",
            feedforward: Optional[FeedForward] = None,
            label_encoding: Optional[str] = None,
            include_start_end_transitions: bool = None,
            constrain_crf_decoding: bool = None,
            calculate_span_f1: bool = True,
            dropout: Optional[float] = 0.3,
            verbose_metrics: bool = False,
            initializer: InitializerApplicator = InitializerApplicator(),
            top_k: int = 1,
            ignore_loss_on_o_tags: bool = False,
            **kwargs
    ):
        super().__init__(vocab=vocab, **kwargs)
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(self.label_namespace)
        self.encoder = encoder
        self.top_k = top_k
        self.include_start_end_transitions = include_start_end_transitions
        self._verbose_metrics = verbose_metrics

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feed_forward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()

        self.tag_projection_layer = TimeDistributed(nn.Linear(output_dim, self.num_tags))

        # if use crf decoder
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None

        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding

        if constrain_crf_decoding:
            if not label_encoding:
                raise KeyError("constrain_crf_decoding is True, but no label_encoding was specified")

            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            num_tags=self.num_tags,
            constraints=constraints,
            include_start_end_transitions=include_start_end_transitions
        )

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy_top_3": CategoricalAccuracy(top_k=top_k)
        }

        self.calculate_span_f1 = calculate_span_f1

        if calculate_span_f1:
            if not label_encoding:
                raise KeyError("calculate_span_f1 is True, but no label_encoding was specified")
            self._f1_metric = SpanBasedF1Measure(
                vocabulary=vocab,
                tag_namespace=label_namespace,
                label_encoding=label_encoding
            )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim"
        ),
        if feedforward is not None:
            check_dimensions_match(
                encoder.get_output_dim(),
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )

        initializer(self)

    def forward(
            self,
            tokens: TextFieldTensors,
            tags: torch.LongTensor = None,
            ignore_loss_on_o_tags: Optional[bool] = None
    ):
        embedding_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedding_text_input = self.dropout(embedding_text_input)

        encoded_text = self.encoder(embedding_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feed_forward:
            encoded_text = self._feed_forward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits=logits, mask=mask, top_k=self.top_k)
        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if self.top_k > 1:
            output["top_k_tags"] = best_paths

        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                crf_mask = mask & (tags != o_tag_index)
            else:
                crf_mask = mask

            log_likelihood = self.crf(logits, tags, crf_mask)
            output["loss"] = -log_likelihood

            class_probabilities = logits * 0.0

            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask)

            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask)

        return output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]
        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]

        if "top_k_tags" in output_dict:
            output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return
