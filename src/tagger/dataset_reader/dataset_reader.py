from src.data_reader import CoNLLReader
from src.schema import InputExample
from typing import Dict, Optional, List, Text, Sequence, Iterable

from allennlp.data import DatasetReader
from allennlp.data import TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField


class CoNLLDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            tag_label: str = "ner",
            feature_labels: Sequence(str) = (),
            label_namespace: str = "labels",
            **kwarg
    ):
        super().__init__(**kwarg)

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.tag_label = tag_label
        self.feature_labels = feature_labels
        self.label_namespace = label_namespace
        self.__original_coding_scheme = "IOB1"

    def _read(self, examples: List[InputExample]) -> Iterable[Instance]:
        for example in examples:
            tokens = example.get_text().split(" ")
            ner_tags = example.get_bio_tags().split(" ")
            tokens = [Token(token) for token in tokens]
            yield self.text_to_instance(tokens, ner_tags)

    def text_to_instance(
            self,
            tokens: List[Token],
            ner_tags:List[str] = None
    ) -> Instance:
        sequence = TextField(tokens)
        instance_fields = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})



