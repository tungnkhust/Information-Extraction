from src.data_reader import CoNLLReader
from src.schema.schema import InputExample
from typing import Dict, Optional, List, Text, Sequence, Iterable

from allennlp.data import DatasetReader
from allennlp.data import TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.tokenizers import WhitespaceTokenizer, Tokenizer


class CoNLLDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            tag_label: str = "ner",
            label_namespace: str = "labels",
            tokenizer: Tokenizer = WhitespaceTokenizer(),
            **kwarg
    ):
        super().__init__(**kwarg)

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.tag_label = tag_label
        # self.feature_labels = feature_labels
        self.label_namespace = label_namespace
        self.__original_coding_scheme = "BIO"
        self.tokenizer = tokenizer

    def _read(self, examples: List[InputExample]) -> Iterable[Instance]:
        for example in examples:
            text = example.get_text()
            ner_tags = example.get_bio_tags()
            if isinstance(ner_tags, str):
                ner_tags = ner_tags.split(" ")
            yield self.text_to_instance(text, ner_tags)

    def text_to_instance(
            self,
            text: Text,
            ner_tags: List[str] = None
    ) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        sequence = TextField(tokens, self._token_indexers)
        instance_fields = {
            "tokens": sequence
        }
        if ner_tags:
            instance_fields["tags"] = SequenceLabelField(ner_tags, sequence, self.label_namespace)
        return Instance(instance_fields)


