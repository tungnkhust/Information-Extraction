import os.path
from typing import Text, Dict, List, Union, Optional
import shutil
import numpy as np
from scipy.special import softmax

import torch
import torch.nn.functional as f

from src.tagger import TaggerBase
from src.data_reader import BaseReader
from src.data_reader import CoNLLReader
from src.tagger.dataset_reader import CoNLLDatasetReader
from src.tagger.model import CRFTagger
from src.utils.utils import load_json, write_json
from src.utils.utils import load_yaml, write_yaml


from allennlp.training.util import evaluate
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.optimizers import AdamOptimizer, AdamWOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.modules.seq2seq_encoders import GruSeq2SeqEncoder, RnnSeq2SeqEncoder, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, LstmSeq2VecEncoder, GruSeq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import archive_model, Archive


class LstmNER(TaggerBase):
    def __init__(
            self,
            model: Model = None,
            dataset_reader: DatasetReader = None,
            data_reader: CoNLLReader = None,
            device: str = None,
            **kwargs
    ):
        self.model = model
        self.dataset_reader = dataset_reader
        self.data_reader = data_reader
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if "cuda" in device:
            cuda_device = 0
        else:
            cuda_device = -1

        self.device = torch.device(device)
        self.cuda_device = cuda_device

    def run(self, text: Text, **kwargs):
        instance = self.dataset_reader.text_to_instance(text)
        output = self.model.forward_on_instance(instance)
        tags = output["tags"]
        kwargs["tags"] = tags
        kwargs["text"] = text
        return kwargs

    def train(
            self,
            config=None,
            lr=0.001,
            weight_decay=0.0001,
            batch_size=64,
            num_epochs=50,
            grad_clipping=5,
            train_path=None,
            dev_path=None,
            test_path=None,
            serialization_dir=None,
            model_name="ner",
            rm_metric=False,
            **kwargs
    ):
        if serialization_dir and os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)

        parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if config:
            optimizer_name = config["TRAINING_MODEL"]["hyper_params"]["optimize"]["optimizer"]
            lr = config["TRAINING_MODEL"]["hyper_params"]["optimize"]["learning_rate"]
            weight_decay = config["TRAINING_MODEL"]["hyper_params"]["optimize"]["weight_decay"]
            serialization_dir = config["TRAINING_MODEL"]["base_path"]
            model_name = config["TRAINING_MODEL"]["model_name"]
            if optimizer_name == "adamW":
                optimizer = AdamWOptimizer(parameters, lr=lr, weight_decay=weight_decay)
            else:
                optimizer = AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay)
            batch_size = config["TRAINING_MODEL"]["hyper_params"]["batch_size"]
            num_epochs = config["TRAINING_MODEL"]["hyper_params"]["num_epochs"]
            grad_clipping = config["TRAINING_MODEL"]["hyper_params"]["grad_clipping"]
        else:
            optimizer = AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay)
        train_data = list(self.dataset_reader.read(self.data_reader.get_examples("train")))
        val_data = list(self.dataset_reader.read(self.data_reader.get_examples("dev")))

        train_loader = SimpleDataLoader(train_data, batch_size=batch_size)
        train_loader.index_with(self.model.vocab)
        val_loader = SimpleDataLoader(val_data, batch_size=batch_size)
        val_loader.index_with(self.model.vocab)

        model_dir = os.path.join(serialization_dir, model_name)
        serialization_dir = os.path.join(model_dir, "training")

        print('='*80)
        print(self.model)
        print('_' * 40)
        print(optimizer)
        print('=' * 80)

        trainer = GradientDescentTrainer(
            model=self.model,
            data_loader=train_loader,
            validation_data_loader=val_loader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            grad_clipping=grad_clipping,
            cuda_device=self.cuda_device,
            serialization_dir=serialization_dir
        )

        trainer.train()
        if model_dir:
            vocabulary_dir = os.path.join(model_dir, "vocabulary")
            config_path = os.path.join(model_dir, "config.json")
            self.model.vocab.save_to_files(vocabulary_dir)
            config["VOCAB"]["vocabulary_dir"] = vocabulary_dir
            config["VOCAB"]["extend_from_data"] = False
            config["DATASET"]["train_path"] = None
            config["DATASET"]["dev_path"] = None
            config["DATASET"]["test_path"] = None
            print(f"Save config in {config_path}")
            write_json(config, config_path)
            shutil.move(os.path.join(serialization_dir, "best.th"), os.path.join(model_dir, "best.th"))

            if rm_metric:
                shutil.rmtree(serialization_dir)

    # def evaluate(self, test_path=None, batch_size=64, **kwargs):
    #     data_reader = CoNLLReader(test_path=test_path)
    #     if test_path is not None:
    #         test_data = list(self.dataset_reader.read(data_reader.get_examples("test")))
    #     else:
    #         raise Exception("Don't have test data please pass test_path arguments")
    #
    #     test_loader = SimpleDataLoader(test_data, batch_size=batch_size, shuffle=False)
    #     test_loader.index_with(self.model.vocab)
    #     self.model.eval()
    #     results = evaluate(self.model.to(self.device), test_loader)
    #     return results

    @classmethod
    def from_config(cls, config: Dict, weight_path=None, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        data_cfg = config["DATASET"]
        data_reader = CoNLLReader(
            train_path=data_cfg.get("train_path", None),
            dev_path=data_cfg.get("dev_path", None),
            test_path=data_cfg.get("test_path", None)
        )
        model_clf = config["MODEL"]
        levels = model_clf["input_features"]["level"]
        if len(levels) == 0:
            raise Exception("Level of input features must be specific")

        # create dataset reader
        token_indexers = dict()
        for level in set(levels):
            if level == "word":
                token_indexers["tokens"] = SingleIdTokenIndexer(namespace="tokens")
            elif level == "character":
                token_indexers["token_characters"] = TokenCharactersIndexer(
                        namespace="token_characters",
                        min_padding_length=3
                    )
        dataset_reader = CoNLLDatasetReader(token_indexers=token_indexers)

        # create vocab
        vocab_cfg = config["VOCAB"]
        extend_from_data = vocab_cfg["extend_from_data"]
        vocabulary_dir = vocab_cfg.get("vocabulary_dir", None)

        if vocabulary_dir:
            vocab = Vocabulary.from_files(vocabulary_dir)
            print(f"Load vocab from {vocabulary_dir}")
        else:
            vocab = Vocabulary.from_instances(dataset_reader.read(data_reader.get_examples("train")))
            print("Build vocab from data")
        if extend_from_data and vocabulary_dir is not None:
            vocab.from_instances(dataset_reader.read(data_reader.get_examples("train")))
            print("Extend vocab from data")
        print(vocab)

        # create model
        # embedding
        encoder_cfg = model_clf["input_features"]["encoder"]["args"]
        embed_cfg = encoder_cfg["embedding"]
        text_field_embedder = {}
        for key in embed_cfg:
            if key in levels:
                if key == "word":
                    text_field_embedder["tokens"] = Embedding(
                            embedding_dim=embed_cfg[key]["embedding_dim"],
                            vocab_namespace='tokens',
                            vocab=vocab,
                            pretrained_file=embed_cfg[key]["pretrained_file"])

                    if embed_cfg[key]["pretrained_file"]:
                        print(f"pretrain word embedding: {embed_cfg[key]['pretrained_file']}")

                elif key == "character":
                    char_embedding = Embedding(
                            embedding_dim=embed_cfg[key]["embedding_dim"],
                            vocab_namespace='token_characters',
                            vocab=vocab,
                            pretrained_file=embed_cfg[key]["pretrained_file"])
                    if embed_cfg[key]["encoder_type"] == "cnn":
                        char_encoder = CnnEncoder(
                                embedding_dim=embed_cfg[key]["embedding_dim"],
                                num_filters=embed_cfg[key]["num_filters"],
                                ngram_filter_sizes=tuple(embed_cfg[key]["ngram_filter_sizes"])
                        )
                    elif embed_cfg[key]["encoder_type"] == "gru":
                        char_encoder = GruSeq2VecEncoder(
                                input_size=embed_cfg[key]["embedding_dim"],
                                hidden_size=embed_cfg[key]["num_filters"],
                                num_layers=2
                        )
                    else:
                        char_encoder = LstmSeq2VecEncoder(
                            input_size=embed_cfg[key]["embedding_dim"],
                            hidden_size=embed_cfg[key]["num_filters"],
                            num_layers=2
                        )
                    character_encoder = TokenCharactersEncoder(char_embedding, char_encoder)
                    text_field_embedder["token_characters"] = character_encoder
        text_field_embedder = BasicTextFieldEmbedder(token_embedders=text_field_embedder)

        # sequence encoder
        seq_encoder_cfg = encoder_cfg["sequence_encoder"]
        rnn_type = seq_encoder_cfg.get("rnn_type", "lstm")
        dropout = seq_encoder_cfg.get("dropout", None)
        bidirectional = seq_encoder_cfg.get("bidirectional", True)
        hidden_size = seq_encoder_cfg.get("hidden_size", 200)
        num_layers = seq_encoder_cfg.get("num_layers", 2)

        if rnn_type == "gru":
            encoder = GruSeq2SeqEncoder(
                        input_size=text_field_embedder.get_output_dim(),
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        dropout=dropout)
        else:
            encoder = LstmSeq2SeqEncoder(
                        input_size=text_field_embedder.get_output_dim(),
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        dropout=dropout)

        decoder_cfg = config["MODEL"]["output_features"]["decoder"]
        if "crf" in decoder_cfg:
            constrain_crf_decoding = decoder_cfg["crf"].get("constrain_crf_decoding", False)
        else:
            constrain_crf_decoding = False

        base_path = config["TRAINING_MODEL"]["base_path"]
        model_name = config["TRAINING_MODEL"]["model_name"]
        model_path = os.path.join(base_path, model_name)

        model = CRFTagger(
            vocab=vocab,
            label_encoding=model_clf["output_features"]["label_encoding"],
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            top_k=1,
            dropout=dropout,
            serialization_dir=model_path,
            calculate_span_f1=config["TRAINING_MODEL"]["metrics"]["calculate_span_f1"],
            verbose_metrics=config["TRAINING_MODEL"]["metrics"]["verbose_metrics"],
            constrain_crf_decoding=constrain_crf_decoding,
            label_namespace=config["VOCAB"]["label_namespace"]
        )

        if weight_path and os.path.exists(weight_path):
            if device == "cpu":
                model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
            else:
                model.load_state_dict(torch.load(weight_path))

            print(f"Load model weight from {weight_path}\n")

        return cls(
            model=model,
            dataset_reader=dataset_reader,
            data_reader=data_reader,
            device=None
        )

    @classmethod
    def from_pretrained(cls, model_dir: Text):
        config_path = os.path.join(model_dir, "config.json")
        config_path = os.path.abspath(config_path)
        config = load_json(config_path)
        vocab_dir = os.path.join(model_dir, "vocabulary")
        config["VOCAB"]["vocabulary_dir"] = vocab_dir
        weight_path = os.path.join(model_dir, "best.th")
        return cls.from_config(config=config, weight_path=weight_path)
