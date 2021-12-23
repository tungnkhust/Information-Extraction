from src.tagger.LstmNER import LstmNER
from src.data_reader.CoNLLReader import CoNLLReader
from src.tagger.dataset_reader.CoNLLDatasetReader import CoNLLDatasetReader
from src.tagger.model.CRFTagger import CRFTagger
from src.utils.utils import load_yaml, write_yaml


config = load_yaml("examples/ner/config.yaml")
ner = LstmNER.from_config(config)
ner.train(config=config)
