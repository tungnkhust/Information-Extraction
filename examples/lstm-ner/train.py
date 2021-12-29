from src.tagger.LstmNER import LstmNER
from src.data_reader.CoNLLReader import CoNLLReader
from src.tagger.dataset_reader.CoNLLDatasetReader import CoNLLDatasetReader
from src.tagger.model.CRFTagger import CRFTagger
from src.utils.utils import load_yaml, write_yaml
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default="examples/lstm-ner/config.yaml")
    parser.add_argument("-rm_metric", "--rm_metric", default=False, action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config_path)
    ner = LstmNER.from_config(config)
    ner.train(config=config, rm_metric=args.rm_metric)
