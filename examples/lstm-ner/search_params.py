from src.tagger.LstmNER import LstmNER
from src.data_reader.CoNLLReader import CoNLLReader
from src.tagger.dataset_reader.CoNLLDatasetReader import CoNLLDatasetReader
from src.tagger.model.CRFTagger import CRFTagger
from src.utils.utils import load_yaml, write_yaml, write_json
from argparse import ArgumentParser


def update_config(config, key, value, w2v_base=None):
    if key == "embedding_dim":
        config["MODEL"]["input_features"]["encoder"]["args"]["embedding"]["word"]["embedding_dim"] = value
        if w2v_base:
            w2v_base = w2v_base.format(value)
        config["MODEL"]["input_features"]["encoder"]["args"]["embedding"]["word"]["pretrained_file"] = w2v_base
    elif key == "num_filters":
        config["MODEL"]["input_features"]["encoder"]["args"]["embedding"]["character"]["num_filters"] = value
    elif key == "dropout":
        config["MODEL"]["input_features"]["encoder"]["args"]["sequence_encoder"]["dropout"] = value
    elif key == "hidden_size":
        config["MODEL"]["input_features"]["encoder"]["args"]["sequence_encoder"]["hidden_size"] = value
    elif key == "num_layers":
        config["MODEL"]["input_features"]["encoder"]["args"]["sequence_encoder"]["num_layers"] = value
    elif key == "feedforward":
        config["MODEL"]["input_features"]["encoder"]["args"]["sequence_encoder"]["feedforward"]["add_feedforward"] = value
    elif key == "grad_clipping":
        config["TRAINING_MODEL"]["hyper_params"]["grad_clipping"] = value
    return config


def search_params(config, grid_hyperparams, w2v_base=None):
    result = {}
    best_config = config.copy()
    best_score = -1
    best_params = {}

    for key, values in grid_hyperparams.items():
        for value in values:
            _config = update_config(best_config.copy(), key, value, w2v_base)
            print("x"*90)
            print(_config)
            ner = LstmNER.from_config(_config)
            final_train_result = ner.train(config=_config, rm_metric=True)
            best_validation_f1 = final_train_result["best_validation_f1-measure-overall"]
            if best_validation_f1 > best_score:
                best_score = best_validation_f1
                best_config = _config.copy()
                best_params[key] = value

            if key not in result:
                result[key] = {}

            result[key][value] = final_train_result

    return {
        "best_params": best_params,
        "best_config": best_config,
        "result": result
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", default="examples/lstm-ner/pipeline_config.yaml")
    parser.add_argument("-rm_metric", "--rm_metric", default=False, action="store_true")
    parser.add_argument("-w2v_base", "--w2v_base", default="pre-trained/glove/glove.6B.{}d.txt")
    args = parser.parse_args()
    config = load_yaml(args.config_path)

    grid_hyperparams = {
        "embedding_dim": [50, 100, 200],
        "num_filters": [50, 100, 128],
        "dropout": [0.3, 0.34, 0.4, 0.5],
        "hidden_size": [100, 128, 200, 256],
        "num_layers": [2, 3],
        "feedforward": [False, True],
        "grad_clipping": [5.0, 10.0]
    }

    # grid_hyperparams = {
    #     "embedding_dim": [50, 100],
    #     "num_filters": [50],
    #     "dropout": [0.3],
    #     "hidden_size": [100],
    #     "num_layers": [2],
    #     "feedforward": [True],
    #     "grad_clipping": [5.0]
    # }

    output = search_params(config=config, grid_hyperparams=grid_hyperparams, w2v_base=args.w2v_base)
    print(output["best_params"])
    write_json(output, 'models/lstm-ner-search-params.json')





