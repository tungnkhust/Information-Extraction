from src.pipelines.InfoExPipeline import InfoExPipeline
from src.tagger import BertNER
from src.relation_extraction import BertRelCLF
from argparse import ArgumentParser
from src.utils.utils import load_yaml

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="examples/pipeline/pipeline_config.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pipeline = InfoExPipeline.from_confg(config["PIPELINE"])
    text = "My name is Tung , I study at Standford"
    output = pipeline.run(text)
    print(output)