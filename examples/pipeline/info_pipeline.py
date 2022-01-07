from src.pipelines.InfoExPipeline import InfoExPipeline
from src.tagger import BertNER
from src.relation_extraction import BertRelCLF
from argparse import ArgumentParser
from src.utils.utils import load_yaml
import time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="examples/pipeline/pipeline_config.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pipeline = InfoExPipeline.from_confg(config["PIPELINE"])
    text = "Hồ Chí Minh led the Việt Minh independence movement from 1941 onward. It was supposed to be an umbrella group" \
       " for all parties fighting for Vietnam's independence but was dominated by the Communist Party. " \
       "Hồ Chí Minh led the Communist-ruled Democratic Republic of Vietnam in 1945, defeating the French Union in 1954 " \
       "at the Battle of Điện Biên Phủ, ending the First Indochina War, and resulting in the division of Vietnam, " \
       "with the Communists in control of North Vietnam. He was a key figure in the People's Army of Vietnam and the" \
       " Việt Cộng during the Vietnam War, which lasted from 1955 to 1975. North Vietnam was victorious against South" \
       " Vietnam and its allies, and Vietnam was officially unified in 1976. Saigon, the former capital of" \
       " South Vietnam, was renamed Ho Chi Minh City in his honor. Ho officially stepped down from power in 1965 " \
       "due to health problems and died in 1969."

    s_t = time.time()
    output = pipeline.run(text)
    e_t = time.time()
    print(output)
    print("time:", e_t-s_t)