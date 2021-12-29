from src.tagger import LstmNER

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", default="./models/lstm-ner")
    args = parser.parse_args()

    ner = LstmNER.from_pretrained(args.model_path)

    text = "I Love Anna Marry"

    output = ner.run(text=text)

    print(output)