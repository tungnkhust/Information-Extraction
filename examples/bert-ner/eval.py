import os.path

from src.tagger.BertNER import BertNER

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="./models/bert-ner")
    parser.add_argument("-test", "--test_path", default="data/CoNLL04/test.txt")
    parser.add_argument("-result", "--result_path", default="report/bert-ner")
    parser.add_argument("-soft", "--soft_eval", default=False, action="store_true")
    args = parser.parse_args()

    ner = BertNER(model_name_or_path=args.model_name_or_path)
    if args.soft_eval:
        args.result_path = args.result_path + "-soft"
    eval_result = ner.evaluate(args.test_path, result_dir=args.result_path, soft_eval=args.soft_eval)

    print(eval_result)