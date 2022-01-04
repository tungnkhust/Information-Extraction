import os.path

from src.relation_extraction.BertRelCLF import BertRelCLF

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="./models/bert-rel")
    parser.add_argument("-test", "--test_path", default="data/CoNLL04/test.txt")
    parser.add_argument("-result", "--result_path", default="report/bert-rel")
    args = parser.parse_args()

    rel = BertRelCLF(model_name_or_path=args.model_name_or_path)
    eval_result = rel.evaluate(args.test_path, has_direction=False)

    print(eval_result)