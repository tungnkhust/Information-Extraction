from src.datasets.ConllDataset import Conll2003Dataset
from src.data_reader import CoNLLReader
from src.tagger.BertNER import BertNER
from transformers import AutoTokenizer

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("-train", "--train_path", default="data/CoNLL04/train.txt")
    parser.add_argument("-dev", "--dev_path", default="data/CoNLL04/dev.txt")
    parser.add_argument("-test", "--test_path", default="data/CoNLL04/dev.txt")
    parser.add_argument("-out", "--output_dir", default="./models/bert-ner")
    parser.add_argument("-bs", "--batch_size", type=int, default=2)
    parser.add_argument("-epoch", "--num_epochs", type=int, default=2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=int, default=1.0)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--fp16",  default=False, action="store_true")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    reader = CoNLLReader(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path
    )
    bio_tags = reader.get_bio_list()

    label2idx = {value: key for key, value in enumerate(bio_tags)}

    ner = BertNER(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        label2idx=label2idx
    )

    train_dataset = Conll2003Dataset(
        tokenizer=tokenizer,
        examples=reader.get_examples("train"),
        label2idx=label2idx
    )

    dev_dataset = Conll2003Dataset(
        tokenizer=tokenizer,
        examples=reader.get_examples("dev"),
        label2idx=label2idx
    )

    ner.train(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16
    )