from src.datasets.RelDataset import RelDataset
from src.data_reader import CoNLLReader
from src.relation_extraction.BertRelCLF import BertRelCLF
from transformers import AutoTokenizer

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("-train", "--train_path", default="data/CoNLL04/train.txt")
    parser.add_argument("-dev", "--dev_path", default="data/CoNLL04/dev.txt")
    parser.add_argument("-test", "--test_path", default="data/CoNLL04/test.txt")
    parser.add_argument("-out", "--output_dir", default="./models/bert-rel")
    parser.add_argument("-len", "--max_seq_length", type=int, default=256)
    parser.add_argument("-bs", "--batch_size", type=int, default=2)
    parser.add_argument("-epoch", "--num_epochs", type=int, default=2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=int, default=1.0)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--entity_marker", type=str, default="entity", help="entity or standard")
    parser.add_argument("--fp16",  default=False, action="store_true")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    reader = CoNLLReader(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path
    )
    label_list = reader.get_relation_list()

    label2idx = {value: key for key, value in enumerate(label_list)}
    print(label2idx)

    ner = BertRelCLF(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        label2idx=label2idx,
        max_seq_length=args.max_seq_length,
        entity_marker=args.entity_marker
    )

    train_dataset = RelDataset(
        tokenizer=tokenizer,
        examples=reader.get_examples("train"),
        label2idx=label2idx,
        max_length=args.max_seq_length,
        marker_mode=args.entity_marker
    )

    dev_dataset = RelDataset(
        tokenizer=tokenizer,
        examples=reader.get_examples("dev"),
        label2idx=label2idx,
        max_length=args.max_seq_length,
        marker_mode=args.entity_marker
    )

    test_examples = reader.get_examples("test")
    if test_examples:
        test_dataset = RelDataset(
            tokenizer=tokenizer,
            examples=test_examples,
            label2idx=label2idx,
            max_length=args.max_seq_length,
            marker_mode=args.entity_marker
        )
    else:
        test_dataset = None

    ner.train(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        test_dataset=test_dataset,
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