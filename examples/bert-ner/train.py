from src.datasets.ConllDataset import Conll2003Dataset
from src.data_reader import CoNLLReader
from src.tagger.BertNER import BertNER
from transformers import AutoTokenizer


model_name_or_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
reader = CoNLLReader(
    train_path="data/CoNLL04/train.txt",
    dev_path="data/CoNLL04/dev.txt",
    test_path="data/CoNLL04/dev.txt"
)
examples = reader.get_examples("dev")
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
    num_epochs=2,
    batch_size=2,
    output_dir="models/bert-ner"
)