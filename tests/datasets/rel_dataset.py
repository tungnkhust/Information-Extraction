from src.datasets.RelDataset import RelDataset
from src.data_reader.CoNLLReader import CoNLLReader

from transformers import AutoTokenizer

reader = CoNLLReader(
    train_path="data/CoNLL04/train.txt",
    dev_path="data/CoNLL04/dev.txt",
    test_path="data/CoNLL04/test.txt"
)

label_list = reader.get_relation_list()
label2idx = {value: key for key, value in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = RelDataset(
    examples=reader.get_examples("test"),
    tokenizer=tokenizer,
    label2idx=label2idx,
    max_length=256
)

x = dataset[0]


# from collections import Counter
#
# count = Counter()
# for x in dataset:
#     count[label_list[x["labels"]]] += 1
#
# print(count)