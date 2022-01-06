from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.relation_extraction.BertRelCLF import BertRelCLF
from src.data_reader import CoNLLReader
model_name_or_path = "models/bert-rel"


reader = CoNLLReader(
    train_path="data/CoNLL04/train.txt",
    dev_path="data/CoNLL04/dev.txt",
    test_path="data/CoNLL04/test.txt"
)

label_list = reader.get_relation_list()
label2idx = {value: key for key, value in enumerate(label_list)}


model = BertRelCLF(
    model_name_or_path=model_name_or_path,
    label2idx=label2idx,
    max_seq_length=256
)

text = "In Indiana , downed tree limbs interrupted power in parts of Indianapolis ."
entities = [{'entity': 'Loc', 'start': 3, 'end': 10, 'value': 'Indiana', 'start_token': 1, 'end_token': 2},
            {'entity': 'Loc', 'start': 61, 'end': 73, 'value': 'Indianapolis', 'start_token': 11, 'end_token': 12}]

print(text)
out = model.run(text=text, entities=entities)
print(out)

