from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.tagger.BertNER import BertNER

model_name_or_path = "models/bert-ner"
ner = BertNER(model_name_or_path=model_name_or_path)

text = "In Indiana , downed tree limbs interrupted power in parts of Indianapolis ."
out = ner.run(text)
print(out)
