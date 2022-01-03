from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.tagger.BertNER import BertNER

model_name_or_path = "models/bert-ner"
ner = BertNER(model_name_or_path=model_name_or_path)

text = "My name is Tung, I study at Ha No University of Science and Technology"
out = ner.run(text)
print(out)