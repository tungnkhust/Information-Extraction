from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.tagger.BertNER import BertNER

model_name_or_path = "models/bert-ner"
ner = BertNER(model_name_or_path=model_name_or_path)

text = "Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. His mother is Maye Musk, a model and dietitian born in Saskatchewan, Canada, but raised in South Africa. His father is Errol Musk, a South African electromechanical engineer, pilot, sailor, consultant, and property developer who was once a half owner of a Zambian emerald mine near Lake Tanganyika."
out = ner.run(text)
for e in out["entities"]:
    print(e, "--------->", text[e["start"]: e["end"]], ":", e["entity"])
