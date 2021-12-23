from src.tagger import LstmNER

ner = LstmNER.from_pretrained("./models/ner")
output = ner.run("I Love You so much Anna Taylor")
print(output)