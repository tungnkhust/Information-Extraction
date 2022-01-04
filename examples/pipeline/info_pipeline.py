from src.pipelines.base import InformationPipeline
from src.tagger import BertNER
from src.relation_extraction import BertRelCLF


if __name__ == '__main__':
    ner = BertNER(model_name_or_path="models/bert-ner")
    rel = BertRelCLF(model_name_or_path="models/bert-rel")

    pipeline = InformationPipeline(
        coref_model=None,
        ner_model=ner,
        rel_model=rel
    )
    text = "My name is Tung , I study at Standford"
    output = pipeline.run(text)
    print(output)