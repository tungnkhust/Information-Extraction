from src.data_reader.CoNLL2004Reader import CoNLL2004Reader


def test_conll_reader(split):
    train_path = "data/CoNLL04/conll04.txt"

    reader = CoNLL2004Reader(train_path)
    e_counter = reader.count_entities(split=split)
    r_counter, rdf_counter = reader.count_relations(split=split)
    print("Num sample:", reader.num_examples(split=split))
    print(e_counter)
    print(r_counter)
    print(rdf_counter)


if __name__ == '__main__':
    train_path = "data/CoNLL04/conll04.txt"
    split = "train"
    reader = CoNLL2004Reader(train_path)
    e_counter = reader.count_entities(split=split)
    r_counter, rdf_counter = reader.count_relations(split=split)
    print("Num sample:", reader.num_examples(split=split))
    print(e_counter)
    print(r_counter)
    print(rdf_counter)
    reader.save("data/CoNLL04/conll04_formatted_tacred.json", split=split)
