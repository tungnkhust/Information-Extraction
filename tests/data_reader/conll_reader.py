from src.data_reader import CoNLLReader


def test_conll_reader(split):
    train_path = "data/CoNLL04/train.txt"
    dev_path = "data/CoNLL04/dev.txt"
    test_path = "data/CoNLL04/test.txt"

    reader = CoNLLReader(train_path, dev_path, test_path)
    e_counter = reader.count_entities(split=split)
    r_counter, rdf_counter = reader.count_relations(split=split)
    print("Num sample:", reader.num_examples(split=split))
    print(e_counter)
    print(r_counter)
    print(rdf_counter)


if __name__ == '__main__':
    test_conll_reader(split="train")
    print('-'*80)
    test_conll_reader(split="dev")
    print('-' * 80)
    test_conll_reader(split="test")
    print('-' * 80)