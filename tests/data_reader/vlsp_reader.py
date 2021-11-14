from src.data_reader import VLSPReader


def test_vlsp_reader(split):
    train_dir = "data/VLSP2020"
    reader = VLSPReader(train_dir=train_dir)
    e_counter = reader.count_entities(split=split)
    r_counter, rdf_counter = reader.count_relations(split=split)
    print("Num sample:", reader.num_examples(split=split))
    print(e_counter)
    print(r_counter)
    print(rdf_counter)


if __name__ == '__main__':
    test_vlsp_reader(split="train")
    print('-'*80)
