from src.db_api.Neo4jDB import Neo4jDB
from src.schema.schema import Relation, Entity

from src.data_reader import CoNLLReader


db = Neo4jDB(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="160199"
)


def create_relation():
    s_e = Entity(entity="Per", value="Nguyen Van A")
    t_e = Entity(entity="Loc", value="Ha Noi")
    rel_type = "Live_In"
    r = Relation(source_entity=s_e, target_entity=t_e, relation=rel_type)
    db.create_relationship(r)
    results = db.query_relation_entities(s_e)
    for record in results:
        print(record)


def create_conll_database():
    train_path = "data/CoNLL04/train.txt"
    dev_path = "data/CoNLL04/dev.txt"
    test_path = "data/CoNLL04/test.txt"
    reader = CoNLLReader(train_path, dev_path, test_path)
    relations = reader.get_relations(split="test")
    for r in relations:
        db.create_relationship(r)


if __name__ == '__main__':
    create_relation()
    create_conll_database()