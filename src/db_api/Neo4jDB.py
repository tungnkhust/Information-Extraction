from neo4j import Neo4jDriver
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from src.schema import Relation, Entity
import logging
from typing import Dict, Union, List


class Neo4jDB:
    def __init__(self,
                 uri="neo4j://localhost:7687",
                 user="neo4j",
                 password="password"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_entity_node(self, entity: Union[Entity, Dict]):
        if isinstance(entity, Dict):
            entity = Entity.from_dict(entity)

        cypher = f"CREATE {entity.to_cypher('e')}"
        try:
            with self.driver.session() as graphDB_Session:
                graphDB_Session.run(query=cypher)

        except ServiceUnavailable as exception:
            logging.error("query raised an error: \n {exception}".format(exception=exception))

    def create_entities(self, entities):
        for e in entities:
            self.create_entity_node(e)

    def create_relations(self, relations):
        for r in relations:
            self.create_relationship(r)

    def create_relationship(self, relation: Union[Dict, Relation]):
        if isinstance(relation, Dict):
            relation = Relation.from_dict(relation)

        rel = f'[r:{relation.relation} ' + '{source: ' + str(relation.source) + '}]'
        cypher = (
            f"MATCH {relation.source_entity.to_cypher('e1')}"
            f"MATCH {relation.target_entity.to_cypher('e2')}"
            f"MERGE (e1)-{rel}->(e2)"
        )

        query_se = f"MERGE {relation.source_entity.to_cypher()}"
        query_te = f"MERGE {relation.target_entity.to_cypher()}"
        try:
            with self.driver.session() as graphDB_Session:
                graphDB_Session.run(query=query_se)
                graphDB_Session.run(query=query_te)
                graphDB_Session.run(query=cypher)

        except ServiceUnavailable as exception:
            logging.error("query raised an error: \n {exception}".format(exception=exception))

    def query_relation_entities(self, entity: Entity):
        query = (
            f"MATCH {entity.to_cypher()}-->(t_e)"
            "RETURN t_e"
        )
        with self.driver.session() as graphDB_Session:
            results = graphDB_Session.run(query, entity=entity.entity, value=entity.value)
            return results

