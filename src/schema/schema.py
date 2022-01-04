import tokenize
from typing import Union, Text, List, Iterable, Any, Dict
import logging
from src.utils.utils import convert_entities_to_bio
from src.utils.utils import update_char_index_entity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Token:
    def __init__(
            self,
            index: int = None,
            text: Text = None,
            bio_tag: Text = None,
            rel_tags: List[Text] = None,
            rel_indies: List[int] = None,
            pos_tag: Text = None,
            chunking_tag: Text = None,
            start: int = None,
            end: int = None,
            entity_index: int = None

    ):
        """
        :param index: index of token in sequence
        :param text: value of token
        :param bio_tag: bio tag of token
        :param rel_tags: relation tag of token
        :param rel_indies: if token has rel_tag is not `N` then token is the head entity and rel_index is tail entity
                        in a relation. if token relate with many other entities then have many rel_tags and rel_indies
        :param pos_tag: pos tag of token
        :param chunking_tag: chunking tag of token
        """
        self.index = index
        self.text = text
        self.bio_tag = bio_tag
        self.rel_tags = rel_tags
        self.rel_indies = rel_indies
        self.pos_tag = pos_tag
        self.chunking_tag = chunking_tag
        self.start = start
        self.end = end
        self.entity_index = entity_index

    def to_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "bio_tag": self.bio_tag,
            "rel_tags": self.rel_tags,
            "rel_indies": self.rel_indies,
            "pos_tag": self.pos_tag,
            "chunking_tag": self.chunking_tag
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())


class Entity:
    def __init__(
            self,
            entity: Text = None,
            value: Text = None,
            start: int = None,
            end: int = None,
            start_token: int = None,
            end_token: int = None
    ):
        self.entity = entity
        self.value = value
        self.start = start
        self.end = end
        self.start_token = start_token
        self.end_token = end_token

    def to_dict(self):
        entity = {
            "entity": self.entity,
            "value": self.value,
            "start_token": self.start_token,
            "end_token": self.end_token
        }
        if self.start:
            entity = {
                "entity": self.entity,
                "value": self.value,
                "start_token": self.start_token,
                "end_token": self.end_token,
                "start": self.start,
                "end": self.end
            }

        return entity

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def to_cypher(self, name=''):
        return f'({name}:{self.entity} ' + '{value: "' + self.value + '"})'


class Relation:
    rel_map = {
        "Live_In": "LIVE_IN",
        "Located_In": "LOCATED_IN",
        "OrgBased_IN": "OGR_BASED_IN",
        "Kill": "KILL"
    }

    def __init__(
            self,
            source_entity: Entity = None,
            target_entity: Entity = None,
            relation: Text = None,
            source: List[Text] = None,
            meta: Dict = None
    ):
        self.source_entity = source_entity
        self.target_entity = target_entity
        self.relation = relation
        self.source = source if source else []
        self.meta = meta

    def __str__(self):
        source = f'({self.source_entity.entity}:{self.source_entity.value})'
        target = f'({self.target_entity.entity}:{self.target_entity.value})'
        relation = f'[:{self.relation} ' + '{source: "' + str(self.source) + '"}]'
        return f'({source})-[{relation}]->({target})'

    def __repr__(self):
        return str(self)

    def to_cypher(self, e1_name='', e2_name=''):
        source = self.source_entity.to_cypher(e1_name)
        target = self.target_entity.to_cypher(e2_name)
        relation = self.relation
        return f'{source}-[:{relation}]->{target}'

    def get_pair_entity(self):
        return f'{self.source_entity.entity}-{self.target_entity.entity}#{self.relation}'

    @classmethod
    def from_dict(cls, data: Dict):
        source_entity = Entity.from_dict(data["source_entity"])
        target_entity = Entity.from_dict(data["target_entity"])
        relation_label = data["relation"]
        return cls(source_entity=source_entity, target_entity=target_entity, relation=relation_label)

    def to_dict(self):
        return {
            "source_entity": self.source_entity.to_dict(),
            "target_entity": self.target_entity.to_dict(),
            "relation": self.relation
        }


class InputExample:
    def __init__(
            self,
            id: Text = None,
            tokens: List[Token] = None,
            rel_in: Text = "end",
            entities: List[Entity] = [],
            relations: List[Relation] = []
    ):
        self.id = id
        self.rel_in = rel_in
        self.tokens = tokens if tokens else []

        if entities:
            self.entities = entities
        else:
            self.entities = self._get_entities()
        self._update_char_index_entity()

        if relations:
            self.relations = relations
        else:
            self.relations = self._get_relations()

    def _update_char_index_entity(self):
        entities = []
        for e in self.entities:
            entity = update_char_index_entity(e.to_dict(), self.get_tokens())
            entities.append(Entity.from_dict(entity))
        self.entities = entities

    @classmethod
    def from_dict(cls, data):
        """
        format:
        {
            "id": "",
            "text": "",
            "tokens": [],
            "entities": [],
            "relations": []
        }
        :return:
        """
        token_texts = data["tokens"]
        text = " ".join(token_texts)
        bio = convert_entities_to_bio(entities=data["entities"], text=text)
        entities = [Entity.from_dict(e) for e in data["entities"]]

        tokens = []
        for i, word in enumerate(token_texts):
            tokens.append(Token(text=word, bio_tag=bio[i]))

        relations = [Relation.from_dict(r) for r in data["relations"]]

        return cls(
            id=data["id"],
            tokens=tokens,
            entities=entities,
            relations=relations
        )

    def to_tacred(self):
        tac_examples = []
        if self.relations:
            for rel in self.relations:
                tac_examples.append({
                    "id": self.id,
                    "relation": rel.relation,
                    "token": self.get_tokens(),
                    "subj_start": rel.source_entity.start_token,
                    "subj_end": rel.source_entity.end_token,
                    "obj_start": rel.target_entity.start_token,
                    "obj_end": rel.target_entity.end_token,
                    "subj_type": rel.source_entity.entity,
                    "obj_type": rel.target_entity.entity,
                    "stanford_pos": self.get_pos_tags(),
                    "stanford_ner": [tag.strip("BI-") for tag in self.get_pos_tags()],
                    "stanford_head": [],
                    "stanford_deprel": []
                })
        else:
            tac_examples.append({
                "id": self.id,
                "relation": "no_relation",
                "token": self.get_tokens(),
                "subj_start": 0,
                "subj_end": 0,
                "obj_start": 0,
                "obj_end": 0,
                "subj_type": "no_entity",
                "obj_type": "no_entity",
                "stanford_pos": self.get_pos_tags(),
                "stanford_ner": [tag.strip("BI-") for tag in self.get_bio_tags()],
                "stanford_head": [],
                "stanford_deprel": []
            })
        return tac_examples

    def get_entity(self, start_token=None, end_token=None, value=None):
        _entities = []
        for e in self.entities.copy():
            if e.start_token == start_token:
                _entities.append(e)
            elif e.end_token == end_token:
                _entities.append(e)
            elif e.value == value:
                _entities.append(e)
        return _entities

    def get_length(self):
        return len(self.tokens)

    def num_entities(self):
        return len(self.entities)

    def num_relations(self):
        return len(self.relations)

    def get_entities(self):
        return [e.to_dict() for e in self.entities]

    def _get_entities(self):
        entities = []
        s = 0
        e = 0
        entity = None

        for i, token in enumerate(self.tokens):
            if token.bio_tag[0] == "B":
                if i > 0 and self.tokens[i-1].bio_tag[0] != "O":
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    ent = Entity(entity=entity, value=value, start_token=s, end_token=e + 1)
                    entities.append(ent)

                s = i
                e = i
                entity = token.bio_tag[2:]

                if i == len(self.tokens) - 1:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    ent = Entity(entity=entity, value=value, start_token=s, end_token=e + 1)
                    entities.append(ent)

            elif token.bio_tag[0] == "I":
                e += 1
                if i == len(self.tokens) - 1:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    ent = Entity(entity=entity, value=value, start_token=s, end_token=e + 1)
                    entities.append(ent)
            elif token.bio_tag[0] == "O":
                if entity is not None:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    ent = Entity(entity=entity, value=value, start_token=s, end_token=e + 1)
                    entities.append(ent)
                    entity = None

        return entities

    def find_entity(self, entity: Text = None, start_token: int = None, end_token: int = None):
        entities = self.entities if self.entities else self.get_entities()
        for e in entities:
            if e.entity == entity or e.start_token == start_token or e.end_token == end_token:
                return e
        return None

    def get_relations(self):
        return [rel.to_dict() for rel in self.relations]

    def _get_relations(self):

        relations = []

        for token in self.tokens:
            for i, rel_tag in enumerate(token.rel_tags):
                if rel_tag != 'N':
                    if self.rel_in == "end":
                        source_entity = self.find_entity(end_token=token.index+1)
                        target_entity = self.find_entity(end_token=token.rel_indies[i]+1)
                    elif self.rel_in == "begin":
                        source_entity = self.find_entity(start_token=token.index)
                        target_entity = self.find_entity(start_token=token.rel_indies[i])
                    else:
                        raise KeyError("rel_in mus be `begin`: begin of entity or `end`: end of entity")
                    if source_entity and target_entity:
                        rel = Relation(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relation=rel_tag,
                            source=[self.get_text()]
                        )
                        relations.append(rel)
        return relations

    def get_tokens(self):
        return [token.text for token in self.tokens]

    def get_text(self):
        tokens = [token.text for token in self.tokens]
        return " ".join(tokens)

    def get_bio_tags(self):
        bio_tags = [token.bio_tag for token in self.tokens]
        return bio_tags

    def get_pos_tags(self):
        pos_tags = [token.pos_tag for token in self.tokens]
        return pos_tags

    def get_chunking_tags(self):
        chunking_tags = [token.chunking_tag for token in self.tokens]
        return chunking_tags

    def __str__(self):
        text = self.get_text()
        bio = self.get_bio_tags()
        entities = self.entities
        relations = self.relations
        return f'id: {self.id}\ntext: {text}\nbio: {bio}\nentities: {entities}\nrelations: {relations}'
