from typing import Union, Text, List, Iterable
import logging
import ast

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

    @classmethod
    def from_text(cls, text):
        tmp = text.split('\t')
        index = int(tmp[0])
        text = tmp[1]
        bio_tag = tmp[2]
        rel_tags = ast.literal_eval(tmp[3])
        rel_indies = ast.literal_eval(tmp[4])
        return cls(
            index=index,
            text=text,
            bio_tag=bio_tag,
            rel_tags=rel_tags,
            rel_indies=rel_indies
        )


class InputExample:
    entities = []
    relations = []

    def __init__(
            self,
            id: Text = None,
            tokens: List[Token] = None,
    ):
        self.id = id
        self.tokens = tokens if tokens else []
        self.entities = self.get_entities()
        self.relations = self.get_relations()

    def num_entities(self):
        return len(self.entities)

    def num_relations(self):
        return len(self.relations)

    def get_entities(self):
        entities = []
        s = 0
        e = 0
        entity = None

        for i, token in enumerate(self.tokens):
            if token.bio_tag[0] == "B":
                if i > 0 and self.tokens[i-1].bio_tag[0] != "O":
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    entities.append({"entity": entity, "start_token": s, "end_token": e, "value": value})

                s = i
                e = i
                entity = token.bio_tag[2:]

                if i == len(self.tokens) - 1:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    entities.append({"entity": entity, "start_token": s, "end_token": e, "value": value})

            elif token.bio_tag[0] == "I":
                e += 1
                if i == len(self.tokens) - 1:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    entities.append({"entity": entity, "start_token": s, "end_token": e, "value": value})
            elif token.bio_tag[0] == "O":
                if entity is not None:
                    value = " ".join([t.text for t in self.tokens][s:e + 1])
                    entities.append({"entity": entity, "start_token": s, "end_token": e, "value": value})
                    entity = None

        return entities

    def find_entity(self, entity: Text = None, start_token: int = None, end_token: int = None):
        entities = self.entities if self.entities else self.get_entities()
        for e in entities:
            if e['entity'] == entity or e['start_token'] == start_token or e['end_token'] == end_token:
                return e
        return None

    def get_relations(self):
        relations = []

        for token in self.tokens:
            for i, rel_tag in enumerate(token.rel_tags):
                if rel_tag != 'N':
                    head_entity = self.find_entity(end_token=token.index)
                    tail_entity = self.find_entity(end_token=token.rel_indies[i])
                    if head_entity and tail_entity:
                        relation = rel_tag
                        relations.append({
                            "head_entity": head_entity,
                            "tail_entity": tail_entity,
                            "relation": relation
                        })
        return relations

    def get_text(self):
        tokens = [token.text for token in self.tokens]
        return " ".join(tokens)

    def get_bio_tags(self):
        bio_tags = [token.bio_tag for token in self.tokens]
        return " ".join(bio_tags)

    def get_pos_tags(self):
        pos_tags = [token.pos_tag for token in self.tokens]
        return " ".join(pos_tags)

    def get_chunking_tags(self):
        chunking_tags = [token.chunking_tag for token in self.tokens]
        return " ".join(chunking_tags)

    def __str__(self):
        text = self.get_text()
        bio = self.get_bio_tags()
        entities = self.entities
        relations = self.relations
        return f'id: {self.id}\ntext: {text}\nbio: {bio}\nentities: {entities}\nrelations: {relations}'
