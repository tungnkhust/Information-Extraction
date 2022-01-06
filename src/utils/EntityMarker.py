from typing import Text, Dict, List, Union
from src.schema import Entity


class EntityMarker():
    def __init__(self, marker_mode: str = "entity"):
        self.marker_mode = marker_mode

    def mark(self, tokens: List, src_entity, trg_entity, mode=None):
        if isinstance(src_entity, Entity):
            src_entity = src_entity.to_dict()

        if isinstance(trg_entity, Entity):
            trg_entity = trg_entity.to_dict()

        if mode is None:
            mode = self.marker_mode

        if mode == "entity":
            return self.entity_mark(tokens, src_entity, trg_entity)
        else:
            return self.standard_mark(tokens, src_entity, trg_entity)

    def entity_mark(self, tokens: List, src_entity, trg_entity):
        tokens = tokens.copy()
        src_e = src_entity["entity"]
        src_start = src_entity["start_token"]
        src_end = src_entity["end_token"]

        trg_e = trg_entity["entity"]
        trg_start = trg_entity["start_token"]
        trg_end = trg_entity["end_token"]

        if src_start < trg_start:
            first_start = src_start
            first_end = src_end
            first_e = src_e
            second_start = trg_start
            second_end = trg_end
            second_e = trg_e
        else:
            first_start = trg_start
            first_end = trg_end
            first_e = trg_e
            second_start = src_start
            second_end = src_end
            second_e = src_e

        new_tokens = tokens[: first_start]

        new_tokens.append(f'[{first_e}]')
        new_tokens.extend(tokens[first_start: first_end])
        new_tokens.append(f'[/{first_e}]')

        new_tokens += tokens[first_end: second_start]

        new_tokens.append(f'[{second_e}]')
        new_tokens.extend(tokens[second_start: second_end])
        new_tokens.append(f'[/{second_e}]')

        new_tokens += tokens[second_end:]

        return new_tokens

    def standard_mark(self, tokens: List, src_entity, trg_entity):
        tokens = tokens.copy()
        src_e = src_entity["entity"]
        src_start = src_entity["start_token"]
        src_end = src_entity["end_token"]

        trg_e = trg_entity["entity"]
        trg_start = trg_entity["start_token"]
        trg_end = trg_entity["end_token"]

        if src_start < trg_start:
            first_start = src_start
            first_end = src_end
            first_e = src_e
            second_start = trg_start
            second_end = trg_end
            second_e = trg_e
        else:
            first_start = trg_start
            first_end = trg_end
            first_e = trg_e
            second_start = src_start
            second_end = src_end
            second_e = src_e

        new_tokens = tokens[: first_start]
        new_tokens.append(f'[{first_e}]')
        new_tokens += tokens[first_end: second_start]
        new_tokens.append(f'[{second_e}]')
        new_tokens += tokens[second_end:]

        return new_tokens
