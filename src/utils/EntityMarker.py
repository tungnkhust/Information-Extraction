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

        mark_tokens = dict()
        mark_tokens[src_start] = f"[{src_e}]"
        mark_tokens[src_end] = f"[/{src_e}]"

        trg_e = trg_entity["entity"]
        if "start_token" in src_entity:
            trg_start = trg_entity["start_token"]
        else:
            trg_start = trg_entity["start"]

        if "end_token" in trg_entity:
            trg_end = trg_entity["end_token"]
        else:
            trg_end = trg_entity["end"]

        mark_tokens[trg_start] = f"[{trg_e}]"
        mark_tokens[trg_end] = f"[/{trg_e}]"
        mark_tokens = sorted(mark_tokens.items(), key=lambda x: x[0])
        for i, (index, token) in enumerate(mark_tokens):
            tokens.insert(index + i, token)

        return tokens

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
