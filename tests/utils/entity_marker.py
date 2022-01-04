from src.utils.EntityMarker import EntityMarker

marker = EntityMarker()

tokens = ['It', "'s", 'called', 'the', 'Sixth', 'Floor', ',', 'after', 'the', 'location', '--', 'the', 'sixth', 'floor', 'of', 'the', 'old', 'Texas', 'School', 'Book', 'Depository', ',', 'from', 'which', 'Lee', 'Harvey', 'Oswald', 'fired', 'the', 'shots', 'that', 'killed', 'Kennedy', '.']
relation = {'source_entity': {'entity': 'Peop', 'start_token': 24, 'end_token': 27, 'value': 'Lee Harvey Oswald'},
            'target_entity': {'entity': 'Peop', 'start_token': 32, 'end_token': 33, 'value': 'Kennedy'},
            'relation': 'Kill'}

_tokens = marker.entity_mark(tokens=tokens, src_entity=relation["source_entity"], trg_entity=relation["target_entity"])
print(_tokens)


_tokens = marker.standard_mark(tokens=tokens, src_entity=relation["source_entity"], trg_entity=relation["target_entity"])
print(_tokens)