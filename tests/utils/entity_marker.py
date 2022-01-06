from src.utils.EntityMarker import EntityMarker

marker = EntityMarker()

tokens = ['It', "'s", 'called', 'the', 'Sixth', 'Floor', ',', 'after', 'the', 'location', '--', 'the', 'sixth', 'floor', 'of', 'the', 'old', 'Texas', 'School', 'Book', 'Depository', ',', 'from', 'which', 'Lee', 'Harvey', 'Oswald', 'fired', 'the', 'shots', 'that', 'killed', 'Kennedy', '.']
relation = {'source_entity': {'entity': 'Peop', 'start_token': 24, 'end_token': 27, 'value': 'Lee Harvey Oswald'},
            'target_entity': {'entity': 'Peop', 'start_token': 32, 'end_token': 33, 'value': 'Kennedy'},
            'relation': 'Kill'}

_tokens = marker.entity_mark(tokens=tokens, src_entity=relation["source_entity"], trg_entity=relation["target_entity"])


_tokens = marker.standard_mark(tokens=tokens, src_entity=relation["source_entity"], trg_entity=relation["target_entity"])


tokens = ['Announces', 'Offer', 'to', 'End', 'Strike', 'PA0302025694', 'Panama', 'City', 'ACAN', 'in', 'Spanish', '2153', 'GMT', '2', 'Feb', '94']
src_e = {'entity': 'Other', 'value': '2153 GMT', 'start_token': 11, 'end_token': 13, 'start': 71, 'end': 80}
trg_e ={'entity': 'Other', 'value': '2 Feb 94', 'start_token': 13, 'end_token': 16, 'start': 80, 'end': 89}
print(marker.standard_mark(tokens=tokens, src_entity=src_e, trg_entity=trg_e))
