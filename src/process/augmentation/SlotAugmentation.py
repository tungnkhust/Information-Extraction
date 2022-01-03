from typing import Text, List, Dict
from src.schema.schema import InputExample
import random


class SlotAugmentation:
    def __init__(self, ontology: Dict):
        self.ontology = ontology

    def get_slot_value(self, key):
        values = self.ontology["entity"][key]
        n_values = len(values)
        index = random.randint(0, n_values-1)
        return values[index]

    


