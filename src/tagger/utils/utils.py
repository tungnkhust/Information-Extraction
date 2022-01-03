from typing import Dict, Text


def convert_entities_to_bio(entities, text):
    list_text_label = []
    tokens = text.split(" ")

    for i in range(len(tokens)):
        list_text_label.append('O')

    if entities is None:
        return ' '.join(list_text_label)

    for info in entities:
        if info["entity"] == 'attribute' or info["entity"] == 'object_type':
            label = '{}:{}'.format(info["entity"], info["value"])
        else:
            label = info["entity"]

        start = info['start']
        end = info['end']

        value = text[start:end]
        list_value = value.split(" ")

        index = len(text[:start].split(" ")) - 1
        list_text_label[index] = 'B-' + str(label)
        for j in range(1, len(list_value)):
            try:
                list_text_label[index + j] = 'I-' + str(label)
            except Exception as e:
                print(str(e))
                print(text)
                print(entities)
    return list_text_label


def convert_bio_to_entities(text, bio):
    if isinstance(text, str):
        tokens = text.split(' ')
    else:
        tokens = text

    if isinstance(bio, str):
        tags = bio.split(' ')
    else:
        tags = bio

    s = 0
    e = 0

    entity = None
    entities = []

    # get index start words
    char_ids = [0]
    temp = 0
    for i in range(1, len(tokens)):
        char_ids.append(temp + len(tokens[i - 1]) + 1)
        temp = char_ids[-1]
    char_ids.append(len(text) + 1)

    for i, tag in enumerate(tags):
        if tag[0] == 'B':
            entity = tag[2:]
            s = i
            e = i
            if i == len(tags) - 1:
                entities.append({"entity": entity,
                                 "start_token": s, "end_token": e,
                                 "start": char_ids[s], "end": char_ids[e+1],
                                 "value": " ".join(tokens[s: e+1])})
        elif tag[0] == 'I':
            e += 1
            if i == len(tags) - 1:
                entities.append({"entity": entity,
                                 "start_token": s, "end_token": e,
                                 "start": char_ids[s], "end": char_ids[e + 1],
                                 "value": " ".join(tokens[s: e + 1])})
        elif tag == 'O':
            if entity is not None:
                entities.append({"entity": entity,
                                 "start_token": s, "end_token": e,
                                 "start": char_ids[s], "end": char_ids[e + 1],
                                 "value": " ".join(tokens[s: e + 1])})
                entity = None

    return entities
