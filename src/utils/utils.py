import json
import yaml


def batch(elements: list, batch_size: int, drop_last=False):
    n_batch = int(len(elements)/batch_size)
    for i in range(n_batch):
        yield elements[i*batch_size: (i+1)*batch_size]

    if drop_last is False:
        yield elements[n_batch*batch_size:]


def update_word_index_entity(entity, text):
    pre_start_token = text[:entity["start"]].strip(" ").split(" ")
    pre_start_token = [token for token in pre_start_token if token != '']
    pre_end_token = text[:entity["end"]].strip(" ").split(" ")
    pre_end_token = [token for token in pre_end_token if token != '']

    start_token = len(pre_start_token)
    end_token = len(pre_end_token)

    entity["start_token"] = start_token
    entity["end_token"] = end_token

    return entity


def update_char_index_entity(entity, tokens):
    text = " ".join(tokens)
    # get index start words
    char_ids = [0]
    temp = 0
    for i in range(1, len(tokens)):
        char_ids.append(temp + len(tokens[i - 1]) + 1)
        temp = char_ids[-1]
    char_ids.append(len(text) + 1)
    entity["start"] = char_ids[entity["start_token"]]
    entity["end"] = char_ids[entity["end_token"]]
    return entity


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

        start = info["start"]
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
    if tokens:
        char_ids = [0]
        temp = 0
        for i in range(1, len(tokens)):
            char_ids.append(temp + len(tokens[i - 1]) + 1)
            temp = char_ids[-1]
        char_ids.append(len(text) + 1)
    else:
        char_ids = None

    for i, tag in enumerate(tags):
        if tag[0] == 'B':
            entity = tag[2:]
            s = i
            e = i
            if i == len(tags) - 1:
                if tokens:
                    entities.append({"entity": entity,
                                     "start_token": s, "end_token": e+1,
                                     "start": char_ids[s], "end": char_ids[e+1],
                                     "value": " ".join(tokens[s: e+1])})
                else:
                    entities.append({"entity": entity, "start_token": s, "end_token": e + 1})

        elif tag[0] == 'I':
            e += 1
            if i == len(tags) - 1:
                if tokens:
                    entities.append({"entity": entity,
                                     "start_token": s, "end_token": e + 1,
                                     "start": char_ids[s], "end": char_ids[e + 1],
                                     "value": " ".join(tokens[s: e + 1])})
                else:
                    entities.append({"entity": entity, "start_token": s, "end_token": e + 1})

        elif tag == 'O':
            if entity is not None:
                if tokens:
                    entities.append({"entity": entity,
                                     "start_token": s, "end_token": e + 1,
                                     "start": char_ids[s], "end": char_ids[e + 1],
                                     "value": " ".join(tokens[s: e + 1])})
                else:
                    entities.append({"entity": entity, "start_token": s, "end_token": e + 1})
                entity = None

    return entities


def write_json(data, file_path, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as pf:
        json.dump(data, pf, ensure_ascii=False, indent=4)


def load_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as pf:
        data = json.load(pf)
        return data


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.Loader)
    return yaml_data


def write_yaml(data, file_path, **kwargs):
    if "encoding" in kwargs:
        encoding = kwargs['encoding']
    else:
        encoding = 'utf-8'
    with open(file_path, 'w', encoding=encoding) as pf:
        yaml.dump(data, pf, allow_unicode=True, default_flow_style=False, sort_keys=False)

