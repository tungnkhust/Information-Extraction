from typing import List, Text, Union, Dict
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
from collections import Counter
import copy
import os
from src.evaluation.utils import plot_confusion_matrix, Column
from src.evaluation.TagEvaluation import compare_entity
ROOT_PATH = sys.path[1]


def compare_relation(rel1, rel2, has_direction=False):
    src_e_1 = rel1["source_entity"]
    trg_e_1 = rel1["target_entity"]
    label1 = rel1["relation"]

    src_e_2 = rel2["source_entity"]
    trg_e_2 = rel2["target_entity"]
    label2 = rel2["relation"]
    if has_direction:
        src_status = compare_entity(src_e_1, src_e_2)
        trg_status = compare_entity(trg_e_1, trg_e_2)
        if (src_status in [1, 3]) and (trg_status in [1, 3]) and (label1 == label2):
            return True

        return False
    else:
        src_status = (compare_entity(src_e_1, src_e_2) in [1, 3]) or (compare_entity(src_e_1, trg_e_2) in [1, 3])
        trg_status = (compare_entity(trg_e_1, src_e_2) in [1, 3]) or (compare_entity(trg_e_1, trg_e_2) in [1, 3])
        if src_status and trg_status and (label1 == label2):
            return True
        return False


def check_rel_in(rel, list_rels, has_direction=False):
    for _rel in list_rels:
        if compare_relation(rel, _rel, has_direction):
            return True
    return False


def compute_score(true_rels, pred_rels, has_direction=False):
    metrics = {
        'n_correct': 0,
        'n_incorrect': 0,
        'n_truth': len(true_rels),
        'n_predict': len(pred_rels)
    }
    for pre_rel in pred_rels:
        if check_rel_in(pre_rel, true_rels, has_direction):
            metrics["n_correct"] += 1
        else:
            metrics["n_incorrect"] += 1

    return metrics


class RelEvaluation:
    def __init__(
            self,
            has_direction=False,
            epsilon=10e-13
    ):
        self.has_direction = has_direction
        self.epsilon = epsilon
        self.n_correct = 0
        self.n_incorrect = 0
        self.n_predict = 0
        self.n_truth = 0

    def get_metrics(self):
        precision = (self.n_correct + self.epsilon) / (self.n_predict + self.epsilon)
        recall = (self.n_correct + self.epsilon) / (self.n_truth + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def run(self, relations, relations_label, **kwargs):
        score = compute_score(relations_label, relations, has_direction=self.has_direction)
        self.n_correct += score["n_correct"]
        self.n_incorrect += score["n_incorrect"]
        self.n_truth += score["n_truth"]
        self.n_predict += score["n_predict"]
        kwargs["relations"] = relations
        return kwargs

    def evaluate(
            self,
            true_relations: List = [],
            pred_relations: List = [],
            has_direction=None
    ):
        assert len(true_relations) == len(pred_relations)
        if has_direction is None:
            has_direction = self.has_direction

        n_samples = len(true_relations)
        n_correct = 0
        n_incorrect = 0
        n_truth = 0
        n_predict = 0
        for i in range(n_samples):
            true_rels = true_relations[i]
            pred_rels = pred_relations[i]
            score = compute_score(true_rels, pred_rels, has_direction=has_direction)
            n_correct += score["n_correct"]
            n_incorrect += score["n_incorrect"]
            n_truth += score["n_truth"]
            n_predict += score["n_predict"]

        precision = (n_correct + self.epsilon)/(n_predict + self.epsilon)
        recall = (n_correct + self.epsilon)/(n_truth + self.epsilon)
        f1 = 2*precision*recall/(precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }



