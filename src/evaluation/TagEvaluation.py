from typing import List, Text, Union, Dict
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
from collections import Counter
import copy
import os
from src.evaluation.utils import plot_confusion_matrix, Column

ROOT_PATH = sys.path[1]


def get_entity_from_BIO(tags: List[str], tokens: List = None) -> List:
    """
    Get entities from BIO tags.
    :param tags: List of tagging for each tokens.
    :return: List entities that tagged in sentence.
    """
    if tags is None:
        return []
    s = 0
    e = 0
    entity = None
    entities = []
    for i, tag in enumerate(tags):
        if tag[0] == 'B':
            entity = tag[2:]
            s = i
            e = i
            if i == len(tags) - 1:
                entity_dict = {'entity': entity, 'start': s, 'end': e}
                if tokens:
                    entity_dict["value"] = " ".join(tokens[s:e+1])
                entities.append(entity_dict)
        elif tag[0] == 'I':
            e += 1
            if i == len(tags) - 1:
                entity_dict = {'entity': entity, 'start': s, 'end': e}
                if tokens:
                    entity_dict["value"] = " ".join(tokens[s:e+1])
                entities.append(entity_dict)
        elif tag == 'O':
            if entity is not None:
                entity_dict = {'entity': entity, 'start': s, 'end': e}
                if tokens:
                    entity_dict["value"] = " ".join(tokens[s:e+1])
                entities.append(entity_dict)
                entity = None

    return entities


def compare_entity(e_true: dict, e_pred: dict):
    """
    Compare 2 entities:
    Have 5 state of two entities:
        1 - Correct(cor): Both are the same.
        2 - Incorrect(inc): The predicted entity and the true entity don’t match
        3 - Partial(par): Both are the same entity but the boundaries of the surface string wrong
        4 - Missing(mis): The system doesn't predict entity
        5 - Spurius(spu): The system predict entity which doesn't exist in the true label.

    :param e_true: Entity in ground truth label. e
    :param e_pred: Entity in predicted label.
    :return:
    """
    s1 = int(e_true['start'])
    e1 = int(e_true['end'])
    s2 = int(e_pred['start'])
    e2 = int(e_pred['end'])
    if s1 == s2 and e1 == e2:
        if e_true['entity'] == e_pred['entity']:
            return 1
        else:
            return 2
    if ((s1 <= s2) and (s2 <= e1)) or ((s2 <= s1) and (s1 <= e2)):
        if e_true['entity'] == e_pred['entity']:
            return 3
        else:
            return 2
    if e1 < s2:
        return 4
    if e2 < s1:
        return 5


def compute_score(y_true: list, y_pred: list, tokens: list = None):
    """
    Get metric to evaluate for y_true and y_pred.
    :param y_true: List of tagging for each tokens of ground truth label .
    :param y_pred: List of tagging for each tokens of predicted label.
    :return: Dict include metric to evaluate for each entity
    and list of incorrect, missing and spurius entities.
    """
    entities_true = get_entity_from_BIO(y_true, tokens)
    entities_pred = get_entity_from_BIO(y_pred, tokens)
    metrics = {
        'support': len(entities_true),
        'cor': 0,
        'inc': 0,
        'par': 0,
        'mis': 0,
        'spu': 0,
    }
    correct = []
    incorrect = []
    missing = []
    spurius = []
    while len(entities_true) != 0 or len(entities_pred) != 0:
        if len(entities_true) == 0:
            metrics['spu'] += 1
            spurius.append(entities_pred[0])
            del entities_pred[0]
            continue

        if len(entities_pred) == 0:
            metrics['mis'] += 1
            spurius.append(entities_true[0])
            del entities_true[0]
            continue

        e1 = entities_true[0]
        e2 = entities_pred[0]

        state = compare_entity(e1, e2)
        if state == 1:
            metrics['cor'] += 1
            correct.append((e1, e2))
            del entities_true[0]
            del entities_pred[0]
        elif state == 2:
            metrics['inc'] += 1
            incorrect.append((e1, e2))
            del entities_true[0]
            del entities_pred[0]
        elif state == 3:
            metrics['par'] += 1
            del entities_true[0]
            del entities_pred[0]
        elif state == 4:
            metrics['mis'] += 1
            missing.append(e1)
            del entities_true[0]
        elif state == 5:
            metrics['spu'] += 1
            spurius.append(e2)
            del entities_pred[0]

    return {
        "metric": metrics,
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "spurius": spurius,
    }


def get_metrics(y_true: List[List], y_pred: List[List]):
    n_samples = len(y_true)
    metrics = {
        'support': 0,
        'cor': 0,
        'inc': 0,
        'par': 0,
        'mis': 0,
        'spu': 0
    }

    corrects = []
    incorrects = []
    missings = []
    spuriuses = []

    for i in range(n_samples):
        scores = compute_score(y_true[i], y_pred[i])
        metric = scores["metric"]
        correct = scores["correct"]
        incorrect = scores["incorrect"]
        missing = scores["missing"]
        spurius = scores["spurius"]

        metrics['cor'] += metric['cor']
        metrics['inc'] += metric['inc']
        metrics['par'] += metric['par']
        metrics['mis'] += metric['mis']
        metrics['spu'] += metric['spu']
        metrics['support'] += metric['support']
        corrects.append(correct)
        incorrects.append(incorrect)
        missings.append(missing)
        spuriuses.append(spurius)

    return {
        "metrics": metrics,
        "corrects": corrects,
        "incorrects": incorrects,
        "missings": missings,
        "spuriuses": spuriuses,
    }


def precision_score(y_true: List[List], y_pred: List[List], epsilon=1e-6, soft_eval=False):
    """
    Compute the precision score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: precision score.
    """
    scores = get_metrics(y_true, y_pred)
    metrics = scores["metrics"]

    n_predict = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
    if soft_eval is True:
        precision = (metrics['cor'] + metrics['par'] + epsilon) / (n_predict + epsilon)
    else:
        precision = (metrics['cor'] + epsilon) / (n_predict + epsilon)
    return precision


def recall_score(y_true: List[List], y_pred: List[List], epsilon=1e-6, soft_eval=False):
    """
    Compute the recall score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: recall score.
    """
    scores = get_metrics(y_true, y_pred)
    metrics = scores["metrics"]

    n_truth = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']
    if soft_eval is True:
        recall = (metrics['cor'] + metrics['par'] + epsilon)/(n_truth+epsilon)
    else:
        recall = (metrics['cor']+epsilon)/(n_truth+epsilon)
    return recall


def f1_score(y_true: List[List], y_pred: List[List], epsilon=1e-6, soft_eval=False):
    """
    Compute the f1-score.
    :param y_true: List of ground truth labels, Each label is list of tag for each tokens.
    :param y_pred: List of predicted labels, Each label is list of tag for each tokens.
    :param epsilon:
    :return: f1-score score.
    """
    scores = get_metrics(y_true, y_pred)
    metrics = scores["metrics"]

    n_predict = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
    n_truth = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']
    if soft_eval is True:
        precision = (metrics['cor'] + metrics['par'] + epsilon)/(n_predict+epsilon)
        recall = (metrics['cor'] + metrics['par'] + epsilon)/(n_truth+epsilon)
    else:
        precision = (metrics['cor']+epsilon)/(n_predict+epsilon)
        recall = (metrics['cor']+epsilon)/(n_truth+epsilon)
    f1_score = (2*precision*recall)/(precision + recall)
    return f1_score


class TagEvaluation:
    def __init__(
            self,
            epsilon=10e-13
    ):
        self.epsilon = epsilon
        self.cor = 0
        self.inc = 0
        self.mis = 0
        self.spu = 0
        self.par = 0
        self.support = 0

    def get_metrics(self, soft_eval=False):
        metrics = {}
        n_predict = self.cor + self.inc + self.par + self.spu
        n_truth = self.cor + self.inc + self.par + self.mis

        precision = (self.cor + self.epsilon) / (n_predict + self.epsilon)
        recall = (self.cor + self.epsilon) / (n_truth + self.epsilon)
        f1_score = (2 * precision * recall) / (precision + recall)

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1_score

        if soft_eval is True:
            precision = (self.cor + self.par + self.epsilon) / (n_predict + self.epsilon)
            recall = (self.cor + self.par + self.epsilon) / (n_truth + self.epsilon)
            f1_score = (2 * precision * recall) / (precision + recall)
            metrics["precision-soft"] = precision
            metrics["recall-soft"] = recall
            metrics["f1-soft"] = f1_score
        return metrics

    def run(
            self,
            true_tags: Union[List[Text], Text],
            pred_tags: Union[List[Text], Text],
            **kwargs
    ):
        if isinstance(true_tags, str):
            true_tags = [tag.split(' ') for tag in true_tags]

        if isinstance(pred_tags, str):
            pred_tags = [tag.split(' ') for tag in pred_tags]

        score = compute_score(true_tags, pred_tags)
        metric = score["metric"]
        self.cor += metric["cor"]
        self.inc += metric["inc"]
        self.par += metric["par"]
        self.mis += metric["mis"]
        self.spu += metric["spu"]
        self.support += metric["support"]
        kwargs["entities_score"] = metric

        return kwargs

    @staticmethod
    def analyse_miss(missings):
        mis = []
        for sample in missings:
            if sample:
                for e in sample:
                    mis.append(e['entity'])
        count_mis = Counter()
        for e in mis:
            count_mis[e] += 1
        return count_mis

    @staticmethod
    def analyse_spu(spuriuses):
        spu = []
        for sample in spuriuses:
            if sample:
                for e in sample:
                    spu.append(e['entity'])
        count_spu = Counter()
        for e in spu:
            count_spu[e] += 1
        return count_spu

    @staticmethod
    def analyse_cor(corrects):
        count_cor = Counter()
        for sample in corrects:
            check = []
            for e in sample:
                e = e[0]
                if e not in check:
                    count_cor[e["entity"]] += 1
                    check.append(sample)
        return count_cor

    @staticmethod
    def analyse_inc(incorrects):
        inc_true = []
        inc_pred = []

        for sample in incorrects:
            if sample:
                for inc in sample:
                    inc_true.append(inc[0]['entity'])
                    inc_pred.append(inc[1]['entity'])
        count_inc = Counter()

        # if entity be predicted fail then count 1
        for e in inc_true:
            count_inc[e] += 1

        return count_inc

    def plot_confusion_matrix(self, **kwargs):
        pass

    def evaluate(
            self,
            true_tags: List[List],
            pred_tags: List[List],
            epsilon=1e-5,
            soft_eval=False,
            result_dir="report/ner",
            **kwargs
    ):

        scores = get_metrics(true_tags, pred_tags)
        metrics = scores["metrics"]
        corrects = scores["corrects"]
        incorrects = scores["incorrects"]
        missings = scores["missings"]
        spuriuses = scores["spuriuses"]

        n_predict = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['spu']
        if soft_eval is True:
            precision = (metrics['cor'] + metrics['par'] + epsilon) / (n_predict + epsilon)
        else:
            precision = (metrics['cor'] + epsilon) / (n_predict + epsilon)
        n_truth = metrics['cor'] + metrics['inc'] + metrics['par'] + metrics['mis']

        if soft_eval is True:
            recall = (metrics['cor'] + metrics['par'] + epsilon) / (n_truth + epsilon)
        else:
            recall = (metrics['cor'] + epsilon) / (n_truth + epsilon)
        f1_score = (2 * precision * recall) / (precision + recall)

        count_cor = self.analyse_cor(corrects)
        count_inc = self.analyse_inc(incorrects)
        count_mis = self.analyse_miss(missings)
        count_spu = self.analyse_spu(spuriuses)

        report_entity = {}
        entity_report = list(count_cor.keys()) + list(count_inc.keys()) + list(count_mis.keys()) + list(count_spu.keys())
        entity_report = sorted(set(entity_report))

        for e in entity_report:
            report = {'cor': 0, 'inc': 0, 'mis': 0, 'spu': 0, 'support': 0}
            if e in count_cor:
                report['cor'] = count_cor[e]
            if e in count_inc:
                report['inc'] = count_inc[e]
            if e in count_mis:
                report['mis'] = count_mis[e]
            if e in count_spu:
                report['spu'] = count_spu[e]

            support = report['cor'] + report['inc'] + report['mis']
            report['support'] = support

            report_entity[e] = report

        results = dict()
        results["precision"] = round(precision, 4)
        results["recall"] = round(recall, 4)
        results["f1_score"] = round(f1_score, 4)
        results["entity_report"] = report_entity.copy()

        report_entity = sorted(report_entity.items(), key=lambda x: x[1]['inc'], reverse=True)
        entity_report = [report[0] for report in report_entity]
        cor_report = [report[1]['cor'] for report in report_entity]
        inc_report = [report[1]['inc'] for report in report_entity]
        mis_report = [report[1]['mis'] for report in report_entity]
        spu_report = [report[1]['spu'] for report in report_entity]
        support_report = [report[1]['support'] for report in report_entity]

        e_c = Column('entity', entity_report)
        c_c = Column('cor', cor_report)
        i_c = Column('inc', inc_report)
        m_c = Column('mis', mis_report)
        s_c = Column('spu', spu_report)
        sp_c = Column('support', support_report)

        if result_dir:
            if os.path.exists(result_dir) is False:
                os.makedirs(result_dir)

            metric_path = os.path.join(result_dir, "metric_results.txt")
            with open(metric_path, 'w') as pf:
                c1 = Column(key='precision', value=[round(precision, 4)])
                c2 = Column(key='recall', value=[round(recall, 4)])
                c3 = Column(key='f1_score', value=[round(f1_score, 4)])
                pf.write('### Precision-Recall-F1 Score\n')
                pf.write(c1.print_key() + '    ' + c2.print_key() + '    ' + c3.print_key())
                pf.write('\n')
                pf.write(c1.print_item(0) + '    ' + c2.print_item(0) + '    ' + c3.print_item(0))
                pf.write('\n')
                pf.write('\n')
                pf.write('### MUC Score\n')
                pf.write(e_c.print_key() + '    ' + c_c.print_key()
                         + '    ' + i_c.print_key() + '    ' + m_c.print_key()
                         + '    ' + s_c.print_key() + '    ' + sp_c.print_key())
                pf.write('\n')
                for i in range(len(entity_report)):
                    pf.write(e_c.print_item(i) + '    ' + c_c.print_item(i)
                             + '    ' + i_c.print_item(i) + '    ' + m_c.print_item(i)
                             + '    ' + s_c.print_item(i) + '    ' + sp_c.print_item(i))
                    pf.write('\n')

                pf.write('\n')
                pf.write('- Correct(cor): Both are the same\n')
                pf.write('- Incorrect(inc): The predicted entity and the true entity don’t match\n')
                pf.write('- Partial(par): Both are the same entity but the boundaries of the surface string wrong\n')
                pf.write("- Missing(mis): The system doesn't predict entity\n")
                pf.write("- Spurius(spu): The system predict entity which doesn't exist in the true label.\n")

            print(f"Save eval result in {os.path.abspath(result_dir)}")

        return results