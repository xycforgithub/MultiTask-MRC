""" Official evaluation script for v1.1 of the SQuAD dataset.
Credit from: https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/
"""
from __future__ import print_function
from collections import Counter
import os
import string
import re
import argparse
import json
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    res = max(scores_for_ground_truths)
    return res

def load_gold(datasets, data_path, dev_name='dev'):
    gold_data={}
    for dataset_name in datasets:
        if dataset_name !='marco': # marco needs other supervision
            gold_path = os.path.join(data_path, dataset_name,'%s.json' % dev_name)
            with open(gold_path) as dataset_file:
                gold_data[dataset_name] = json.load(dataset_file)
    return gold_data



def evaluate_file(dataset_json, predictions):
    """Used for validate"""
    # expected_version = '1.1'
    dataset = dataset_json['data']
    return evaluate(dataset, predictions)

# def evaluate_file(data_path, predictions):
#     """Used for validate"""
#     expected_version = '1.1'
#     with open(data_path) as dataset_file:
#         dataset_json = json.load(dataset_file)
#         # if (dataset_json['version'] != expected_version):
#         #     print('Evaluation expects v-' + expected_version +
#         #           ', but got dataset with v-' + dataset_json['version'],
#         #           file=sys.stderr)
#         dataset = dataset_json['data']
#         return evaluate(dataset, predictions)

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + \
                              ' will receive score 0.'
                    # print(message, file=sys.stderr)
                    continue
                total += 1
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

def eval_files(dataset_fn, pred_fn):
    with open(dataset_fn) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(pred_fn) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate(dataset, predictions)

if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))

