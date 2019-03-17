import json
from .utils import repeat_save
import numpy as np

def load_rank_score(path):
    with open(path, 'r', encoding='utf-8') as reader:
        rank_scores = {}
        for line in reader:
            record = json.loads(line)
            rank_scores[record['query_id']] = eval(record['score'])
        return rank_scores



def generate_submit(ids_list, span_scores, predictions, rank_scores, yn_dict):
    span_score_dict = {}
    for uid, span_score, pred in zip(ids_list, span_scores, predictions):
        if uid in span_score_dict.keys():
            span_score_dict[uid].append({'answer': pred, 'score': span_score})
        else:
            span_score_dict[uid] = [{'answer': pred, 'score': span_score}]
    answer_list = []
    for key, val in span_score_dict.items():
        answer_list.append({'query_id': key, 'answers': '{}'.format(val)})
    assert len(span_score_dict) == len(rank_scores)

    rank_answer_list=[]
    for uid, value in span_score_dict.items():
        rank = rank_scores[uid]
        pred = [v['answer'] for v in value]
        span = [v['score'] for v in value]
        best = np.argmax([r * s for r, s in zip(rank, span)])
        final = {'query_id': uid, 'answers':[pred[best]]}
        rank_answer_list.append(final)

    yesno_answer_list=[]
    for uid, value in span_score_dict.items():
        rank = rank_scores[uid]
        pred = [v['answer'] for v in value]
        span = [v['score'] for v in value]
        best = np.argmax([r * s for r, s in zip(rank, span)])
        if yn_dict[str(uid)]:
            final = {'query_id': uid, 'answers':['yes']}
        else:
            final = {'query_id': uid, 'answers':[pred[best]]}
        yesno_answer_list.append(final)
    return answer_list, rank_answer_list, yesno_answer_list


def eval_model(model, data, output, span_output, output_yn, match_score, yn_dict):
    data.reset()
    dev_predictions = []
    dev_best_scores = []
    dev_ids_list = []
    for batch in data:
        phrase, phrase_score = model.predict(batch)
        dev_predictions.extend(phrase)
        dev_best_scores.extend(phrase_score)
        dev_ids_list.extend(batch['uids'])
    generate_submit(output, span_output, output_yn, dev_ids_list, dev_best_scores, dev_predictions, match_score, yn_dict)
