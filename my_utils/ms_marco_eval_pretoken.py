"""
This module computes evaluation metrics for MS MaRCo data set.

For first time execution, please use run.sh to download necessary dependencies.
Command line:
/ms_marco_metrics$ PYTHONPATH=./bleu python ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : Dec-15-2016
Last Modified : Fri 16 December 2016 07:00:00 PT
Authors : Tri Nguyen <trnguye@microsoft.com>, Xia Song <xiaso@microsoft.com>, Tong Wang <tongw@microsoft.com>
"""

from __future__ import print_function

import json
import sys

from .bleu import Bleu
from .rouge import Rouge
from spacy.lang.en import English as NlpEnglish

QUERY_ID_JSON_ID = 'query_id'
ANSWERS_JSON_ID = 'answers'
MAX_BLEU_ORDER = 4
NLP=None

def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=5):
    """Normalize and tokenize strings.

    Args:
    p_iter (iter): iter over strings to normalize and tokenize.
    p_batch_size (int): number of batches.
    p_thread_count (int): number of threads running.

    Returns:
    iter: iter over normalized and tokenized string.
    """

    global NLP
    if not NLP:
       NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, \
                          batch_size=p_batch_size, \
                          n_threads=p_thread_count)
    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield ' '.join(tokens)

def load_file(data_list):
    """Load data from json file.

    Args:
    p_path_to_data (str): path to file to load.
        File should be in format:
            {QUERY_ID_JSON_ID: <a_query_id_int>,
             ANSWERS_JSON_ID: [<list_of_answers_string>]}

    Returns:
    query_id_to_answers_map (dict):
        dictionary mapping from query_id (int) to answers (list of strings).
    no_answer_query_ids (set): set of query ids of no-answer queries.
    """

    all_answers = []
    query_ids = []
    no_answer_query_ids = set()
    for json_object in data_list:

        assert \
            QUERY_ID_JSON_ID in json_object, \
            '\"%s\" json does not have \"%s\" field' % \
                (line, QUERY_ID_JSON_ID)
        query_id = json_object[QUERY_ID_JSON_ID]

        assert \
            ANSWERS_JSON_ID in json_object, \
            '\"%s\" json does not have \"%s\" field' % \
                (line, ANSWERS_JSON_ID)
        answers = json_object[ANSWERS_JSON_ID]

        if not answers:
            answers = ['']
            no_answer_query_ids.add(query_id)

        all_answers.extend(answers)
        query_ids.extend([query_id]*len(answers))

    all_normalized_answers = normalize_batch(all_answers)

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        if query_id not in query_id_to_answers_map:
            query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)

    return query_id_to_answers_map, no_answer_query_ids

def compute_metrics_from_files(reference_data,
                               candidate_data,
                               p_max_bleu_order):
    """Compute BLEU-N and ROUGE-L metrics.
    IMPORTANT: No-answer reference will be excluded from calculation.

    Args:
    p_path_to_reference_file (str): path to reference file.
    p_path_to_candidate_file (str): path to candidate file.
        Both files should be in format:
            {QUERY_ID_JSON_ID: <a_query_id_int>,
             ANSWERS_JSON_ID: [<list_of_answers_string>]}
    p_max_bleu_order: the maximum n order in bleu_n calculation.

    Returns:
    dict: dictionary of {'bleu_n': <bleu_n score>, 'rouge_l': <rouge_l score>}
    """

    reference_dictionary, reference_no_answer_query_ids = \
        load_file(reference_data)
    candidate_dictionary, _ = load_file(candidate_data)

    filtered_reference_dictionary = \
        {key: value for key, value in reference_dictionary.items() \
                    if key not in reference_no_answer_query_ids}

    filtered_candidate_dictionary = \
        {key: value for key, value in candidate_dictionary.items() \
                    if key not in reference_no_answer_query_ids}

    for query_id, answers in filtered_candidate_dictionary.items():
        assert \
            len(answers) <= 1, \
            'query_id %d contains more than 1 answer \"%s\" in candidate file' % \
            (query_id, str(answers))

    reference_query_ids = set(filtered_reference_dictionary.keys())
    candidate_query_ids = set(filtered_candidate_dictionary.keys())
    common_query_ids = reference_query_ids.intersection(candidate_query_ids)
    assert (len(common_query_ids) == len(reference_query_ids)) and \
            (len(common_query_ids) == len(candidate_query_ids)), \
           'Reference and candidate files must share same query ids'

    all_scores = {}
    bleu_scores, _ = \
        Bleu(p_max_bleu_order).compute_score(filtered_reference_dictionary, \
                                             filtered_candidate_dictionary)
    for i, bleu_score in enumerate(bleu_scores):
        all_scores['bleu_%d' % (i+1)] = bleu_score

    rouge_score, _ = Rouge().compute_score(filtered_reference_dictionary, \
                                           filtered_candidate_dictionary)
    all_scores['rouge_l'] = rouge_score

    return all_scores

def main():
    """Command line: /ms_marco_metrics$ PYTHONPATH=./bleu python ms_marco_eval.py <path_to_reference_file> <path_to_candidate_file>"""

    path_to_referene_file = sys.argv[1]
    path_to_candidate_file = sys.argv[2]

    metrics = compute_metrics_from_files(path_to_referene_file, \
                                         path_to_candidate_file, \
                                         MAX_BLEU_ORDER)

    print('############################')
    for metric in sorted(metrics):
        print('%s: %s' % (metric, metrics[metric]))
    print('############################')

if __name__ == "__main__":
    main()
