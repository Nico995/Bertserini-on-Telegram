""" Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was
modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0
In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""


import collections
import json
import logging
import math
import re
import string
import torch
from typing import DefaultDict, Dict, List, Mapping, Tuple, Union
import numpy as np

from transformers import BertTokenizer
from transformers.data.metrics.squad_metrics import squad_evaluate, get_final_text
from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadResult
from transformers.data.metrics.squad_metrics import normalize_answer, compute_exact

# from transformers.tokenization_bert import BasicTokenizer

logger = logging.getLogger(__name__)


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    """Converts a cuda tensor to a list (used to convert logits in a more favorable format).

    Args:
        tensor (torch.tensor): A gpu tensor to be converted.
    
    Returns:
        List[float]: The converted tensor.
        
    """
    return tensor.detach().cpu().tolist()[0]


def get_best_indices(logits: torch.Tensor, n_best: int) -> List[float]:
    """Return the top n_best logits indices.

    Args:
        logits (torch.tensor): All the logits.
        n_best ([type]): Number of top values to be returned.

    Returns:
        List[float]: A list containing the indices of the top n_best logits.
        
    """
    logits = sorted(enumerate(logits), key=lambda x: -x[1])
    return [index for index, _ in logits[:n_best]]



def compute_predictions(
    all_examples: List[SquadExample],
    all_features: List[SquadFeatures],
    all_results: List[SquadResult],
    n_best: int,
    max_answer_length: int,
    do_lower_case: bool,
    null_score_diff_threshold: float,
    tokenizer: BertTokenizer,
    language: str = "en",
    fancy_answer: bool = False
) -> Mapping[str, Tuple[str, float]]:
    
    """Compute answers from predicted logits.

    Args:
        all_examples (List[SquadExample]): The list of all SquadExample. Extracted by craft_squad_examples.
        all_features (List[SquadFeatures]): The list of all SquadFeatures. Extracted by squad_convert_examples_to_features.
        all_results (List[SquadResult]): The list of SquadResult. Extracted by the NLP model and wrapped by a SquadResult object.
        n_best (int): Number of top results to consider.
        max_answer_length (int): Maximum length of any answer.
        do_lower_case (bool): Whether to lowercase input when tokenizing.
        null_score_diff_threshold (float): The threshold used to determine whether an answer should be Null or not.
        tokenizer (BertTokenizer): The tokenizer to use when converting the answer back to Natural Language.
        language (str, optional): The language to use for the answers. Defaults to "en".
        fancy_answer (bool, optional): Whether to use get_final_text to obtain a clean answer (takes a lot of time). Defaults to False.

    Returns:
        Mapping[str, Tuple[str, float]] [description]: A mapping between the unique id of the question and a tuple containing the text and the ans_score
        
    """

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    # build dict to index result through result id
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # build a namedtuple to store prediction data: features, start/end indices and logits
    PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    # build variables to store predictions
    all_predictions = DefaultDict(lambda: [])

    # loop over all the examples (which are SquadExamples object directly from get_dev_samples)
    # all_examples is essentially all the data used for train/inference
    for (example_index, example) in enumerate(all_examples):

        # get features for the current example
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        null_score = 1000000
        null_prediction = None

        # loop over the features for the given example, in some edge cases, the features are split
        # in that case, just repeat the process for the two splits, the loop is designed so that
        # the best result among the different splits will be retained
        # for ease of the mind, you can reason like this loop doesnt exist :)
        for (feature_index, feature) in enumerate(features):

            result = unique_id_to_result[feature.unique_id]

            # get the logits and change it into a more favorable format
            start_logits = tensor_to_list(result.start_logits)
            end_logits = tensor_to_list(result.end_logits)

            # this indices refer to the position of the logits with respect to the example span
            # example span = "[SEP]<question>[CLS]<context>[SEP]"
            start_indices = get_best_indices(start_logits, n_best)
            end_indices = get_best_indices(end_logits, n_best)

            # Get the score of the null answer (a null answer starts and ends on the [CLS] token)
            # the cls token is the first token of the answer
            feature_null_score = start_logits[0] + end_logits[0]

            # check shape of logits

            # save the lowest null score of the current example
            if feature_null_score < null_score:
                null_score = feature_null_score
                null_prediction = PrelimPrediction(feature_index, 0, 0, start_logits[0], end_logits[0])

            # we can now try to find an answer by trying all the different combination of the top k logits
            # this is the robust alternative to the naive implementation with argmax(logit)
            for start_index in start_indices:
                for end_index in end_indices:
                    # we need to prune our choices, discard invalid answers according to
                    # the following criteria

                    # answer is nonsensical -> discard
                    # we run this condition first so that we now can assume the relative position of starat - end
                    if end_index < start_index:
                        continue
                    # if the start token index is greater than the token list length -> discard
                    if start_index >= len(feature.tokens):
                        continue
                    # if stard token index is not in the token_to_orig map -> discard
                    if start_index not in feature.token_to_orig_map:
                        continue
                    # if stard token index is not in the token_to_orig map -> discard
                    if end_index not in feature.token_to_orig_map:
                        continue
                    # the token_to_orig_map is a structure that maps tokens to the original word in the sentence (only spans the context)
                    # i.e. orig="the normans (norman: nourmands; french: normands [...]"
                    # i.e.            tokens="the, norman, ##s, (, norman, :, no, ##ur, ##man, ##ds, ;, french, :, norman, ##ds"
                    # i.e. tokens_to_orig_map="0,    1,    1,  2,   2,    2,  3,   3,    3,     3,  3,    4,   4,   5,      5"

                    # no idea what token_is_max_context is used for
                    if not feature.token_is_max_context.get(start_index, False):
                        continue

                    # answer is too long -> discard
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    # store prediction
                    prelim_predictions.append(
                        PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logits[start_index],
                            end_logit=end_logits[end_index],
                        )
                    )

        # add the null prediction to the list of predictions
        prelim_predictions.append(null_prediction)

        # sort predictions according to the total score
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        BestPrediction = collections.namedtuple(
            "BestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        best_predictions = []

        # looping in descending score order over the preliminary predictions found for a specific feature
        # prune the preliminary predictions to a top n_best list and convert to text
        for pred in prelim_predictions:
            if len(best_predictions) >= n_best:
                break

            feature = features[pred.feature_index]

            # retain only non-null predictions
            if pred.start_index > 0:
                
                norm_tokens = feature.tokens[pred.start_index:pred.end_index+1]
                norm_string_tokens = tokenizer.convert_tokens_to_string(norm_tokens)

                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_string_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                norm_text = " ".join(norm_string_tokens.split())
                orig_text = " ".join(orig_string_tokens)

                if fancy_answer:
                    final_text = get_final_text(norm_text, orig_text, do_lower_case, language)
                else:
                    final_text = orig_text

            else:
                final_text = ""

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            best_predictions.append(BestPrediction(
                text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if "" not in seen_predictions:
            best_predictions.append(BestPrediction(
                text="", start_logit=null_prediction.start_logit, end_logit=null_prediction.end_logit))


        # For some reason, the official code does this checks
        if len(best_predictions) == 1:
            best_predictions.append(BestPrediction(
                text="", start_logit=0, end_logit=0))
            
        #
        if len(best_predictions) == 0:
            best_predictions.append(BestPrediction(
                text="empty", start_logit=0, end_logit=0))

        # save the best non-null entry
        best_non_null_entry = None

        # iterate over the best prediction we found, convert them in a more favorable format
        # loop in descending score order (the first is the best)
        for entry in best_predictions:
            if entry.text:
                best_non_null_entry = entry
                break

        if best_non_null_entry:
            # compute the score diff as the difference between the null_score and the best-non_null score
            score_diff = (best_non_null_entry.start_logit + best_non_null_entry.end_logit) - null_score

            # if the score difference is below the threshold,
            # we predict our best-non_null answer
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id].append({
                    'answer': final_text, #best_prediction.text, 
                    'bert_score': float(best_non_null_entry.start_logit + best_non_null_entry.end_logit)
                })
            # otherwise we predict a null answer
            else:
                all_predictions[example.qas_id].append({
                    'answer': "", 
                    'bert_score': 0
                })
        # otherwise we predict a null answer
        else:
            all_predictions[example.qas_id].append({
                'answer': "", 
                'bert_score': 0
            })

    return all_predictions

def compute_recall(examples: List[SquadExample], num_ctx: int) -> float:
    num_questions = len(examples) // num_ctx
    sum_answers = 0
    for i in range(num_questions):
        examples_q = examples[i*num_ctx:(i+1)*num_ctx]

        for ex in examples_q:
            if ex.has_answer:
                sum_answers += 1
                break
        
    return sum_answers / num_questions


def compute_em_k(examples: List[SquadExample], predictions: Dict[str, List[Dict[str, Union[str, float]]]]):
    num_questions = len(predictions)
    num_ctx = len(examples) // num_questions

    sum_em_k = 0
    for i, qas_id in enumerate(predictions.keys()):
        
        gold_answers = [answer["text"] for answer in examples[i*num_ctx].answers if normalize_answer(answer["text"])]

        pred_answers = [normalize_answer(v['text']) for v in predictions[qas_id]]
        
        matches = 0
        for pred_ans in pred_answers:
            for gold_ans in gold_answers:
                matches += int(compute_exact(pred_ans, gold_ans))
        
        matches = matches > 0                 
                    
        sum_em_k += matches
            

    return sum_em_k/num_questions


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def compute_predictions_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = DefaultDict(lambda: [])
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits[0], n_best_size)
            end_indexes = _get_best_indexes(result.end_logits[0], n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0][0] + result.end_logits[0][0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0][0]
                    null_end_logit = result.end_logits[0][0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[0][start_index],
                            end_logit=result.end_logits[0][end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id].append(nbest_json[0]["text"])
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id].append({"text": "", "bert_score": score_null})
            else:
                all_predictions[example.qas_id].append({"text": best_non_null_entry.text, "bert_score": best_non_null_entry.start_logit + best_non_null_entry.end_logit})
        all_nbest_json[example.qas_id] = nbest_json

    # if output_prediction_file:
    #     with open(output_prediction_file, "w") as writer:
    #         writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # if output_nbest_file:
    #     with open(output_nbest_file, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # if output_null_log_odds_file and version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
