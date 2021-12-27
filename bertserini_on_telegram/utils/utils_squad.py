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
from typing import List, Mapping, Tuple

from transformers import BertTokenizer
from transformers.data.metrics.squad_metrics import squad_evaluate, get_final_text
from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadResult


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
    all_predictions = collections.OrderedDict()

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

        if len(best_predictions) == 1:
            best_predictions.append(BestPrediction(
                text="", start_logit=0, end_logit=0))

        # save the best non-null entry
        best_non_null_entry = None

        # iterate over the best prediction we found, convert them in a more favorable format
        # loop in descending score order (the first is the best)
        for entry in best_predictions:
            if entry.text:
                best_non_null_entry = entry
                break

        # compute the score diff as the difference between the null_score and the best-non_null score
        score_diff = null_score - (best_non_null_entry.start_logit + best_non_null_entry.end_logit)

        # if the score difference is below the threshold,
        # we predict a null answer
        if score_diff > null_score_diff_threshold:
            all_predictions[example.qas_id] = ("", 0)

        # otherwise we predict our best-non_null answer
        else:
            all_predictions[example.qas_id] = (
                best_non_null_entry.text,
                float(best_non_null_entry.start_logit + best_non_null_entry.end_logit))

    return all_predictions
