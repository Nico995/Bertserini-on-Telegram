import numpy as np
from bertserini_on_telegram.utils.io import print_ts
from transformers.data.processors.squad import SquadResult
from typing import Dict, List, Tuple
from pytorch_lightning import LightningModule
from transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from bertserini_on_telegram.utils.utils_squad import compute_predictions_logits

from transformers.data.metrics.squad_metrics import apply_no_ans_threshold, find_all_best_thresh, get_raw_scores, make_eval_dict, merge_eval

from bertserini_on_telegram.utils.utils_squad import compute_predictions
from transformers.data.metrics.squad_metrics import squad_evaluate
import json
from pprint import pprint

from bertserini_on_telegram.utils.utils_squad import compute_recall, compute_em_k


@MODEL_REGISTRY
class BERTModule(LightningModule):
    """A LightningModule is a neat way to organize the code necessary to train/evaluate/inference a Torch.nn.Module.

    Args:
        model_name (str, optional): The name of the pretrained model to use.
        mu (float): Weights used to compute the aggregated score. Defaults to 0.5.
        n_best (int): Number of best results to choose from when computing predictions. Defaults to 10.
        results_file (str): Name of the file where to store the optimal F1 threshold to use at inference time. Defaults to "./tmp/results_.json".

    Attributes:
        model (Torch.nn.Module): The effective Torch module, ready for validation/inference.
        tokenizer (BertTokenizer): The tokenizer used to tokenize all texts coming in or out of the model.
    """

    def __init__(self,
                 model_name: str,
                 results_file: str,
                 mu: float = 0.5,
                 n_best: int = 10,):

        super().__init__()

        self.save_hyperparameters()

        print_ts(f'Initializing {" ".join(self.hparams.model_name.split("-"))} for Inference')

        self.model = BertForQuestionAnswering.from_pretrained(self.hparams.model_name).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)

        self.all_results = []

    def get_best_f1_threshold(self):
        """Read the optimal F1 threshold from the filesystem.

        Returns:
            float: The optimal F1 threshold.

        """
        try:
            file = open(self.hparams.results_file, 'rb')
        except FileNotFoundError:
            print(f'Could not find file {self.hparams.results_file}. Remember to run a validation '
                  'loop first, before doing inference')
            exit()
        return json.load(file)['best_f1_thresh']

    def gradient_step(self, batch, batch_idx):
        """A simple training step (not used yet)
        """
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        outputs = self.model(**inputs)
        loss = outputs['loss']
        return loss

    def non_gradient_step(self, batch, batch_idx, dataloader_idx=0):
        """The common step for non_gradient loops (validation/test/inference)
        """

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        features = self.trainer.datamodule.features

        feature_indices = batch[3]

        outputs = self.model(**inputs)

        for feature_index in feature_indices:

            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            start_logits = outputs['start_logits'].detach().cpu()
            end_logits = outputs['end_logits'].detach().cpu()

            result = SquadResult(unique_id, start_logits, end_logits)

            self.all_results.append(result)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """The validation step
        """
        self.non_gradient_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        """This hook is called after the last validation step. 
        This is convenient if we want to gather the results of the validation.
        """

        print('Computing predictions logits')
        predictions = compute_predictions_logits(
            np.array(self.trainer.datamodule.new_examples),
            np.array(self.trainer.datamodule.features),
            np.array(self.all_results),
            n_best_size=10,
            max_answer_length=378,
            do_lower_case=True,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer,
            version_2_with_negative=True,
            output_prediction_file="./tmp/out_pred",
            output_nbest_file="./tmp/out_nbest",
            output_null_log_odds_file="./tmp/out_null_log_odds",
            verbose_logging=False,
        )

        # aggregate bert scores with pyserini
        # iterate over all the questions
        for i, qid in enumerate(predictions.keys()):
            # iterate over all the contexts for a give question
            for ctxid, _ in enumerate(predictions[qid]):
                # copy pyserini score from datamodule class
                predictions[qid][ctxid]['pyserini_score'] = self.trainer.datamodule.pyserini_scores[i][ctxid]
                # aggregate bert score with pyserini score with parameter mu
                predictions[qid][ctxid]['total_score'] = \
                    (1 - self.hparams.mu) * predictions[qid][ctxid]['bert_score'] + \
                    (self.hparams.mu) * predictions[qid][ctxid]['pyserini_score']

        
        em_k = compute_em_k(self.trainer.datamodule.new_examples, predictions)

        # sort answers for the different contexts by the total score
        # & transform prediction to feed them to squad_evaluate
        predictions = {k: sorted(v, key=lambda x: -x['total_score'])[0]['text'] for k, v in predictions.items()}

        result = squad_evaluate(self.trainer.datamodule.examples, predictions)
        recall = compute_recall(self.trainer.datamodule.new_examples, self.trainer.datamodule.hparams.num_contexts)
        pprint(f"em_k: {em_k}")
        pprint(f"recall: {recall}")
        print(f"other metrics: ")
        pprint(result)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The prediction step
        """
        self.non_gradient_step(batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, results):
        """This hook is called after the last prediction step. This is convenient if we want to gather the results of the prediction.
        """

        # Answers contains the n_best candidate answers for all possible question-context (q1, c1), ..., (q1, cn) pairs
        print('Computing predictions logits')
        predictions = compute_predictions_logits(
            np.array(self.trainer.datamodule.examples),
            np.array(self.trainer.datamodule.features),
            np.array(self.all_results),
            n_best_size=10,
            max_answer_length=378,
            do_lower_case=True,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer,
            version_2_with_negative=True,
            output_prediction_file="./tmp/out_pred",
            output_nbest_file="./tmp/out_nbest",
            output_null_log_odds_file="./tmp/out_null_log_odds",
            verbose_logging=False,
        )

        # aggregate bert scores with pyserini
        # iterate over all the questions
        for i, qid in enumerate(predictions.keys()):
            # iterate over all the contexts for a give question
            predictions[qid][0]['pyserini_score'] = self.trainer.datamodule.pyserini_scores[i]

            # aggregate bert score with pyserini score with parameter mu
            predictions[qid][0]['total_score'] = \
                (1 - self.hparams.mu) * predictions[qid][0]['bert_score'] + \
                (self.hparams.mu) * predictions[qid][0]['pyserini_score']

        # We now need to compute the aggregate scores for the answers and select the highest scoring one
        sorted_answers = sorted([v[0] for v in predictions.values()], key=lambda x: -x['total_score'])
        best_answer = sorted_answers[0]['text']

        print(f"answers: \n{sorted_answers}")
        self.answer = best_answer

