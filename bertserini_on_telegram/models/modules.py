from bertserini_on_telegram.utils.io import print_ts
from transformers.data.processors.squad import SquadResult
from typing import Dict, List, Tuple
from pytorch_lightning import LightningModule
from transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from transformers.data.metrics.squad_metrics import apply_no_ans_threshold, find_all_best_thresh, get_raw_scores, make_eval_dict, merge_eval

from bertserini_on_telegram.utils.utils_squad import compute_predictions
from transformers.data.metrics.squad_metrics import squad_evaluate
import json


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
        searcher (Searcher): Wrapped SimpleSearcher object from pyserini, used to retrieve Context objects from prebuilt indices.
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
        """This hook is called after the last validation step. This is convenient if we want to gather the results of the validation.
        """

        all_predictions = compute_predictions(
            self.trainer.datamodule.examples,
            self.trainer.datamodule.features,
            self.all_results,
            n_best=10,
            max_answer_length=378,
            do_lower_case=True,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer
        )

        # Remove the scores from the predictions, only retain texts
        predictions = {k: v[0] for k, v in all_predictions.items()}

        # This is mainly done to retrieve the best_f1_threshold
        result = squad_evaluate(self.trainer.datamodule.examples, predictions)
        
        # Save the results to filesystem, we will need the best_f1_thresh later at 
        # inference time to act as the threshold for predicting the null answer
        with open(self.hparams.results_file, "w") as f:
            f.write(json.dumps(result, indent=4) + "\n")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The prediction step
        """
        self.non_gradient_step(batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, results):
        """This hook is called after the last prediction step. This is convenient if we want to gather the results of the prediction.
        """

        # Answers contains the n_best candidate answers for all possible question-context (q1, c1), ..., (q1, cn) pairs
        answers = compute_predictions(
            self.trainer.datamodule.examples,
            self.trainer.datamodule.features,
            self.all_results,
            n_best=self.hparams.n_best,
            max_answer_length=30,
            do_lower_case=True,
            null_score_diff_threshold=self.get_best_f1_threshold(),
            tokenizer=self.tokenizer,
            language="en")

        # We now need to compute the aggregate scores for the answers and select the highest scoring one
        scored_answers = zip([ctx.score for ctx in self.trainer.datamodule.contexts], answers.values())
        scored_answers = sorted(scored_answers, key=lambda x: self.hparams.mu * x[0] + (1-self.hparams.mu) * x[1][1], reverse=True)
    
        self.answer = scored_answers[0][1][0]
