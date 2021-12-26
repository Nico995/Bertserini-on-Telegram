from bertserini_pytorch.utils.pyserini import build_searcher, craft_squad_examples, retriever
from bertserini_pytorch.utils.io import print_ts
from bertserini_pytorch.utils.base import Answer, Question, Context
from transformers.data.processors.squad import SquadResult, squad_convert_examples_to_features
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Union
from pytorch_lightning import LightningModule
from transformers import AdamW, AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
from datetime import datetime
from torch.utils.data import DataLoader, SequentialSampler
import datasets
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from transformers.data.metrics.squad_metrics import apply_no_ans_threshold, compute_predictions_logits, find_all_best_thresh, get_raw_scores, make_eval_dict, merge_eval

from bertserini_pytorch.utils.utils_squad import compute_logits
from transformers.data.metrics.squad_metrics import squad_evaluate
from bertserini_pytorch.utils.utils_squad import tensor_to_list
import json


@MODEL_REGISTRY
class BERTTrainer(LightningModule):

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        #
        self.tokenizer = AutoTokenizer.from_pretrained('rsvp-ai/bertserini-bert-base-squad')
        #

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.hparams.num_labels)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, config=self.config)
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.metric = datasets.load_metric("squad", self.hparams.task_name, experiment_id=now)

        self.args = {
            "max_seq_length": 384,
            "doc_stride": 128,
            "max_query_length": 64,
            "threads": 1,
            "tqdm_enabled": False,
            "n_best_size": 20,
            "max_answer_length": 30,
            "do_lower_case": True,
            "output_prediction_file": False,
            "output_nbest_file": None,
            "output_null_log_odds_file": None,
            "verbose_logging": False,
            "version_2_with_negative": True,
            "null_score_diff_threshold": 0,
        }

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        # labels = batch["labels"]

        # return {"loss": val_loss, "preds": preds, "labels": labels}
        return {"loss": val_loss, "preds": preds}

    def validation_epoch_end(self, outputs):
        print('on validation epoch end: ', outputs)
        preds = torch.stack([x["preds"] for x in outputs]).detach().cpu().numpy()
        # labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


@MODEL_REGISTRY
class BERTPredictor(LightningModule):
    def __init__(self,
                 pretrained_model_name: str = 'bert-large-uncased-whole-word-masking-finetuned-squad',
                 context_size: int = 20,
                 mu: float = 0.5,
                 n_best: int = 10,
                 score_diffs_file: str = './tmp/null_odds_.json',
                 all_predictions_file: str = './tmp/predictions_.json',
                 results_file: str = './tmp/results_.json'):

        super().__init__()

        self.save_hyperparameters()

        print_ts(f'Initializing {" ".join(self.hparams.pretrained_model_name.split("-"))} for Inference')

        self.model = BertForQuestionAnswering.from_pretrained(self.hparams.pretrained_model_name).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_model_name)
        self.searcher = build_searcher('enwiki-paragraphs')

        self.all_results = []

    def compute_scores(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
        qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
        has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
        no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

        if no_answer_probs is None:
            no_answer_probs = {k: 0.0 for k in preds}

        exact, f1 = get_raw_scores(examples, preds)

        exact_threshold = apply_no_ans_threshold(
            exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )
        f1_threshold = apply_no_ans_threshold(
            f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

        evaluation = make_eval_dict(exact_threshold, f1_threshold)

        if has_answer_qids:
            has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
            merge_eval(evaluation, has_ans_eval, "HasAns")

        if no_answer_qids:
            no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
            merge_eval(evaluation, no_ans_eval, "NoAns")

        if no_answer_probs:
            find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

        return evaluation

    def get_thresholds(self):
        try:
            file = open(self.hparams.results_file, 'rb')
        except FileNotFoundError:
            print(f'Could not find file {self.hparams.results_file}. Remember to run a validation '
                  'loop first, before doing inference')
            exit()
        return json.load(file)

    def non_gradient_step(self, batch, batch_idx, dataloader_idx=0):
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

            # output = [list(output[i].detach().cpu()) for output in outputs]

            start_logits = outputs['start_logits'].detach().cpu()
            end_logits = outputs['end_logits'].detach().cpu()

            result = SquadResult(unique_id, start_logits, end_logits)

            self.all_results.append(result)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.non_gradient_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):

        all_predictions = compute_logits(
            self.trainer.datamodule.examples,
            self.trainer.datamodule.features,
            self.all_results,
            n_best=10,
            max_answer_length=378,
            do_lower_case=True,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer
        )

        predictions = {k: v[0] for k, v in all_predictions.items()}

        result = squad_evaluate(self.trainer.datamodule.examples, predictions)

        with open(self.hparams.results_file, "w") as f:
            f.write(json.dumps(result, indent=4) + "\n")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.non_gradient_step(batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, results):

        answers = compute_logits(
            self.trainer.datamodule.examples,
            self.trainer.datamodule.features,
            self.all_results,
            n_best=self.hparams.n_best,
            max_answer_length=30,
            do_lower_case=True,
            null_score_diff_threshold=self.get_thresholds()['best_f1_thresh'],
            tokenizer=self.tokenizer,
            language="en")

        scored_answers = zip([ctx.score for ctx in self.trainer.datamodule.contexts], answers.values())
        scored_answers = sorted(scored_answers, key=lambda x: x[0] + x[1][1], reverse=True)
        # scored_answers = sorted(scored_answers, key=lambda x: x[1][1], reverse=True)
        # print(scored_answers)
        self.answer = scored_answers[0][1][0]

    def _get_best_answer(self, answers: Dict[str, List[Tuple[str, float, float]]]) -> Tuple[str, float]:
        answers_scores = []

        for answer, bert_score, pyserini_score in answers:
            # linearly interpolate the two scores with the mu value
            overall_score = self.hparams.mu * bert_score + (1 - self.hparams.mu) * pyserini_score
            answers_scores.append((answer, overall_score))

        # sort the answers and get the highest-scoring one
        best_answer = sorted(answers_scores, key=lambda x: -x[1])[0]

        return best_answer
