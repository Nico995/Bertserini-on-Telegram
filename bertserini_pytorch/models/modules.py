from collections import defaultdict
from typing import Dict, List, Tuple, Union
from pytorch_lightning import LightningModule
from transformers import AdamW, AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
from datetime import datetime
import datasets
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from bertserini_pytorch.utils.base import Context, Question
from bertserini_pytorch.utils.io import print_ts
from bertserini_pytorch.utils.pyserini import build_searcher, retriever


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
                 mu: float = 0.5):

        super().__init__()
        print_ts(f'Initializing {" ".join(pretrained_model_name.split("-"))} for Inference')

        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.searcher = build_searcher('enwiki-paragraphs')

        self.k = context_size
        self.mu = mu

    def predict(self, question: str) -> Tuple[str, float]:
        """This function predicts start and end logits of the ansewr for a question give a list of pretexts.


        Args:
            question (str): [description]
            pretexts (List[str]): [description]
        """

        # first we retrieve the top k paragraphs from pyserini
        # to do so, we need to wrap our question into a Question object
        contexts = retriever(Question(question, language='en'), self.searcher, self.k)
        answers = []

        # ask the question once for every available context
        for context in contexts:

            # tokenize question and context together
            tokens = self.tokenizer.encode_plus(question, context.text)
            input_ids, token_type_ids = tokens['input_ids'], tokens['token_type_ids']

            # convert ids to string-tokens, keep them to answer in Natural Language
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # get the output of the model
            # the model's output will be composed by
            output = self.model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([token_type_ids]))

            # we care about the log probabilities of the start and end of the
            # sentence, computed for every token in the question/context

            # even though argmax is not affected by the softmax function
            # we also compute the probability distribution because we will need
            # the scores for later
            start_scores = torch.softmax(output.start_logits, dim=-1)
            end_scores = torch.softmax(output.end_logits, dim=-1)
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)

            # we only care about the answer if the start of the sentence occurs
            # before the end. If that's not the case, we say that the model
            # failed to answer.
            if answer_end >= answer_start:

                # now we build the answer in natural language, taking into account
                # that bert uses wordpiece tokenization to handle rare words and
                # to keep the size of the vocabulary from exploding

                answer = tokens[answer_start]
                for token in tokens[answer_start+1:answer_end+1]:

                    # the ## signs leads the second part of a wordpiece tokenization
                    # if we encouter it, we just concatenate directly after the previous
                    # token
                    if token[0:2] == "##":
                        answer += token[2:]
                    # otherwise leave an empty space and insert the next token
                    else:
                        answer += " " + token
            else:
                continue
            # in order to assess which of the obtained answers is "better than the other"
            # we store the retreival score and the prediction scores, which we will use
            # to rank the answers
            bert_score = torch.max(start_scores) + torch.max(end_scores)
            answers.append((answer, bert_score, context.score))

        # now that we predicted from every context we had, it's time to find out which is the
        # best answer we got
        best_answer = self._get_best_answer(answers)

        return best_answer

    def _get_best_answer(self, answers: Dict[str, List[Tuple[str, float, float]]]) -> Tuple[str, float]:
        answers_scores = []

        for answer, bert_score, pyserini_score in answers:
            # linearly interpolate the two scores with the mu value
            overall_score = self.mu * bert_score + (1 - self.mu) * pyserini_score
            answers_scores.append((answer, overall_score))

        # sort the answers and get the highest-scoring one
        best_answer = sorted(answers_scores, key=lambda x: -x[1])[0]

        return best_answer
