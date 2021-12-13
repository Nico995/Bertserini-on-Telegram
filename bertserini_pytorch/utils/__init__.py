
task_text_field_map = {
    "cola": ["sentence"],
    "sst2": ["sentence"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "stsb": ["sentence1", "sentence2"],
    "mnli": ["premise", "hypothesis"],
    "qnli": ["question", "sentence"],
    "rte": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
    "ax": ["premise", "hypothesis"],
}

glue_task_num_labels = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "ax": 3,
}

loader_columns = [
    "datasets_idx",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
    "labels",
]
