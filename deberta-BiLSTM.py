# code
import re
import os
import json
import argparse
import random, string, random
from itertools import chain
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from transformers.utils import PaddingStrategy
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from tokenizers import AddedToken
import evaluate
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict
import string, random

# Custom modules
from src.model_bilstm import CustomModel
from src.losses import FocalLoss, JaccardLoss
from src.sift import AdversarialLearner, hook_sift_layer
from src.piidd_postprocessing import label_postprocessing

import warnings

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["NCCL_IB_GID_INDEX"]="2"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/468844
def filter_no_pii(example, percent_allow=0.4):
    has_pii = set("O") != set(example["provided_labels"])
    return has_pii or (random.random() < percent_allow)


# ===========================================================================================================================================
def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    token_map = []
    labels = []

    idx = 0

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        token_map.extend([idx] * len(t))
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)

        idx += 1

    # actual tokenization
    tokenized = tokenizer(
        "".join(text), return_offsets_mapping=True, max_length=max_length
    )

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        try:
            token_labels.append(label2id[labels[start_idx]])
        except:
            continue

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "token_map": token_map,
    }


# ===========================================================================================================================================
# https://www.kaggle.com/code/conjuring92/pii-metric-fine-grained-eval


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1 + (beta**2)) * p * r / ((beta**2) * p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


# ===========================================================================================================================================
def compute_metrics(p, id2label, valid_ds, valid_df, doc2tokens, data):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    predictions, labels = p

    pred_df = parse_predictions(predictions, id2label, valid_ds, doc2tokens, data)

    print()
    print(pred_df)

    references = {
        (str(row.document), row.token, row.label) for row in valid_df.itertuples()
    }
    predictions = {
        (str(row.document), row.token, row.label) for row in pred_df.itertuples()
    }

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)
        if pred_type != "O":
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != "O":
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    results = {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {
            k: v.to_dict() for k, v in score_per_type.items() if k != "O"
        },
    }

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                if isinstance(v, dict):
                    for n2, v2 in v.items():
                        final_results[f"{key}_{n}_{n2}"] = v2
                else:
                    final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value

    return final_results


# ===========================================================================================================================================
def parse_predictions(predictions, id2label, ds, doc2tokens, data):

    pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=2).reshape(
        predictions.shape[0], predictions.shape[1], 1
    )
    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:, :, : len(id2label) - 1].argmax(-1)
    O_preds = pred_softmax[:, :, len(id2label) - 1]

    thresholds = {
        "EMAIL": 0.5,
        "ID_NUM": 0.6,
        "NAME_STUDENT": 0.8,
        "PHONE_NUM": 0.5,
        "STREET_ADDRESS": 0.5,
        "URL_PERSONAL": 0.5,
        "USERNAME": 0.8,
    }

    print(thresholds)

    indexes = defaultdict(list)
    for k, v in id2label.items():
        if k != len(id2label) - 1:
            indexes[v.split("-")[1]].append(int(k))

    for label_name, label_threshold in thresholds.items():
        if len(indexes[label_name]) == 1:
            preds = np.where(
                O_preds < label_threshold,
                np.where(
                    preds_without_O == indexes[label_name][0], preds_without_O, preds
                ),
                preds,
            )
        else:
            preds = np.where(
                O_preds < label_threshold,
                np.where(
                    (preds_without_O == indexes[label_name][0])
                    | (preds_without_O == indexes[label_name][1]),
                    preds_without_O,
                    preds,
                ),
                preds,
            )

    triplets = set()
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(
        preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]
    ):

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0:
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map):
                break

            token_id = token_map[start_idx]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                triplet = (doc, label_pred, token_id, tokens[token_id])

                if triplet not in triplets:
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[token_id])
                    triplets.add(triplet)

    df = pd.DataFrame(
        {"document": document, "token": token, "label": label, "token_str": token_str}
    )

    df = label_postprocessing(df, doc2tokens, data)

    return df


# ===========================================================================================================================================
def get_reference_df(fold):

    ref_df = pd.read_json(
        f"data/piidd-balanced-cv-split/COMPETITION_FOLD_{fold}.json"
    )
    ref_df = ref_df[["document", "tokens", "labels"]].copy()
    ref_df = (
        ref_df.explode(["tokens", "labels"])
        .reset_index(drop=True)
        .rename(columns={"tokens": "token", "labels": "label"})
    )
    ref_df["token"] = ref_df.groupby("document").cumcount()

    reference_df = ref_df[ref_df["label"] != "O"].copy()
    reference_df = reference_df.reset_index().rename(columns={"index": "row_id"})
    reference_df = reference_df[["row_id", "document", "token", "label"]].copy()

    return reference_df


# ===========================================================================================================================================
def convert_to_ds(data):

    return Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in data],
            "document": [str(x["document"]) for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        }
    )


# ===========================================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--validation_fold", type=int, required=False)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--save_steps", type=float)
    parser.add_argument("--o_weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adv_mode", type=str)
    parser.add_argument("--adv_start", type=int)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--smoke_test", type=int)
    parser.add_argument("--fullfit", type=int)

    args = parser.parse_args()

    seed_everything(args.seed)
    ADV_MODE = args.adv_mode
    ADV_START = args.adv_start
    LOSS = args.loss
    OUTPUT_DIR = f"../working/{args.output_dir}"

    print("args ", args)
    data = json.load(
        open("data/pii-detection-removal-from-educational-data/train.json")
    )
    doc2tokens = {str(row["document"]): row["tokens"] for row in data}
    print("original datapoints: ", len(data))
    mixtral = json.load(open("data/pii-dd-mistral-generated/mixtral-8x7b-v1.json"))
    print("mixtral datapoints: ", len(mixtral))
    mpware = json.load(
        open(
            "data/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1-no-i-username.json"
        )
    )
    print("mpware datapoints: ", len(mpware))

    yuv = json.load(open("data/external/external_data_v8.json"))
    print("yuv datapoints: ", len(yuv))
    # COnvert all other than Name student to label as O
    def keepselectedlabels(ds, labels=["B-NAME_STUDENT", "I-NAME_STUDENT", "O"]):
        for item in ds:
            current_labels = item.get("labels", [])
            updated_labels = []
            for label in current_labels:
                if label not in labels:
                    updated_labels.append("O")
                else:
                    updated_labels.append(label)
            item["labels"] = updated_labels
        return ds

    #     data=keepselectedlabels(data)
    #     mpware=keepselectedlabels(mpware)
    #     mixtral=keepselectedlabels(mixtral)
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    print("Label2id ", label2id)
    id2label = {v: k for k, v in label2id.items()}

    if args.fullfit == 0:
        print("---> Reading validation dataframe")
        reference_df = get_reference_df(args.validation_fold)
        print(reference_df)
        print()
        print()
        validation_df, train_df = None, None
        for FOLD in range(4):
            if FOLD == args.validation_fold:
                validation_df = json.load(
                    open(
                        f"data/piidd-balanced-cv-split/COMPETITION_FOLD_{FOLD}.json"
                    )
                )
            else:
                if train_df is None:
                    train_df = json.load(
                        open(
                            f"data/piidd-balanced-cv-split/COMPETITION_FOLD_{FOLD}.json"
                        )
                    )
                else:
                    train_df += json.load(
                        open(
                            f"data/piidd-balanced-cv-split/COMPETITION_FOLD_{FOLD}.json"
                        )
                    )
    else:
        print("=== Doing Fullfit ===")
        train_df = data
    # train_df=keepselectedlabels(train_df)
    # train_df = train_df.filter(filter_no_pii,num_proc=2,)
    print(set(train_df[0]["labels"]))
    train_df += mpware + yuv

    if args.smoke_test == 1:
        train_df = train_df[:20]
        validation_df = validation_df[:20]

    train_ds = convert_to_ds(train_df)
    train_ds = train_ds.shuffle(seed=42)
    if args.fullfit == 0:
        validation_ds = convert_to_ds(validation_df)
        print(f" VAL {len(validation_ds)} TRAIN {len(train_ds)}")
    else:
        print(f" TRAIN {len(train_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_tokens(AddedToken("\n", normalized=False))
    train_ds = train_ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
        },
        num_proc=torch.cuda.device_count(),
    )
    if args.fullfit == 0:
        validation_ds = validation_ds.map(
            tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label2id": label2id,
                "max_length": args.max_length,
            },
            num_proc=torch.cuda.device_count(),
        )

    metric = evaluate.load("seqeval")
    config = AutoConfig.from_pretrained(args.model_path)
    config.output_hidden_states = False
    config.id2label = id2label
    config.num_labels = len(label2id)
    config.label2id = label2id
    model = CustomModel.from_pretrained(args.model_path, config=config)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    # model= backbone #CustomModel(config,class_weights=None,bilstm_layer=False)

    FREEZE_EMBEDDINGS = False
    FREEZE_LAYERS = 0
    if FREEZE_EMBEDDINGS:
        print("Freezing embeddings.")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False

    if FREEZE_LAYERS > 0:
        print(f"Freezing {FREEZE_LAYERS} layers.")
        for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False

    class CustomTrainer(Trainer):
        def __init__(
            self,
            *args,
            class_weights=None,
            adv_start=2,
            adv_mode="epoch",  # "step"
            loss="ce",  # focal_ce
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            # Assuming class_weights is a Tensor of weights for each class
            self.class_weights = torch.tensor([1.0] * 12 + [0.05]).to("cuda")
            self._adv_started = False
            self.adv_mode = adv_mode
            self.loss = loss
            self.adv_start = adv_start  # step or epoch at which to start perbutation
            adv_modules = hook_sift_layer(
                self.model, hidden_size=self.model.config.hidden_size
            )
            self.adv = AdversarialLearner(self.model, adv_modules)
            self._adv_started = False
            # self.awp = AWP(model=model,adv_eps=1e-3,adv_lr=1e-6,device=self.args.device,start_epoch=0)

        def compute_loss(self, model, inputs, return_outputs=False):

            #             for i in range(len(self.class_weights)):
            #                 if i == 2 or i == 8:
            #                     self.class_weights[i] = 1.0
            #                 elif i == len(self.class_weights) - 1:
            #                     self.class_weights[i] = 0.05
            #                 else:
            #                     self.class_weights[i] = 1.0 #min(1.0, 0.5 + ((1/4) * self.state.epoch))

            labels = inputs.pop("labels").to(self.args.device)
            outputs = model(**inputs)
            logits = outputs.logits
            #             if self.state.global_step % 100 == 0:
            #                 print(self.class_weights)

            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.args.device)
            )
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index = 12, label_smoothing = 0.03)
            if self.label_smoother is not None and "labels" in inputs:
                loss = self.label_smoother(outputs, inputs)
            else:
                if self.loss == "ce":
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )
                elif self.loss == "focal_ce":
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )
                    focal = FocalLoss(weight=self.class_weights.to(self.args.device))
                    loss = loss + focal(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )
                elif self.loss == "jaccard_ce":
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )
                    jaccard_loss = JaccardLoss(log_loss=False, from_logits=True)
                    loss = loss + jaccard_loss(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )
                elif self.loss == "focal":
                    # print("USING FOCAL LOSS")
                    focal = FocalLoss(weight=self.class_weights.to(self.args.device))
                    loss = focal(
                        logits.view(-1, self.model.config.num_labels), labels.view(-1)
                    )

            current_state = (
                self.state.global_step if self.adv_mode == "step" else self.state.epoch
            )

            # Logits fn for adv
            def logits_fn(model, *wargs, **kwargs):
                wargs_device = [arg.to(self.args.device) for arg in wargs]
                kwargs_device = {k: v.to(self.args.device) for k, v in kwargs.items()}
                o = model(*wargs_device, **kwargs_device)
                return o.logits

            # print(f"current_state {current_state}")
            if current_state >= self.adv_start:
                if not self._adv_started:
                    self._adv_started = True
                    print("Starting adversarial training")
                loss = loss + self.adv.loss(
                    outputs.logits, logits_fn, loss_fn="mse", **inputs
                )

            return (loss, outputs) if return_outputs else loss

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    if args.fullfit == 0:
        arguments = TrainingArguments(
            output_dir=OUTPUT_DIR,
            fp16=True,
            # bf16=False,
            # fp16_opt_level="O0",
            # bf16=True,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to="none",
            logging_steps=10,
            gradient_accumulation_steps=2,
            metric_for_best_model="ents_f5",
            greater_is_better=True,
            gradient_checkpointing=True,
            num_train_epochs=args.num_train_epochs,
            dataloader_num_workers=torch.cuda.device_count(),
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            eval_steps=args.save_steps,
            lr_scheduler_type="cosine",
            save_total_limit=2,
            save_strategy="steps",
            save_steps=args.save_steps,
            seed=args.seed,
            save_safetensors=False,
        )
        print(
            "Combined Train with external : ",
            len(train_ds),
            "validation ",
            len(validation_ds),
        )

        trainer = CustomTrainer(
            model=model,
            args=arguments,
            train_dataset=train_ds,
            eval_dataset=validation_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=partial(
                compute_metrics,
                id2label=id2label,
                valid_ds=validation_ds,
                valid_df=reference_df,
                doc2tokens=doc2tokens,
                data=validation_df,
            ),
            adv_mode=ADV_MODE,
            adv_start=ADV_START,
            loss=LOSS,
            callbacks=[early_stopping],
        )
    else:
        arguments = TrainingArguments(
            output_dir=OUTPUT_DIR,
            fp16=True,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to="none",
            logging_steps=100,
            gradient_accumulation_steps=2,
            metric_for_best_model="ents_f5",
            greater_is_better=True,
            gradient_checkpointing=True,
            num_train_epochs=args.num_train_epochs,
            dataloader_num_workers=torch.cuda.device_count(),
            load_best_model_at_end=False,
            lr_scheduler_type="cosine",
            evaluation_strategy="no",
            do_eval=False,
            save_total_limit=3,
            save_strategy="steps",
            save_steps=args.save_steps,
            seed=args.seed,
            save_safetensors=False,
        )

        trainer = CustomTrainer(
            model=model,
            args=arguments,
            train_dataset=train_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            adv_mode=ADV_MODE,
            adv_start=ADV_START,
            loss=LOSS,
            # callbacks=[early_stopping],
        )

    print()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if args.fullfit == 0:
        print("BEST MODEL ", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
