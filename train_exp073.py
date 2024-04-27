from transformers import get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup


def get_scheduler(optimizer, config, num_train_steps):
    if config.scheduler.type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.constant_schedule_with_warmup.n_warmup_steps
        )
    elif config.scheduler.type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.linear_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif config.scheduler.type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.cosine_schedule_with_warmup.n_warmup_steps,
            num_cycles=config.scheduler.cosine_schedule_with_warmup.n_cycles,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler.type == 'polynomial_decay_schedule_with_warmup':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.polynomial_decay_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps,
            power=config.scheduler.polynomial_decay_schedule_with_warmup.power,
            lr_end=config.scheduler.polynomial_decay_schedule_with_warmup.min_lr
        )
    else:
        raise ValueError(f'Unknown scheduler: {config.scheduler.scheduler_type}')

    return scheduler


# File: src/training/scheduler.py

from collections import defaultdict
from typing import Dict
from tqdm.auto import tqdm


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

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_fbeta(valid_df, pred_df):
    references = {(row.document, row.token, row.label) for row in valid_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1] # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:] # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:] # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf


    results = {
        "precision": totals.precision,
        "recall": totals.recall,
        "fbeta": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k!= 'O'},
    }
    return results


def compute_metrics(p, id2label, valid_ds, valid_df, threshold=0.9):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    predictions, labels = p

    pred_df = parse_predictions(predictions, id2label, valid_ds, threshold=threshold)

    all_labels = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']
    keep = []
    for label in tqdm(pred_df.label.values):
        if label not in all_labels:
            keep.append(False)
        else:
            keep.append(True)

    pred_df = pred_df[keep]
    fbeta_post = compute_fbeta(valid_df, pred_df)

    pred_df = parse_predictions(predictions, id2label, valid_ds, threshold=0)
    fbeta = compute_fbeta(valid_df, pred_df)

    fbeta_best = max(fbeta_post['fbeta'], fbeta['fbeta'])

    return {'fbeta': fbeta['fbeta'], 'fbeta_best': fbeta_best, 'fbeta_post': fbeta_post['fbeta']}


def pii_fbeta_score(pred_df, gt_df, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """

    df = pred_df.merge(gt_df, how="outer", on=["document", "token"], suffixes=("_pred", "_gt"))
    df["cm"] = ""

    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    df.loc[(df.label_gt.notna() & df.label_pred.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP" # CHANGED

    df.loc[
        (df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt == df.label_pred), "cm"
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()
    s_micro = (1+(beta**2))*TP/(((1+(beta**2))*TP) + ((beta**2)*FN) + FP)
    #### some changes to check wandb versioning

    return s_micro


# File: src/training/metrics.py

import torch
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_valid_steps(num_train_steps, n_evaluations):
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps


def parse_predictions(predictions, id2label, ds, threshold=0.9):
    pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis = 2).reshape(predictions.shape[0],predictions.shape[1],1)
    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:,:,:12].argmax(-1)
    O_preds = pred_softmax[:,:,12]
    preds_final = np.where(O_preds < threshold, preds_without_O , preds)

    pairs = []
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc, indices in zip(preds_final, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"], ds["token_indices"]):
        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0: continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): break

            original_token_id = token_map[start_idx]
            token_id = indices[original_token_id]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                pair=(doc, token_id)

                if pair not in pairs:
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[original_token_id])
                    pairs.append(pair)

    df = pd.DataFrame({
        # "eval_row": row,
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })

    df = df.drop_duplicates(['document', 'token', 'label', 'token_str']).reset_index(drop=True)

    df["row_id"] = list(range(len(df)))
    return df


# File: src/training/utils.py

# import torch
import torch.nn as nn
from types import SimpleNamespace
import numpy as np
import gc
from tqdm import tqdm
import torch
import random
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_to_device(batch):
    for k, v in batch.items():
        if type(v) == dict:
            for _k, _v in v.items():
                if len(v) == 1:
                    v = v[0].unsqueeze(0)
                v[_k] = _v.to(device)
            batch[k] = v
        else:
            if len(v) == 1:
                v = v[0].unsqueeze(0)
            batch[k] = v.to(device)
    return batch


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            config: SimpleNamespace,
            train_dataloader: torch.utils.data.DataLoader=None,
            valid_dataloader: torch.utils.data.DataLoader=None,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler=None,
            eval_steps=None,
            callbacks=None,
    ) -> None:

        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.training.apex)
        self.eval_steps = eval_steps

        self.callbacks = callbacks

    def validate(self):
        self.model.eval()

        self.callbacks.on_valid_epoch_start()

        predictions = []
        target = []

        for step, inputs in enumerate(self.valid_dataloader):
            inputs = batch_to_device(inputs)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    y_pred, loss = self.model(inputs)

            y_pred = torch.sigmoid(y_pred)
            predictions.append(y_pred.detach().to('cpu').numpy())
            target.append(inputs['labels'].cpu().numpy())

            self.callbacks.on_valid_step_end(loss)

        predictions = np.concatenate(predictions)
        target = np.concatenate(target)
        return predictions, target


    def predict(self, test_loader):
        predictions = []
        self.model.eval()
        self.model.to(device)

        for inputs in tqdm(test_loader, total=len(test_loader)):
            inputs = batch_to_device(inputs)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    y_preds, _ = self.model(inputs)

            y_preds = torch.sigmoid(y_preds)
            predictions.append(y_preds.detach().to('cpu').numpy())
        predictions = np.concatenate(predictions)
        return predictions

    def get_embeddings(self, test_loader):
        predictions = []
        self.model.eval()
        self.model.to(device)

        for inputs in tqdm(test_loader, total=len(test_loader)):
            inputs = batch_to_device(inputs)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model.backbone(inputs['input_ids'], inputs['attention_mask'])
                    embeddings = self.model.pooling(inputs, outputs)

            predictions.append(embeddings.detach().to('cpu').numpy())
        predictions = np.concatenate(predictions)
        return predictions

    def train(self):
        self.model.to(device)

        self.callbacks.on_training_start()

        for epoch in range(self.config.training.epochs):
            self.model.train()

            self.callbacks.on_train_epoch_start()
            for step, inputs in enumerate(self.train_dataloader):
                inputs = batch_to_device(inputs)

                if self.config.training.apex:
                    with torch.cuda.amp.autocast():
                        y_pred, loss = self.model(inputs)
                        raw_loss = loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    y_pred, loss, skip = self.model(inputs)
                    raw_loss = loss.item()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                learning_rates = self.scheduler.get_last_lr()
                self.callbacks.on_train_step_end(raw_loss, grad_norm, learning_rates)

                if (step + 1) in self.eval_steps:
                    predictions, target = self.validate()
                    self.model.train()

                    self.callbacks.on_valid_epoch_end(target, predictions)
                    score_improved = self.callbacks.get('MetricsHandler').is_valid_score_improved()
                    if score_improved:
                        self.save_best_model(predictions)

            self.callbacks.on_train_epoch_end()
            self.save_checkpoint()

        torch.cuda.empty_cache()
        gc.collect()
        return None


    def save_best_model(self, predictions):
        torch.save(
            {
                'model': self.model.state_dict(),
                'predictions': predictions
            },
            self.config.best_model_path
        )

    def save_checkpoint(self):
        torch.save(
            {
                'model': self.model.state_dict()
            },
            self.config.checkpoint_path
        )



# File: src/training/trainer.py

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel
import torch
import torch.nn as nn


class CustomModel(DebertaPreTrainedModel):

    def __init__(self, config, model_path='microsoft/deberta-v3-large'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.deberta = AutoModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_model(config, model_path, id2label, label2id):
    all_labels = list(label2id.keys())
    backbone_config = AutoConfig.from_pretrained(model_path)
    backbone_config.hidden_dropout = config.model.dropout
    backbone_config.hidden_dropout_prob = config.model.dropout
    backbone_config.attention_dropout = config.model.dropout
    backbone_config.attention_probs_dropout_prob = config.model.dropout
    backbone_config.num_labels = len(all_labels)
    backbone_config.id2label = id2label
    backbone_config.label2id = label2id

    if config.model.lstm:
        print('LSTM')
        model = CustomModel(
            backbone_config,
            model_path,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            config=backbone_config,
            ignore_mismatched_sizes=True
        )

    if config.model.freeze_embeddings:
        freeze(model.deberta.embeddings)

    if config.model.freeze_n_layers > 0:
        freeze(model.deberta.encoder.layer[:config.model.freeze_n_layers])

    return model

# File: src/training/model.py

import torch.nn as nn

def get_criterion(config):
    return nn.BCEWithLogitsLoss(reduction='mean')


# File: src/training/criterion.py

from transformers import AdamW
import math


def get_parameters_groups(n_layers, n_groups):
    layers = [f'backbone.encoder.layer.{n_layers - i - 1}.' for i in range(n_layers)]
    step = math.ceil(n_layers / n_groups)
    groups = []
    for i in range(0, n_layers, step):
        if i + step >= n_layers - 1:
            group = layers[i:]
            groups.append(group)
            break
        else:
            group = layers[i:i + step]
            groups.append(group)
    return groups


def get_grouped_llrd_parameters(model,
                                encoder_lr,
                                decoder_lr,
                                embeddings_lr,
                                lr_mult_factor,
                                weight_decay,
                                n_groups):
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    n_layers = model.backbone_config.num_hidden_layers
    parameters_groups = get_parameters_groups(n_layers, n_groups)

    for _, (name, params) in enumerate(named_parameters):

        wd = 0.0 if any(p in name for p in no_decay) else weight_decay

        if name.startswith("backbone.encoder"):
            lr = encoder_lr
            for i, group in enumerate(parameters_groups):
                lr = encoder_lr * (lr_mult_factor ** (i + 1)) if any(p in name for p in group) else lr

            opt_parameters.append({"params": params,
                                   "weight_decay": wd,
                                   "lr": lr})

        if name.startswith("backbone.embeddings"):
            lr = embeddings_lr
            opt_parameters.append({"params": params,
                                   "weight_decay": wd,
                                   "lr": lr})

        if name.startswith("bigram_type_embeddings"):
            lr = embeddings_lr
            opt_parameters.append({"params": params,
                                   "weight_decay": wd,
                                   "lr": lr})

        if name.startswith("fc") or name.startswith('backbone.pooler') or name.startswith('pool') or name.startswith('pooling'):
            lr = decoder_lr
            opt_parameters.append({"params": params,
                                   "weight_decay": wd,
                                   "lr": lr})

    return opt_parameters


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if ("backbone" not in n) and ("backbone_prompt" not in n)],
         'lr': decoder_lr, 'weight_decay': 0.0},
    ]
    return optimizer_parameters


def get_optimizer(model, config):

    if config.optimizer.group_lr_multiplier == 1:
        optimizer_parameters = get_optimizer_params(model,
                                                    config.optimizer.encoder_lr,
                                                    config.optimizer.decoder_lr,
                                                    weight_decay=config.optimizer.weight_decay)
    else:
        optimizer_parameters = get_grouped_llrd_parameters(model,
                                                           encoder_lr=config.optimizer.encoder_lr,
                                                           decoder_lr=config.optimizer.decoder_lr,
                                                           embeddings_lr=config.optimizer.embeddings_lr,
                                                           lr_mult_factor=config.optimizer.group_lr_multiplier,
                                                           weight_decay=config.optimizer.weight_decay,
                                                           n_groups=config.optimizer.n_groups)

    optimizer = AdamW(optimizer_parameters,
                      lr=config.optimizer.encoder_lr,
                      eps=config.optimizer.eps,
                      betas=[config.optimizer.beta1, config.optimizer.beta2])
    return optimizer


# File: src/training/optimizer.py

import json
import pandas as pd


def get_reference_df(raw_df):
    ref_df = raw_df[['document', 'tokens', 'labels']].copy()
    ref_df = ref_df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
    ref_df['token_str'] = ref_df['token'].copy()
    ref_df['token'] = ref_df.groupby('document').cumcount()

    reference_df = ref_df[ref_df['label'] != 'O'].copy()
    reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
    reference_df = reference_df[['row_id', 'document', 'token', 'label', 'token_str']].copy()
    return reference_df


def split_rows(df, max_length, doc_stride):
    new_df = []
    for _, row in df.iterrows():
        tokens = row['tokens']
        if len(tokens) > max_length:
            start = 0
            while start < len(tokens):
                remaining_tokens = len(tokens) - start
                if remaining_tokens < max_length and start != 0:
                    start = max(0, len(tokens) - max_length)
                end = min(start + max_length, len(tokens))
                new_row = {}
                new_row['document'] = row['document']
                new_row['source'] = row['source']
                new_row['valid'] = row['valid']
                new_row['tokens'] = tokens[start:end]
                new_row['trailing_whitespace'] = row['trailing_whitespace'][start:end]
                new_row['labels'] = row['labels'][start:end]
                new_row['token_indices'] = list(range(start, end))
                new_row['full_text'] = rebuild_text(new_row['tokens'], new_row['trailing_whitespace'])
                new_df.append(new_row)
                if remaining_tokens >= max_length:
                    start += doc_stride
                else:
                    break
        else:
            new_row = {
                'document': row['document'],
                'valid': row['valid'],
                'tokens': row['tokens'],
                'trailing_whitespace': row['trailing_whitespace'],
                'labels': row['labels'],
                'token_indices': row['token_indices'],
                'full_text': row['full_text'],
                'source': row['source'],
            }
            new_df.append(new_row)
    return pd.DataFrame(new_df)


def add_token_indices(doc_tokens):
    token_indices = list(range(len(doc_tokens)))
    return token_indices


def rebuild_text(tokens, trailing_whitespace):
    text = ''
    for token, ws in zip(tokens, trailing_whitespace):
        ws = " " if ws == True else ""
        text += token + ws
    return text


# File: src/data/utils.py

import numpy as np
import pandas as pd
from datasets import Dataset
import numpy as np
import torch
from copy import deepcopy


def tokenize(example, tokenizer, max_length, label2id):
    text = []
    token_map = []
    labels = []

    idx = 0
    for t, l, ws in zip(
            example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        token_map.extend([idx]*len(t))
        labels.extend([l] * len(t))
        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)
        idx += 1

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True)
    labels = np.array(labels)
    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue
        if text[start_idx].isspace():
            start_idx += 1
        while start_idx >= len(labels):
            start_idx -= 1
        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)
    return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map,}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length, label2id):
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        sample = {
            'full_text': row.full_text, 'document': row.document,
            'trailing_whitespace': row.trailing_whitespace, 'token_indices': row.token_indices
        }

        tokens = deepcopy(row.tokens)
        labels = row.labels
        if np.random.uniform() > 0.5:
            for i in range(1, len(tokens)):
                if labels[i-1] == 'B-NAME_STUDENT' and labels[i] == 'I-NAME_STUDENT':
                    tokens[i-1], tokens[i] = tokens[i], tokens[i-1]

        sample['tokens'] = tokens
        sample['provided_labels'] = labels
        tokenized = tokenize(sample, self.tokenizer, self.max_length, self.label2id)
        sample.update(tokenized)
        return sample


def create_dataset(data, tokenizer, max_length, label2id):
    ds = Dataset.from_dict({
        "full_text": data.full_text.tolist(),
        "document": data.document.tolist(),
        "tokens": data.tokens.tolist(),
        "trailing_whitespace": data.trailing_whitespace.tolist(),
        "provided_labels": data.labels.tolist(),
        "token_indices": data.token_indices.tolist(),
    })
    ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": max_length}, num_proc=4)
    return ds



# File: src/data/dataset.py

from types import SimpleNamespace


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def namespace_to_dictionary(data):
    dictionary = vars(data)
    for k, v in dictionary.items():
        if type(v) is SimpleNamespace:
            v = namespace_to_dictionary(v)
        dictionary[k] = v
    return dictionary



# File: src/environment/utils.py

from argparse import ArgumentParser
import argparse
from types import SimpleNamespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_default_args():
    args = SimpleNamespace()
    args.debug = False
    args.fold = 0
    args.exp_name = 'test'
    return vars(args)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def get_args():
    its_notebook = is_notebook()
    if its_notebook:
        args = get_default_args()
    else:
        args = get_input_args()

    return args

# File: src/environment/arguments.py

import yaml
import copy
from pathlib import Path


def load_config(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)
    # data = dictionary_to_namespace(data)
    return data


def save_config(config, path):
    config_out = copy.deepcopy(config)
    config_out.tokenizer = None
    config_out = namespace_to_dictionary(config_out)
    for key, value in config_out.items():
        if type(value) == type(Path()):
            config_out[key] = str(value)
    with open(path, 'w') as file:
        yaml.dump(config_out, file, default_flow_style=False)


def concat_configs(
        args,
        config,
        filepaths
):
    config.update(args)
    config.update(filepaths)

    config = dictionary_to_namespace(config)

    if config.debug:
        config.exp_name = 'test'
        config.logger.use_wandb = False

        config.dataset.train_batch_size = 2
        config.dataset.valid_batch_size = 2

    config.run_name = f'{config.exp_name}_{config.job_type}_{config.seed}_{config.fold}' # config.exp_name + f'_fold{config.fold}'
    config.run_id = config.run_name
    return config

# File: src/environment/config.py

import yaml
from pathlib import Path
import os
import shutil


def load_filepaths(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)
    path_to_file = Path(filepath).parents[0]
    for key, value in data.items():
        data[key] = Path(path_to_file / Path(value)).resolve()
    return data


def add_run_specific_filepaths(filepaths, exp_name, job_type, fold, seed):
    filepaths.run_dir = filepaths.models_dir / f'{exp_name}_{job_type}_{seed}_{fold}'

    # filepaths.checkpoint_path = filepaths.run_dir / 'chkp' / f'fold_{fold}_chkp.pth'
    # filepaths.best_model_path = filepaths.run_dir / 'models' / f'fold_{fold}_best.pth'
    # filepaths.log_path = filepaths.run_dir / 'logs' / f'fold-{fold}.log'

    filepaths.config_path = filepaths.run_dir / 'config.yaml'
    filepaths.tokenizer_path = filepaths.run_dir / 'tokenizer'
    filepaths.backbone_config_path = filepaths.run_dir / 'backbone_config.json'
    return filepaths


def create_run_folder(filepath, debug):
    if debug and os.path.isdir(filepath):
        shutil.rmtree(filepath)

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

        logs_dir = filepath / 'logs'
        os.mkdir(logs_dir)

        checkpoints_dir = filepath / 'chkp'
        os.mkdir(checkpoints_dir)

        models_dir = filepath / 'models'
        os.mkdir(models_dir)

    return True




# File: src/environment/filepaths.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from itertools import chain

import random

from types import SimpleNamespace
from pathlib import Path
from functools import partial
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from tokenizers import AddedToken
from datasets import Dataset, concatenate_datasets

from argparse import ArgumentParser
import argparse
from types import SimpleNamespace

import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def replace_labels(data):
    new_data = []
    for sample in data:
        for i in range(len(sample['labels'])):
            if sample['labels'][i] in ['B-INSTRUCTOR_NAME', 'I-INSTRUCTOR_NAME']:
                sample['labels'][i] = sample['labels'][i].replace('INSTRUCTOR_NAME', 'OTHER_NAME')
            if sample['labels'][i] in ['B-ORG_NAME', 'I-ORG_NAME', 'B-COUNTRY_NAME', 'I-COUNTRY_NAME']:
                sample['labels'][i] = 'O'
        new_data.append(sample)
    return new_data



def get_input_args():
    args = SimpleNamespace()
    args.exp_name = 'exp073'
    args.job_type = 'train'
    args.seed = 42
    args.debug = False
    args.pretrain_dataset = 'None'
    args.generated_dataset = 'external_data_v8.json'
    args.prev_exp = 'None'
    args.pretrain_name = 'None'
    args.fold = 0
    return vars(args)


if __name__ == '__main__':

    args = get_input_args()

    args['pseudo_path'] = f'models/{args["pretrain_name"]}'
    if 'exp026' in args["pretrain_name"]:
        args['pseudo_path'] = f'models2/{args["pretrain_name"]}'

    config_fp = 'configs/exp073_config.yaml'
    config = load_config(config_fp)
    filepaths = load_filepaths('configs/filepaths.yaml')
    config = concat_configs(args, config, filepaths)
    config = add_run_specific_filepaths(config, args['exp_name'], args['job_type'], args['fold'], args['seed'])

    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_type)
    # tokenizer.add_tokens(AddedToken("\n", normalized=False))

    label2id = {
        'B-EMAIL': 0, 'B-ID_NUM': 1, 'B-NAME_STUDENT': 2,
        'B-PHONE_NUM': 3, 'B-STREET_ADDRESS': 4, 'B-URL_PERSONAL': 5,
        'B-USERNAME': 6, 'I-ID_NUM': 7, 'I-NAME_STUDENT': 8,
        'I-PHONE_NUM': 9, 'I-STREET_ADDRESS': 10, 'I-URL_PERSONAL': 11,
        'O': 12, # 'B-OTHER_NAME': 13, 'I-OTHER_NAME': 14, #'B-ORG_NAME': 15, 'I-ORG_NAME': 16, 'B-COUNTRY_NAME': 17, 'I-COUNTRY_NAME': 18
    }
    id2label = {v:k for k,v in label2id.items()}

    data = json.load(open("data/processed/train_with_folds.json"))
    # data = json.load(open("data/processed/sample.json"))
    df = pd.DataFrame(data)
    df['fold'] = df['document'] % 4
    df['valid'] = df['fold'] == config.fold
    df['token_indices'] = df['tokens'].apply(add_token_indices)
    df['source'] = 'competition'

    train_folds = df[~df.valid].copy().reset_index(drop=True)
    valid_folds = df[df.valid].copy().reset_index(drop=True)
    print(train_folds.shape, valid_folds.shape)
    print(label2id)

    reference_df = get_reference_df(valid_folds)

    if config.job_type == 'pretrain':
        print('Using pseudo dataset')
        config.training.epochs = 1
        train_data = json.load(open(f'data/external/{config.pretrain_dataset}'))
        train_data = pd.DataFrame(train_data)
        train_data['document'] = -1
        train_data.rename(columns={'labels': 'provided_labels'})
        train_data['token_indices'] = train_data['tokens'].apply(add_token_indices)
        train_data['source'] = 'nbroad'
        train_data['valid'] = False
        train_folds = train_data.copy()

    elif config.job_type == 'fullfit':
        print('Using full dataset')
        train_folds = pd.concat([train_folds, valid_folds])

    if config.generated_dataset != 'None':
        print('Using external dataset')
        external_data = json.load(open(f'data/external/{config.generated_dataset}'))
        external_data = pd.DataFrame(external_data)
        external_data['document'] = -1
        external_data.rename(columns={'labels': 'provided_labels'})
        external_data['token_indices'] = external_data['tokens'].apply(add_token_indices)
        external_data['source'] = 'nbroad'
        train_folds = pd.concat([train_folds, external_data])

    if config.dataset.stride_train:
        train_folds = split_rows(train_folds, config.dataset.doc_max_length, config.dataset.doc_stride)
    if config.dataset.stride_valid:
        valid_folds = split_rows(valid_folds, config.dataset.doc_max_length, config.dataset.doc_stride)

    if config.dataset.filter_no_pii:
        train_folds['pii'] = train_folds['labels'].apply(lambda x: len(set(x)) > 1)
        pii = train_folds[train_folds['pii']].copy()
        no_pii = train_folds[(~train_folds['pii']) & (train_folds['source'] == 'competition')].copy()

        if no_pii.shape[0] > 0:
            no_pii['pii'] = no_pii['document'].apply(lambda x: random.random() < config.dataset.filter_no_pii_ratio)
            no_pii = no_pii[no_pii.pii]
            train_folds = pd.concat([pii, no_pii])
        else:
            train_folds = pii.copy()

        train_folds = train_folds.sort_index()

    if config.job_type == 'debug':
        print('Using debug subset')
        train_folds = train_folds.head(100)
        # valid_folds = valid_folds.head(100)

    train_folds = train_folds.sample(frac=1, random_state=config.seed)

    train_ds = CustomDataset(train_folds, tokenizer, config.dataset.inference_max_length, label2id)
    valid_ds = create_dataset(valid_folds, tokenizer, config.dataset.inference_max_length, label2id)

    print(len(train_ds))
    print(len(valid_ds))

    if config.pretrain_name == 'None':
        model_path = config.model.backbone_type
    else:
        model_path = Path(config.pseudo_path)
    print('State from: ', model_path)

    model = get_model(config, model_path, id2label, label2id)
    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        config.run_dir,
        fp16=config.training.apex,
        learning_rate=config.optimizer.decoder_lr,
        weight_decay=config.optimizer.weight_decay,
        warmup_ratio=config.optimizer.warmup_ratio,
        per_device_train_batch_size=config.dataset.train_batch_size,
        per_device_eval_batch_size=config.dataset.valid_batch_size,
        report_to="none",
        lr_scheduler_type='cosine',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        metric_for_best_model="fbeta_best",
        greater_is_better=True,
        gradient_checkpointing=config.model.gradient_checkpointing,
        num_train_epochs=config.training.epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        dataloader_num_workers=1,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, id2label=id2label, valid_ds=valid_ds, valid_df=reference_df, threshold=config.dataset.fbeta_postproc_thr),
    )
    trainer.train()

# File: src/train_model.py
