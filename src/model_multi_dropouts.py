from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from transformers import DebertaV2Config, DebertaV2ForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel,
    DebertaModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)


## Pooling Strategies
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out


# v2 vor latest
class CustomModel(DebertaV2PreTrainedModel):  # nn.Module):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/deberta/modeling_deberta.py#L1342
    def __init__(self, config):
        # super(CustomModel, self).__init__(config)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        print(f"Num Labels {config.num_labels}")
        self.mean_pooling = MeanPooling()
        # self.max_pooler = MaxPooling()
        # self.min_pooler = MinPooling()
        self.bilstm_layer = True
        self.mult_sample_dpt = False
        self.mean_pool = False
        # Loss Fn
        o_weight = 0.05
        self.class_weights = torch.tensor([1.0] * 12 + [o_weight])
        self.loss_fct = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=-100
        )  # weight=self.class_weights)

        if self.bilstm_layer:
            print(
                f"Including LSTM layer hidden size {self.config.hidden_size} dropout {self.config.hidden_dropout_prob}"
            )
            self.bilstm = nn.LSTM(
                config.hidden_size,
                (config.hidden_size) // 2,
                num_layers=2,
                dropout=config.hidden_dropout_prob,
                batch_first=True,
                bidirectional=True,
            )
            self.initialize_lstm(self.bilstm)
            # n_layers can be tried as 2 as well .
            # self.head = LSTMHead(in_features=config.hidden_size, hidden_dim=(config.hidden_size) // 2, n_layers=1)
            # self.initialize_lstm(self.lstm)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # self._init_weights
        self.post_init()

    def initialize_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

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
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output_backbone = outputs[0]
        # output_backbone=self.deberta(input_ids,attention_mask = attention_mask)
        if self.bilstm_layer:
            # output_backbone = self.dropout(output_backbone)
            # print("output_backbone ",output_backbone)
            self.bilstm.flatten_parameters()
            output, hc = self.bilstm(output_backbone)
            # output = self.head(output_backbone)

        if self.mean_pool:
            output_backbone = self.dropout(output_backbone)
            output = self.mean_pooling(output_backbone, attention_mask)
        # max_pool = self.max_pooler(output, attention_mask)
        # min_pool = self.min_pooler(output, attention_mask)
        # concat = torch.cat([mean_pool], dim=1)

        # Multi-sample dropout.
        if self.mult_sample_dpt:
            output1 = self.classifier(self.dropout1(output_backbone))
            output2 = self.classifier(self.dropout2(output_backbone))
            output3 = self.classifier(self.dropout3(output_backbone))
            output4 = self.classifier(self.dropout4(output_backbone))
            output5 = self.classifier(self.dropout5(output_backbone))
            logits = (output1 + output2 + output3 + output4 + output5) / 5
        else:
            # print("output",output)
            logits = self.classifier(output)

        loss = None

        if labels is not None:
            if self.mean_pool:
                loss = self.loss_fct(logits, labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
