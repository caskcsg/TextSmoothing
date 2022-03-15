# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright huggingface and its affiliates. Licensed under the Apache-2.0 License as part
# of huggingface's transformers package.
# Credit https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification,BertForMaskedLM
#from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead,BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

BERT_MODEL = '/share/wuxing/beifen_gaochaochen/gaochaochen/STS/model/bert-base-uncased'


class Classifier:
    def __init__(self, label_list, device, cache_dir,temp_rate,smooth_rate):
        self._label_list = label_list
        self._device = device

        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,
                                                        do_lower_case=True,
                                                        cache_dir=cache_dir)

        self._model = BertForSequenceClassification.from_pretrained(BERT_MODEL,
                                                                    num_labels=len(label_list),
                                                                    cache_dir=cache_dir)
        self._model.to(device)

        self._optimizer = None

        self.smooth_model = BertForMaskedLM.from_pretrained(BERT_MODEL).to(device)
        self.temp_rate=temp_rate
        self.smooth_rate=smooth_rate
        

        for params in self.smooth_model.parameters():
            params.requires_grad = False
            
        self._dataset = {}
        self._data_loader = {}

    def load_data(self, set_type, examples, batch_size, max_length, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = _make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate, warmup_steps, t_total):
        self._optimizer, self._scheduler = _get_optimizer(
            self._model, learning_rate=learning_rate,
            warmup_steps=warmup_steps, t_total=t_total)

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            self._optimizer.zero_grad()
            outputs = self._model(**inputs)
            loss = outputs[0]  
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

            input_smooth = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }
            input_probs = self.smooth_model(
                **input_smooth
            )

            word_embeddings = self._model.get_input_embeddings().to(self._device)
            one_hot = torch.zeros_like(input_probs[0]).scatter_(2,inputs['input_ids'].reshape(inputs['input_ids'].shape[0],inputs['input_ids'].shape[1],1).long(),1.0).to(self._device)


            now_probs = self.smooth_rate*(torch.nn.functional.softmax(input_probs[0]/self.temp_rate, dim=-1).to(self._device))+(1-self.smooth_rate)*one_hot # 4 2 0.5 0.25
            inputs_embeds_smooth =  now_probs @ word_embeddings.weight
            input_new_smooth={
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'inputs_embeds': inputs_embeds_smooth,
                'labels': batch[3]
                }
            outputs_smooth = self._model(**input_new_smooth)[0]
        
            self._optimizer.zero_grad()
            loss = outputs_smooth  
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]

        for batch in tqdm(data_loader,
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            with torch.no_grad():
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds)
            labels_all.append(inputs["labels"])

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def _get_optimizer(model, learning_rate, warmup_steps, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def _make_data_loader(examples, label_list, tokenizer, batch_size, max_length, shuffle):
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=max_length,
                                            output_mode="classification")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
