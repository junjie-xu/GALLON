import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
            
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)        

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3, reduction='mean')

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)




class BertRegressor(PreTrainedModel):
    def __init__(self, model, dropout=0.0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.MSELoss(reduction='mean')
        # self.loss_func = nn.L1Loss()

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        
        # Assuming a single continuous output for regression
        self.regressor = nn.Linear(hidden_dim, 1, bias=cla_bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=None):
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True,
                                    output_hidden_states=True)
        emb = self.dropout(outputs.hidden_states[-1])
        cls_token_emb = emb[:, 0, :]  # Assuming using the CLS token for regression
        
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        
        predictions = self.regressor(cls_token_emb).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = self.loss_func(predictions, labels)

        return TokenClassifierOutput(loss=loss, logits=predictions)



class BertRegInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_regressor = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.MSELoss(reduction='mean')
        # self.loss_func = nn.L1Loss()

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):
        
        # Extract outputs from the model
        bert_outputs = self.bert_regressor.bert_encoder(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        return_dict=True,
                                                        output_hidden_states=True)      
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb[:, 0, :]  # Assuming using the CLS token for regression
        
        if self.feat_shrink:
            cls_token_emb = self.bert_regressor.feat_shrink_layer(cls_token_emb)
        
        predictions = self.bert_regressor.regressor(cls_token_emb).squeeze(-1)  # Adjust shape for regression outputs
        
        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = predictions.cpu().numpy().astype(np.float16)
        
        loss = None
        if labels is not None:
            loss = self.loss_func(predictions, labels)
        
        return TokenClassifierOutput(loss=loss, logits=predictions)
