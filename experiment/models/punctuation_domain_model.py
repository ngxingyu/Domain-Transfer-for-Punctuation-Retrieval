#%%
import os
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from typing import List, Optional, Sequence, Union, Callable, Dict, Any, Tuple
from nemo.core.neural_types import LogitsType, NeuralType
from core import ClassificationReport
from core.layers import *
from transformers import AutoModel
import logging
# from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
# from nemo.collections.nlp.parts.utils_funcs import tensor2list

__all__ = ['PunctuationDomainModel']


class PunctuationDomainModel(pl.LightningModule):

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "subtoken_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
            "domain": NeuralType(('B'), ChannelType()),
        }
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "domain_logits": NeuralType(('B', 'D'), LogitsType()),
        }

    def __init__(self, 
    # transformer: str, 
    # punct_lossfn: str = 'cel', 
    # domain_lossfn: str = 'cel', 
    # reduction: str = 'mean',
    # unfrozen_layers: int = 0,
    # max_seq_length: int = 128,
    # #dice loss
    # alpha: int = 0.8,
    # gamma: int = 2,
    # #gradient reversal
    # reversal_grad: int = 1,
    # labels_to_ids:Dict[str,int]={'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, ';': 6, '?': 7, '—': 8, '…': 9},
    # num_domains: int = 1,
    cfg: DictConfig
    ): 
        
        super().__init__()
        self.save_hyperparameters(cfg)
        
        self.transformer = AutoModel.from_pretrained(cfg.transformer_path)
        self.labels_to_ids = {_[1]:_[0] for _ in enumerate(self.hparams.punct_label_ids)}
        self.ids_to_labels = {_[0]:_[1] for _ in enumerate(self.hparams.punct_label_ids)}
        self.hparams.num_domains=len(self.hparams.dataset.labelled)+len(self.hparams.dataset.unlabelled)

        self.punct_classifier = TokenClassifier(
            hidden_size=self.transformer.config.hidden_size,
            num_classes=len(self.labels_to_ids),
            activation=self.hparams.punct_head.activation,
            log_softmax=False,
            dropout=self.hparams.punct_head.fc_dropout,
            num_layers=self.hparams.punct_head.punct_num_fc_layers,
            use_transformer_init=self.hparams.punct_head.use_transformer_init,
        )

        self.domain_classifier = SequenceClassifier(
            hidden_size=self.transformer.config.hidden_size,
            num_classes=self.hparams.num_domains,
            num_layers=self.hparams.domain_head.domain_num_fc_layers,
            activation=self.hparams.domain_head.activation,
            log_softmax=self.hparams.domain_head.log_softmax,
            dropout=cfg.domain_head.fc_dropout,
            use_transformer_init=cfg.domain_head.use_transformer_init,
        )

        switch self.hparams.punct_head.loss:
        if not self.hparams.punct_head.loss in ['cel']:
            self.hparams.punct_head.loss='cel'
        self.punctuation_loss = {'cel':CrossEntropyLoss(logits_ndim=3)}[self.hparams.punct_head.loss]

        if not self.hparams.domain_head.loss in ['cel']:
            self.hparams.domain_head.loss='cel'
        self.domain_loss = {'cel':CrossEntropyLoss(logits_ndim=2)}[self.hparams.domain_head.loss]

        self.agg_loss = AggregatorLoss(num_inputs=2)

        self.punct_class_report = ClassificationReport(
            num_classes=len(self.hparams.punct_label_ids),
            label_ids=self.labels_to_ids,
            mode='macro',
            dist_sync_on_step=True,
        )
        self.domain_class_report = ClassificationReport(
            num_classes=self.hparams.num_domains,
            label_ids=list(range(self.hparams.num_domains)),
            mode='macro',
            dist_sync_on_step=True,
        )

        self.grad_reverse= GradientReverse
        self.grad_reverse.scale=self.hparams.domain_head.lbd

    def forward(self, input_ids, attention_mask, token_type_ids=None, domain_ids=None):
        hidden_states = self.transformer(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        reverse_grad_hidden_states = self.grad_reverse.apply(hidden_states)
        domain_logits = self.domain_classifier(hidden_states=reverse_grad_hidden_states)
        return punct_logits, domain_logits

    def _make_step(self, batch):
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        subtoken_mask=batch['subtoken_mask']
        punct_labels=batch['labels']
        domain_labels=batch['domain']
        punct_logits, domain_logits = self(
            input_ids=input_ids, attention_mask=attention_mask
        )

        punct_loss = self.punct_loss(logits=punct_logits, labels=punct_labels, loss_mask=subtoken_mask)
        domain_loss = self.domain_loss(logits=domain_logits, labels=domain_labels)
        loss = self.agg_loss(loss_1=punct_loss, loss_2=domain_loss)
        return loss, punct_logits, domain_logits
    

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        loss, _, _ = self._make_step(batch)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)

        return {'loss': loss, 'lr': lr}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        punct_labels=batch['labels']
        domain_labels=batch['domain'][:,0,:]

        val_loss, punct_logits, domain_logits = self._make_step(batch)

        # attention_mask = attention_mask > 0.5
        punct_preds = torch.argmax(punct_logits, axis=-1)[attention_mask]
        punct_labels = punct_labels[attention_mask]
        self.punct_class_report.update(punct_preds, punct_labels)

        domain_preds = torch.argmax(domain_logits, axis=-1)[attention_mask]
        domain_labels = domain_labels[attention_mask]
        self.domain_class_report.update(domain_preds, domain_labels)

        return {
            'val_loss': val_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'domain_tp': self.domain_class_report.tp,
            'domain_fn': self.domain_class_report.fn,
            'domain_fp': self.domain_class_report.fp,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        punct_labels=batch['labels']
        domain_labels=batch['domain'][:,0,:]

        test_loss, punct_logits, domain_logits = self._make_step(batch)

        # attention_mask = attention_mask > 0.5
        punct_preds = torch.argmax(punct_logits, axis=-1)[attention_mask]
        punct_labels = punct_labels[attention_mask]
        self.punct_class_report.update(punct_preds, punct_labels)

        domain_preds = torch.argmax(domain_logits, axis=-1)[attention_mask]
        domain_labels = domain_labels[attention_mask]
        self.domain_class_report.update(domain_preds, domain_labels)

        return {
            'test_loss': test_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'domain_tp': self.domain_class_report.tp,
            'domain_fn': self.domain_class_report.fn,
            'domain_fp': self.domain_class_report.fp,
        }
    
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        # calculate metrics and log classification report for domainalization task
        domain_precision, domain_recall, domain_f1, domain_report = self.domain_class_report.compute()
        logging.info(f'Domain report: {domain_report}')

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('domain_precision', domain_precision)
        self.log('domain_f1', domain_f1)
        self.log('domain_recall', domain_recall)

    def test_epoch_end(self, outputs):
        """
            Called at the end of test to aggregate outputs.
            outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        # calculate metrics and log classification report for domainalization task
        domain_precision, domain_recall, domain_f1, domain_report = self.domain_class_report.compute()
        logging.info(f'Domain report: {domain_report}')

        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('domain_precision', domain_precision)
        self.log('domain_f1', domain_f1)
        self.log('domain_recall', domain_recall)
    
    def freeze_transformer_to(self, n:int, exclude_types=(torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) -> None:
        """Freeze layers up to layer group `n`.
        Look at each group, and freeze each paraemeter, except excluded types
        """
        print(f"freeze 1st {n} encoder layers of transformer")
        def set_requires_grad_for_module(module: torch.nn.Module, requires_grad: bool):
            "Sets each parameter in lthe module to the `requires_grad` value"
            params = list(module.parameters())
            for param in params: 
                param.requires_grad = requires_grad
            
        for layer in list(self.transformer.encoder.layer)[:n]:
            if not isinstance(layer, exclude_types): 
                set_requires_grad_for_module(layer, False)
        
        for layer in list(self.transformer.encoder.layer)[n:]:
            set_requires_grad_for_module(layer, True)

    def freeze(self) -> None:
        for param in self.transformer.embeddings.parameters():
            param.requires_grad=False

        self.frozen=len(self.transformer.encoder.layer)
        self.freeze_transformer_to(self.frozen)

    def unfreeze(self,i:int=1):
        self.freeze_transformer_to(max(0,self.frozen-i))
        self.frozen-=1

