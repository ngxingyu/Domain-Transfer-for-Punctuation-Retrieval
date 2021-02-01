#%%
import os
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from typing import List, Optional, Sequence, Union, Callable, Dict, Any, Tuple

__all__ = ['PunctuationDomainModel']

class PunctuationDomainModel(pl.LightningModule):

    @property
    def input_types(self) -> Optional[Dict[str, torch.dtype]]:
        return {"input_ids":torch.long, "attention_mask": torch.bool, "labels": torch.long, "domain": torch.long}
    

    def __init__(self, cfg: DictConfig): 
        # num_labels: int = 10, 
        # embedding_dim: int = 768, 
        # lossfn: str = '', 
        # hidden_dropout_prob:float=0.1, 
        # base_model_path:str='google/electra-base-discriminator', 
        # reduction:str='mean',
        # stride:int=256,
        # unfrozen_layers=0,
        # alpha='0.8',
        # gamma='2',
        # lbd=1, #coefficient of gradient reversal.
        # domains: int = 1):
        super().__init__()

    def setup_datamodule_from_config(self, cfg: DictConfig):
        return PunctuationDataModule(
            labelled=list(cfg.model.dataset.labelled),
            unlabeled=list(cfg.model.dataset.unlabelled),
            train_batch_size=cfg.model.train_ds.batch_size,
            val_batch_size=cfg.model.validation_ds.batch_size,
            max_seq_length=self._cfg.max_seq_length,
            num_workers=cfg.model.dataset.num_workers,
            pin_memory=cfg.model.dataset.pin_memory,
            drop_last=cfg.model.dataset.drop_last,
        )

#%%
'''
        self.num_labels=num_labels
        self.embedding_dim=embedding_dim
        self.domains = domains
        self.reduction=reduction
        self.unfrozen_layers=unfrozen_layers
        self.alpha=alpha
        self.gamma=gamma
        self.lossfn=lossfn
        self.stride=stride
        self.grad_reverse=GradientReverse
        self.grad_reverse.scale=lbd
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.transformer = transformers.ElectraModel.from_pretrained(base_model_path)
        self.freeze()
        self.fcl = torch.nn.Linear(self.embedding_dim, self.num_labels)
        if lossfn == 'crf':
            self.loss=DiceCRF(self.num_labels,reduction=self.reduction)
        elif lossfn == 'dice':
            self.loss=DiceLoss(gamma=self.gamma,alpha=self.alpha, num_classes=self.num_labels, reduction=self.reduction)
        else:
            self.loss=CrossEntropyLoss(reduction=self.reduction)
        if self.domains>1:
            self.domainfcl=torch.nn.Linear(self.embedding_dim, self.domains)
            self.domain_loss=CrossEntropyLoss(reduction=self.reduction, punct_classifier=False)
            self.agg_loss=AggregatorLoss(weights=[1,0.5])
            
        self.punct_class_report = ClassificationReport(
            num_classes=self.num_labels,
            label_ids={'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, ';': 6, '?': 7, '—': 8, '…': 9},
            mode='macro',
            dist_sync_on_step=True,
        )
        if self.domains>1:
            self.domain_class_report = ClassificationReport(
                num_classes=self.domains,
                mode='macro',
                dist_sync_on_step=True)
        
        

    def forward(self, x):
        o1 = self.transformer(x['input_ids'],x['attention_mask'])[0]
        d1 = self.dropout(o1)
        p = self.fcl(d1)
        if self.domains>1: ##relook
            d1r= self.grad_reverse.apply(d1)
            d= self.domainfcl(d1r[:,0,:])
            return p, d
        return p

    def _make_step(self, batch):
        punct_logits, domain_logits = self(batch)
        print('make_step',punct_logits.shape,domain_logits.shape)
        punct_loss = self.loss(punct_logits, batch['labels'], batch['attention_mask'])
        if self.domains>1:
            domain_loss = self.domain_loss(domain_logits, batch['domain'])
        loss = punct_loss if self.domains==1 else self.agg_loss(loss_1=punct_loss, loss_2=domain_loss)
        return loss, punct_logits, domain_logits

    def training_step(self, batch, batch_idx):
        loss,_,_ = self._make_step(batch)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)
        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx):
        val_loss, punct_logits, domain_logits = self._make_step(batch)
        punct_preds = F.one_hot(self.loss.decode(punct_logits, batch['attention_mask']).flatten(),self.num_labels).to(device) if self.lossfn=='crf' else punct_logits.view(-1,self.num_labels)
        punct_labels = F.one_hot(batch['labels'].flatten(),self.num_labels)
        print('punct pred, labels',punct_preds.shape,punct_labels.shape)
        self.punct_class_report.update(punct_preds, punct_labels)
        if self.domains>1:
            domain_labels=F.one_hot(batch['domain'].flatten(),self.domains)
            domain_preds = domain_logits.view(-1,self.domains)
            print('domain pred,label,logits',domain_preds.shape,domain_labels.shape, domain_logits.shape)
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
        return {
            'val_loss': val_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
        }

    def test_step(self, batch, batch_idx):
        test_loss, punct_logits, domain_logits = self._make_step(batch)
        punct_preds = F.one_hot(self.loss.decode(punct_logits, batch['attention_mask']).flatten(),self.num_labels).to(device) if self.loss_fn=='crf' else punct_logits.view(-1,self.num_labels)
        punct_labels = F.one_hot(batch['labels'].flatten(),self.num_labels)
        self.punct_class_report.update(punct_preds, punct_labels)
        if self.domains>1:
            domain_labels=F.one_hot(batch['domain'],self.domains)
            domain_preds = domain_logits.view(-1,self.domains)
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
        return {
                'test_loss': test_loss,
                'punct_tp': self.punct_class_report.tp,
                'punct_fn': self.punct_class_report.fn,
                'punct_fp': self.punct_class_report.fp,
            }
    #https://github.com/NVIDIA/NeMo/blob/bb86f88143c89231f970e5b6bd9f78999fc45a90/nemo/collections/nlp/models/token_classification/punctuation_capitalization_model.py#L42
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        if self.domains>1:
            # calculate metrics and log classification report for Capitalization task
            domain_precision, domain_recall, domain_f1, domain_report = self.domain_class_report.compute()
            logging.info(f'Domain report: {domain_report}')
            self.log('domain_precision', domain_precision)
            self.log('domain_f1', domain_f1)
            self.log('domain_recall', domain_recall)
    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')
        # calculate metrics and log classification report for Capitalization task
        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        if self.domains>1:
            domain_precision, domain_recall, domain_f1, domain_report = self.domain_class_report.compute()
            logging.info(f'Domain report: {domain_report}')
            self.log('domain_precision', domain_precision)
            self.log('domain_f1', domain_f1)
            self.log('domain_recall', domain_recall)
        
    def freeze_transformer_to(self, n:int, exclude_types=(torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) -> None:
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
        self.frozen-=1;

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
'''
