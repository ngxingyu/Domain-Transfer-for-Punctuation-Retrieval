#%%
import os
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from typing import List, Optional, Sequence, Union, Callable, Dict, Any, Tuple
from nemo.core.neural_types import LogitsType, NeuralType
from core import ClassificationReport
from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.nlp.parts.utils_funcs import tensor2list

__all__ = ['PunctuationDomainModel']

class PunctuationDomainModel(pl.LightningModule):

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        self.bert_model.input_types
        # return {"input_ids":torch.long, "attention_mask": torch.bool, "labels": torch.long, "domain": torch.long}
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "domain_logits": NeuralType(('B', 'D'), LogitsType()),
        }

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
        # self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)

        self._cfg.punct_label_ids=OmegaConf.create(sorted(self._cfg.punct_label_ids))
        self.labels_to_ids = {_[0]:_[1] for _ in enumerate(self._cfg.punct_label_ids)}
        self.ids_to_labels = {_[1]:_[0] for _ in enumerate(self._cfg.punct_label_ids)}
        self.num_domains = len(self._cfg.dataset.labelled)+len(self._cfg.dataset.unlabelled)
        
        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.punct_classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=len(self._cfg.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=False,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=cfg.punct_head.punct_num_fc_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.domain_classifier = SequenceClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=self.num_domains,
            num_layers=cfg.domain_head.domain_num_fc_layers,
            activation=cfg.domain_head.activation,
            log_softmax=False,
            dropout=cfg.domain_head.fc_dropout,
            use_transformer_init=cfg.domain_head.use_transformer_init,
        )

        self.punctuation_loss = CrossEntropyLoss(logits_ndim=3)
        self.domain_loss = CrossEntropyLoss(logits_ndim=2)
        self.agg_loss = AggregatorLoss(num_inputs=2)

        self.punct_class_report = ClassificationReport(
            num_classes=len(self._cfg.punct_label_ids),
            label_ids=self.labels_to_ids,
            mode='macro',
            dist_sync_on_step=True,
        )
        self.domain_class_report = ClassificationReport(
            num_classes=self.num_domains,
            label_ids=list(range(self.num_domains)),
            mode='macro',
            dist_sync_on_step=True,
        )


    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids=None, domain_ids=None):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        reverse_grad_hidden_states = self.grad_reverse.apply(hidden_states)
        domain_logits = self.domain_classifier(hidden_states=reverse_grad_hidden_states)
        return punct_logits, domain_logits

    def _make_step(self, batch):
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        punct_labels=batch['labels']
        domain_labels=batch['domain'][:,0,:]
        # input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, punct_labels, domain_labels = batch
        punct_logits, domain_logits = self(
            input_ids=input_ids, attention_mask=attention_mask
        )

        punct_loss = self.punct_loss(logits=punct_logits, labels=punct_labels, loss_mask=attention_mask)
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
    
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
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

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
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
    
    def update_data_dir(self, data_dir: str) -> None:
        """
        Update data directory
        Args:
            data_dir: path to data directory
        """
        if os.path.exists(data_dir):
            logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
            self._cfg.dataset.data_dir = data_dir
        else:
            raise ValueError(f'{data_dir} not found')

    def setup_datamodule(self, cfg: Optional[DictConfig] = None):
        if cfg is None:
            cfg = self._cfg.train_ds
            
        self.data_module = PunctuationDataModule(
            labelled=list(cfg.model.dataset.labelled),
            unlabeled=list(cfg.model.dataset.unlabelled),
            train_batch_size=cfg.model.train_ds.batch_size,
            val_batch_size=cfg.model.validation_ds.batch_size,
            max_seq_length=self._cfg.max_seq_length,
            num_workers=cfg.model.dataset.num_workers,
            pin_memory=cfg.model.dataset.pin_memory,
            drop_last=cfg.model.dataset.drop_last,
            tokenizer=self.tokenizer,
        )
        self._train_dl=self.data_module.train_dataloader
        self._validation_dl=self.data_module.dev_dataloader
        self._test_dl=self.data_module.test_dataloader
    
    def _setup_infer_dataloader(self, queries: List[str], batch_size: int) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            queries: lower cased text without punctuation
            batch_size: batch size to use during inference
        Returns:
            A pytorch DataLoader.
        """

        dataset = BertPunctuationInferDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=self._cfg.dataset.max_seq_length
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=False,
        )

    def add_punctuation_capitalization(self, queries: List[str], batch_size: int = None) -> List[str]:
        """
        Adds punctuation and capitalization to the queries. Use this method for debugging and prototyping.
        Args:
            queries: Text
            batch_size: batch size to use during inference
        Returns:
            result: text with punctuation
        """

        if queries is None or len(queries) == 0:
            return []

        if batch_size is None:
            batch_size = len(queries)
            logging.info(f'Using batch size {batch_size} for inference')

        # We will store the output here
        result = []

        # Model's mode and device
        mode = self.training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Switch model to evaluation mode
            self.eval()
            self = self.to(device)
            infer_datalayer = self._setup_infer_dataloader(queries, batch_size)

            # store predictions for all queries in a single list
            all_punct_preds = []

            for batch in infer_datalayer:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                punct_logits, _ = self.forward(
                    input_ids=input_ids.to(device),
                    attention_mask=input_mask.to(device),
                )
                punct_preds = tensor2list(torch.argmax(punct_logits, axis=-1)[subtokens_mask])
                all_punct_preds.extend(punct_preds)
            id2tag = {v: k for k, v in self._cfg.punct_label_ids.items()}
            result.extend([' '.join([_[0]+_[1] for _ in \
                list(zip(self.tokenizer.convert_ids_to_tokens(_[0]),
                            [id2tag[id] for id in _[1].tolist()])
                    )]) for _ in zip(infer_datalayer['input_ids'],all_punct_preds)])
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result


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
    #https://github.com/NVIDIA/NeMo/blob/bb86f88143c89231f970e5b6bd9f78999fc45a90/nemo/collections/nlp/models/token_classification/punctuation_domainalization_model.py#L42
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
            # calculate metrics and log classification report for domainalization task
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
        # calculate metrics and log classification report for domainalization task
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
