# %%
import copy
import math
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tarfile
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from core import ClassificationReport
from core.layers import *
from core.losses import (AggregatorLoss, CrossEntropyLoss, FocalDiceLoss, FocalLoss,
                         LinearChainCRF)
from pytorch_lightning.utilities import rank_zero_only
from core.optim import get_optimizer, parse_optimizer_args, prepare_lr_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoModel, AutoTokenizer
import torch.utils.data.dataloader as dataloader
from data import PunctuationDataModule, PunctuationInferenceDataset
from os import path
import tempfile
from core.common import Serialization, FileIO
from time import time
from core.utils import view_aligned

# from nemo.core.neural_types import LogitsType, NeuralType
# from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
# from nemo.collections.nlp.parts.utils_funcs import tensor2list

__all__ = ['PunctuationDomainModel']

_MODEL_CONFIG_YAML = "model_config.yaml"
_MODEL_WEIGHTS = "model_weights.ckpt"

class PunctuationDomainModel(pl.LightningModule, Serialization, FileIO):

    def __init__(self,
                 cfg: DictConfig,
                 trainer: pl.Trainer = None,
                 data_id: str = '',
                 log_dir: str = '',
                 ):
        if trainer is not None and not isinstance(trainer, pl.Trainer):
            raise ValueError(
                f"trainer constructor argument must be either None or pytroch_lightning.Trainer. But got {type(trainer)} instead."
            )
        super().__init__()
        self._cfg = cfg
        self._cfg.log_dir=log_dir
        self.save_hyperparameters(cfg)
        self._optimizer = None
        self._scheduler = None
        self._trainer = trainer

        self.transformer = AutoModel.from_pretrained(self.hparams.model.transformer_path)
        self.tokenizer=AutoTokenizer.from_pretrained(self._cfg.model.transformer_path)
        if self._cfg.model.no_space_label is not None:
            s=set(self.hparams.model.punct_label_ids)
            s.add(self._cfg.model.no_space_label)
            self.hparams.model.punct_label_ids=sorted(list(s))
        self.ids_to_labels = {_[0]: _[1]
                              for _ in enumerate(self.hparams.model.punct_label_ids)}
        self.labels_to_ids = {v:k
                              for k,v in self.ids_to_labels.items()}
        self.label_map={k:v for k,v in self._cfg.model.label_map.items()}
        self.data_id=data_id
        assert(len(self._cfg.model.dataset.labelled)>0,'Please include at least 1 labelled dataset')
        self.setup_datamodule()

        if (self.hparams.model.punct_class_weights==True and self.hparams.model.punct_head.loss!='crf'):
            self.hparams.model.punct_class_weights=OmegaConf.create(self.dm.train_dataset.determine_class_weights().tolist())
        else:
            self.hparams.model.punct_class_weights=None

        self.punct_classifier = TokenClassifier(
            hidden_size=self.transformer.config.hidden_size,
            num_classes=len(self.labels_to_ids),
            activation=self.hparams.model.punct_head.activation,
            log_softmax=self.hparams.model.punct_head.log_softmax,
            dropout=self.hparams.model.punct_head.fc_dropout,
            num_layers=self.hparams.model.punct_head.punct_num_fc_layers,
            use_transformer_init=self.hparams.model.punct_head.use_transformer_init,
        )

        self.domain_classifier = SequenceClassifier(
            hidden_size=self.transformer.config.hidden_size,
            num_classes=self.hparams.model.dataset.num_domains,
            num_layers=self.hparams.model.domain_head.domain_num_fc_layers,
            activation=self.hparams.model.domain_head.activation,
            log_softmax=self.hparams.model.domain_head.log_softmax,
            dropout=self.hparams.model.domain_head.fc_dropout,
            use_transformer_init=self.hparams.model.domain_head.use_transformer_init,
            pooling=self.hparams.model.domain_head.pooling,
            idx_conditioned_on = self.hparams.model.domain_head.idx_conditioned_on,
        )

        if not self.hparams.model.punct_head.loss in ['cel', 'dice', 'crf', 'focal']:
            self.log('punct_head loss not found, fallback to cross entropy loss')
            self.hparams.model.punct_head.loss = 'cel'
        if self.hparams.model.punct_head.loss == 'dice':
            self.punctuation_loss = FocalDiceLoss(**self.hparams.model.dice_loss, weight=self.hparams.model.punct_class_weights, num_labels=self.hparams.model.dataset.num_labels)
        elif self.hparams.model.punct_head.loss == 'crf':
            self.punctuation_loss = LinearChainCRF(self.hparams.model.dataset.num_labels)
        elif self.hparams.model.punct_head.loss == 'focal':
            self.punctuation_loss = FocalLoss(**self.hparams.model.focal_loss, weight=self.hparams.model.punct_class_weights)
        else: 
            self.punctuation_loss = CrossEntropyLoss(logits_ndim=3, weight=self.hparams.model.punct_class_weights)

        if self.hparams.model.punct_head.bilstm:
            self.bilstm = torch.nn.LSTM(bidirectional=True, num_layers=2, input_size=self.transformer.config.hidden_size, hidden_size=self.transformer.config.hidden_size//2, batch_first=True)             
        if not self.hparams.model.domain_head.loss in ['cel']:
            self.log('domain_head loss not found, fallback to cross entropy loss')
            self.hparams.model.domain_head.loss = 'cel'
        # self.hparams.model.domain_head.loss
        self.domain_loss = CrossEntropyLoss(logits_ndim=2)

        self.agg_loss = AggregatorLoss(num_inputs=2)

        self.punct_class_report = ClassificationReport(
            num_classes=self.hparams.model.dataset.num_labels,
            label_ids=self.labels_to_ids,
            mode='macro',
            dist_sync_on_step=True,
        )

        self.chunked_punct_class_report = ClassificationReport(
            num_classes=self.hparams.model.dataset.num_labels,
            label_ids=self.labels_to_ids,
            mode='macro',
            dist_sync_on_step=True,
        )


        self.domain_class_report = ClassificationReport(
            num_classes=self.hparams.model.dataset.num_domains,
            label_ids={v:v for v in list(range(self.hparams.model.dataset.num_domains))},
            mode='macro',
            dist_sync_on_step=True,
        )

        self.grad_reverse = GradientReverse
        self.grad_reverse.scale = 0 #self.hparams.model.domain_head.gamma
        self.freeze()

    def forward(self, input_ids, attention_mask, subtoken_mask=None, domain_ids=None):
        hidden_states = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        if self.hparams.model.punct_head.bilstm:
            hidden_states,_=self.bilstm(hidden_states)
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        reverse_grad_hidden_states = self.grad_reverse.apply(hidden_states)
        assert not torch.isnan(input_ids).any(), (input_ids,'inputid')
        assert not torch.isnan(attention_mask).any(), ('amask',attention_mask)
        if torch.isnan(hidden_states).any():
            logging.error(hidden_states,attention_mask.sum(1),'hiddenstate')
        domain_logits = self.domain_classifier(
            hidden_states=reverse_grad_hidden_states,
            attention_mask=attention_mask)
        # print(attention_mask.sum(axis=1),domain_logits)
        return punct_logits, domain_logits

    def _make_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        subtoken_mask = batch['subtoken_mask']
        punct_labels = batch['labels']
        domain_labels = batch['domain']
        punct_logits, domain_logits = self(
            input_ids=input_ids, attention_mask=attention_mask, subtoken_mask=subtoken_mask,
        )
        punctuation_loss = self.punctuation_loss(
            logits=punct_logits[subtoken_mask[:,0]>0], labels=punct_labels[subtoken_mask[:,0]>0], loss_mask=subtoken_mask[subtoken_mask[:,0]>0])
        domain_loss = self.domain_loss(
            logits=domain_logits, labels=domain_labels)
        loss = self.agg_loss(loss_1=punctuation_loss, loss_2=domain_loss)
        return loss, punct_logits, domain_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        p=(self.current_epoch*self.train_size+batch_idx)/(self.train_size*self.hparams.trainer.max_epochs)
        if (batch_idx%1000==0):
            print('gamma:',p)
        self.grad_reverse.scale=(2/(1+math.exp(-10*p))-1)*self.hparams.model.domain_head.gamma
        loss, _, _ = self._make_step(batch)
        lr = self._optimizer.param_groups[0]['lr']


        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)
        self.log('gamma', self.grad_reverse.scale,logger=True)

        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        subtoken_mask = batch['subtoken_mask']
        punct_labels = batch['labels']
        domain_labels = batch['domain']
        labelled_mask=subtoken_mask[:,0]>0

        val_loss, punct_logits, domain_logits = self._make_step(batch)
        # attention_mask = attention_mask > 0.5
        punct_preds = self.punctuation_loss.decode(punct_logits[labelled_mask], subtoken_mask[labelled_mask]) \
            if self.hparams.model.punct_head.loss == 'crf' else torch.argmax(punct_logits[labelled_mask], axis=-1)[subtoken_mask[labelled_mask]]
        # pp(punct_preds.device)
        punct_labels = punct_labels[labelled_mask][subtoken_mask[labelled_mask]]
        self.punct_class_report.update(punct_preds, punct_labels)
        domain_preds = torch.argmax(domain_logits, axis=-1)
        domain_labels = domain_labels.view(-1)
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
        attention_mask = batch['attention_mask']
        subtoken_mask = batch['subtoken_mask']
        punct_labels = batch['labels']
        domain_labels = batch['domain']
        labelled_mask=subtoken_mask[:,0]>0
        if self.hparams.model.test_chunk_percent:
            chunk=self.hparams.model.test_chunk_percent
        else:
            chunk=0.5
        chunk_mask=torch.zeros_like(subtoken_mask)
        chunk_mask[:,torch.arange(int((0.5-chunk/2)*subtoken_mask.shape[-1]),int((0.5+chunk/2)*subtoken_mask.shape[-1]))]=1
        chunk_mask=chunk_mask[labelled_mask][subtoken_mask[labelled_mask]]

        test_loss, punct_logits, domain_logits = self._make_step(batch)
        # attention_mask = attention_mask > 0.5
        punct_preds = self.punctuation_loss.decode(punct_logits[labelled_mask], subtoken_mask[labelled_mask]) \
            if self.hparams.model.punct_head.loss == 'crf' else torch.argmax(punct_logits[labelled_mask], axis=-1)[subtoken_mask[labelled_mask]]
        chunked_punct_preds = punct_preds[chunk_mask]

        
        punct_labels = punct_labels[labelled_mask][subtoken_mask[labelled_mask]]
        chunked_punct_labels = punct_labels[chunk_mask]


        self.punct_class_report.update(punct_preds, punct_labels)
        self.chunked_punct_class_report.update(chunked_punct_preds, chunked_punct_labels)
        domain_preds = torch.argmax(domain_logits, axis=-1)
        domain_labels = domain_labels.view(-1)
        self.domain_class_report.update(domain_preds, domain_labels)

        return {
            'test_loss': test_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'chunked_punct_tp': self.chunked_punct_class_report.tp,
            'chunked_punct_fn': self.chunked_punct_class_report.fn,
            'chunked_punct_fp': self.chunked_punct_class_report.fp,
            'domain_tp': self.domain_class_report.tp,
            'domain_fn': self.domain_class_report.fn,
            'domain_fp': self.domain_class_report.fp,
        }

    def validation_epoch_end(self, outputs):
        
        self.dm.train_dataset.shuffle()
        if outputs is not None and len(outputs) == 0:
            return {}
        if type(outputs[0]) == dict:
            output_dict = self.multi_validation_epoch_end(outputs)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            return output_dict
        

    

    def multi_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report, punctuation_cm = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        # calculate metrics and log classification report for domainalization task
        domain_precision, domain_recall, domain_f1, domain_report, domain_cm = self.domain_class_report.compute()
        logging.info(f'Domain report: {domain_report}')

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('domain_precision', domain_precision)
        self.log('domain_f1', domain_f1)
        self.log('domain_recall', domain_recall)
        # self.log('punctuation_cm', punctuation_cm)
        # self.log('domain_cm', domain_cm)

    def test_epoch_end(self, outputs):
        if outputs is not None and len(outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if type(outputs[0]) == dict:
            output_dict = self.multi_test_epoch_end(outputs)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            return output_dict

    def multi_test_epoch_end(self, outputs):
        """
            Called at the end of test to aggregate outputs.
            outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report, punct_cm = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        chunked_punct_precision, chunked_punct_recall, chunked_punct_f1, chunked_punct_report, chunked_punct_cm = self.chunked_punct_class_report.compute()
        logging.info(f'Chunked Punctuation report: {chunked_punct_report}')

        # calculate metrics and log classification report for domainalization task
        domain_precision, domain_recall, domain_f1, domain_report, domain_cm = self.domain_class_report.compute()
        logging.info(f'Domain report: {domain_report}')

        path=f"{self.hparams.log_dir}/test.txt" if self.hparams.log_dir!='' else f'{self.hparams.exp_manager.exp_dir}{self.hparams.exp_manager.name}'
        logging.info(f'saving to {path}')
        with open(path,'w') as f:
            f.write("Punct report\n")
            f.write(punct_report)
            f.write("\nChunked Punct report\n")
            f.write(chunked_punct_report)
            f.write("\nDomain report\n")
            f.write(domain_report)
            f.write('\n\n')
            f.write(f'test_loss: {avg_loss}\n')
            f.write(f'punct_precision: {punct_precision}\n')
            f.write(f'punct_f1: {punct_f1}\n')
            f.write(f'punct_recall: {punct_recall}\n')
            f.write(f'chunked_punct_precision: {chunked_punct_precision}\n')
            f.write(f'chunked_punct_f1: {chunked_punct_f1}\n')
            f.write(f'chunked_punct_recall: {chunked_punct_recall}\n')
            f.write(f'domain_precision: {domain_precision}\n')
            f.write(f'domain_f1: {domain_f1}\n')
            f.write(f'domain_recall: {domain_recall}\n')

        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('chunked_punct_precision', chunked_punct_precision)
        self.log('chunked_punct_f1', chunked_punct_f1)
        self.log('chunked_punct_recall', chunked_punct_recall)
        self.log('domain_precision', domain_precision)
        self.log('domain_f1', domain_f1)
        self.log('domain_recall', domain_recall)

        # self.log('punctuation_cm', punct_cm)
        # self.log('domain_cm', domain_cm)

    def setup_optimization(self, optim_config: Optional[Union[DictConfig, Dict]] = None):
        """
        Prepares an optimizer from a string name and its optional config parameters.
        Args:
            optim_config: A dictionary containing the following keys:
                * "lr": mandatory key for learning rate. Will raise ValueError if not provided.
                * "optimizer": string name pointing to one of the available optimizers in the registry. \
                If not provided, defaults to "adam".
                * "opt_args": Optional list of strings, in the format "arg_name=arg_value". \
                The list of "arg_value" will be parsed and a dictionary of optimizer kwargs \
                will be built and supplied to instantiate the optimizer.
        """
        # If config was not explicitly passed to us
        if optim_config is None:
            # See if internal config has `optim` namespace
            if self._cfg.model is not None and hasattr(self._cfg.model, 'optim'):
                optim_config = self._cfg.model.optim

        # If config is still None, or internal config has no Optim, return without instantiation
        if optim_config is None:
            logging.info(
                'No optimizer config provided, therefore no optimizer was created')
            return

        else:
            # Preserve the configuration
            if not isinstance(optim_config, DictConfig):
                optim_config = OmegaConf.create(optim_config)

            # See if internal config has `optim` namespace before preservation
            if self._cfg.model is not None and hasattr(self._cfg.model, 'optim'):
                if self._cfg.model.optim is None:
                    self._cfg.model.optim = copy.deepcopy(optim_config)
                else:
                    with open_dict(self._cfg.model.optim):
                        self._cfg.model.optim = copy.deepcopy(optim_config)

        # Setup optimizer and scheduler
        if optim_config is not None and isinstance(optim_config, DictConfig):
            optim_config = OmegaConf.to_container(optim_config, resolve=True)

        if 'sched' in optim_config and self._trainer is not None:
            if not isinstance(self._trainer.accumulate_grad_batches, int):
                raise ValueError(
                    "We do not currently support gradient acculumation that is not an integer.")
            if self._trainer.max_steps is None:
                # Store information needed to calculate max_steps
                optim_config['sched']['t_max_epochs'] = self._trainer.max_epochs
                optim_config['sched']['t_accumulate_grad_batches'] = self._trainer.accumulate_grad_batches
                optim_config['sched']['t_limit_train_batches'] = self._trainer.limit_train_batches
                if self._trainer.distributed_backend is None:
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus or 1
                elif self._trainer.distributed_backend == "ddp_cpu":
                    optim_config['sched']['t_num_workers'] = self._trainer.num_processes * \
                        self._trainer.num_nodes
                elif self._trainer.distributed_backend == "ddp":
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * \
                        self._trainer.num_nodes
                else:
                    logging.warning(
                        f"The lightning trainer received accelerator: {self._trainer.distributed_backend}. We "
                        "recommend to use 'ddp' instead."
                    )
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * \
                        self._trainer.num_nodes
            else:
                optim_config['sched']['max_steps'] = self._trainer.max_steps

        # Force into DictConfig from nested structure
        optim_config = OmegaConf.create(optim_config)
        # Get back nested dict so we its mutable
        optim_config = OmegaConf.to_container(optim_config, resolve=True)

        # Extract scheduler config if inside optimizer config
        if 'sched' in optim_config:
            scheduler_config = optim_config.pop('sched')
        else:
            scheduler_config = None


        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_cls = optim_config.get('_target_', None)

        if optimizer_cls is None:
            # Try to get optimizer name for dynamic resolution, defaulting to Adam
            optimizer_name = optim_config.get('name', 'adam')
        else:
            if inspect.isclass(optimizer_cls):
                optimizer_name = optimizer_cls.__name__.lower()
            else:
                # resolve the class name (lowercase) from the class path if not provided
                optimizer_name = optimizer_cls.split(".")[-1].lower()

        # We are guarenteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_config.get('lr', None)

        # Check if caller has optimizer kwargs, default to empty dictionary
        if 'args' in optim_config:
            optimizer_args = optim_config.pop('args')
            optimizer_args = parse_optimizer_args(
                optimizer_name, optimizer_args)
        else:
            optimizer_args = copy.deepcopy(optim_config)

            # Remove extra parameters from optimizer_args nest
            # Assume all other parameters are to be passed into optimizer constructor
            optimizer_args.pop('name', None)
            optimizer_args.pop('cls', None)
            optimizer_args.pop('lr', None)

        # Adaptive schedulers don't need `lr`
        if lr is not None:
            optimizer_args['lr'] = lr

        # Actually instantiate the optimizer
        if optimizer_cls is not None:
            if inspect.isclass(optimizer_cls):
                optimizer = optimizer_cls(self.parameters(), **optimizer_args)
                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer

            else:
                # Attempt class path resolution
                try:
                    optimizer_cls = OmegaConf.create(
                        {'_target_': optimizer_cls})
                    if lr is not None:
                        optimizer_config = {'lr': lr}
                    else:
                        optimizer_config = {}
                    optimizer_config.update(optimizer_args)

                    optimizer_instance = hydra.utils.instantiate(
                        optimizer_cls, self.parameters(), **optimizer_config
                    )  # type: DictConfig

                    logging.info("Optimizer config = %s",
                                 str(optimizer_instance))

                    self._optimizer = optimizer_instance

                except Exception as e:
                    logging.error(
                        "Could not instantiate class path - {} with kwargs {}".format(
                            optimizer_cls, str(optimizer_config)
                        )
                    )
                    raise e

        else:
            optimizer = get_optimizer(optimizer_name)
            optimizer = optimizer(self.parameters(), **optimizer_args)

            logging.info("Optimizer config = %s", str(optimizer))

            self._optimizer = optimizer

        # Try to instantiate scheduler for optimizer
        self._scheduler = prepare_lr_scheduler(
            optimizer=self._optimizer, scheduler_config=scheduler_config,
            train_dataloader=pp({'num_samples' : self.train_size*self.hparams.model.dataset.train_ds.batch_size, 
            'batch_size': self.hparams.model.dataset.train_ds.batch_size,
            'drop_last' : self.hparams.model.dataset.drop_last})
            )

        # Return the optimizer with/without scheduler
        # This return allows multiple optimizers or schedulers to be created
        return self._optimizer, self._scheduler

    def configure_optimizers(self):
        self.setup_optimization(self.hparams.model.optim)
        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def setup_datamodule(self, data_config: Optional[DictConfig] = None):
        if data_config is None:
            data_config = self._cfg.model.dataset
        self._cfg.model.punct_label_ids=OmegaConf.create(sorted(self._cfg.model.punct_label_ids))
        labels_to_ids = {_[1]:_[0] for _ in enumerate(self._cfg.model.punct_label_ids)}
        data_config.num_labels=len(self._cfg.model.punct_label_ids)
        data_config.labelled = OmegaConf.create([] if data_config.labelled==None else data_config.labelled)
        data_config.unlabelled = OmegaConf.create([] if data_config.unlabelled==None else data_config.unlabelled)
        data_config.num_domains = len(data_config.labelled)+len(data_config.unlabelled)
        self.dm=PunctuationDataModule(
            tokenizer= self._cfg.model.transformer_path,
            labelled= list(data_config.labelled),
            unlabelled= list(data_config.unlabelled),
            punct_label_ids= labels_to_ids,
            label_map=self.label_map,
            train_batch_size= data_config.train_ds.batch_size,
            max_seq_length= data_config.max_seq_length,
            val_batch_size= data_config.validation_ds.batch_size,
            num_workers= data_config.num_workers,
            pin_memory= data_config.pin_memory,
            train_shuffle= data_config.train_ds.shuffle,
            val_shuffle= data_config.validation_ds.shuffle,
            seed=self._cfg.seed,
            data_id=self.data_id,
            tmp_path=self.hparams.tmp_path,
            test_unlabelled=data_config.test_unlabelled,
            attach_label_to_end=data_config.attach_label_to_end,
            manual_len=data_config.train_ds.manual_len,
            no_space_label=self._cfg.model.no_space_label,
            pad_start=data_config.pad_start
        )
        self.dm.setup()
        self._train_dl=self.dm.train_dataloader
        self.train_size = len(self.dm.train_dataset)
        self._validation_dl=self.dm.val_dataloader
        self._test_dl=self.dm.test_dataloader
    
    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl()

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl()

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl()

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            # tar.add(source_dir, arcname=path.basename(source_dir))
            tar.add(source_dir, arcname="./")

    @rank_zero_only
    def save_to(self, save_path: str):

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = path.join(tmpdir, _MODEL_CONFIG_YAML)
            model_weights = path.join(tmpdir, _MODEL_WEIGHTS)

            self.to_config_file(path2yaml_file=config_yaml)
            torch.save(self.state_dict(), model_weights)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        global _MODEL_RESTORE_PATH
        _MODEL_RESTORE_PATH = os.path.abspath(os.path.expanduser(restore_path))
        # Load .nemo tar archive using the old restore method.
        cwd = os.getcwd()
        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls._set_model_restore_state(is_being_restored=True)
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                if override_config_path is not None:
                    # Resolve the override config
                    conf = OmegaConf.to_container(conf, resolve=True)
                    conf = OmegaConf.create(conf)
                    # If override is top level config, extract just `model` from it
                    if 'model' in conf:
                        conf = conf.model

                if return_config:
                    instance = conf
                else:
                    model_weights = path.join(tmpdir, _MODEL_WEIGHTS)
                    OmegaConf.set_struct(conf, True)
                    instance = cls.from_config_dict(config=conf)
                    instance = instance.to(map_location)
                    instance.load_state_dict(torch.load(model_weights, map_location=map_location), strict=strict)

                    logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                cls._set_model_restore_state(is_being_restored=False)
                os.chdir(cwd)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
        checkpoint = None
        try:
            cls._set_model_restore_state(is_being_restored=True)

            checkpoint = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                *args,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint

    @staticmethod
    def _is_model_being_restored() -> bool:
        global _MODEL_IS_RESTORED
        return _MODEL_IS_RESTORED

    @staticmethod
    def _set_model_restore_state(is_being_restored: bool):
        global _MODEL_IS_RESTORED
        _MODEL_IS_RESTORED = is_being_restored

    def freeze_transformer_to(self, n: int, exclude_types=(torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) -> None:
        """Freeze layers up to layer group `n`.
        Look at each group, and freeze each paraemeter, except excluded types
        """
        pp(f"1st {n} encoder layers of transformer frozen")

        def set_requires_grad_for_module(module: torch.nn.Module, requires_grad: bool):
            "Sets each parameter in lthe module to the `requires_grad` value"
            params = list(module.parameters())
            for param in params:
                param.requires_grad = requires_grad
        try:
            encoder = self.transformer.encoder
        except:
            encoder = self.transformer.transformer

        for layer in list(encoder.layer)[:n]:
            if not isinstance(layer, exclude_types):
                set_requires_grad_for_module(layer, False)

        for layer in list(encoder.layer)[n:]:
            set_requires_grad_for_module(layer, True)
        
        # Set output layer to true.
        # last_iter=iter(encoder.layer[n-1].children())
        # last = next(last_iter)
        # for last in last_iter:
        #     continue
        # set_requires_grad_for_module(last, True)
        for name, param in self.transformer.named_parameters():                
            if param.requires_grad:
                print(name)


    def freeze(self) -> None:
        try:
            encoder = self.transformer.encoder
        except:
            encoder = self.transformer.transformer
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        self.frozen = len(encoder.layer)-self.hparams.model.unfrozen
        self.freeze_transformer_to(self.frozen)
        for name, param in encoder.named_parameters(): 
            if param.requires_grad: 
                print(name)

    def unfreeze(self, i: int = 1):
        self.frozen -= i
        self.hparams.model.unfrozen+=i
        assert(self.hparams.model.unfrozen<=self.hparams.model.maximum_unfrozen)
        # if self.hparams.model.unfrozen>self.hparams.model.maximum_unfrozen:
        #     self.frozen+=self.hparams.model.unfrozen-self.hparams.model.maximum_unfrozen
        #     self.hparams.model.unfrozen=self.hparams.model.maximum_unfrozen
        self.freeze_transformer_to(max(0, self.frozen))
        

    def teardown(self, stage: str):
        """
        Called at the end of fit and test.
        Args:
            stage: either 'fit' or 'test'
        """
        if stage == 'fit':
            # Update env variable to bypass multi gpu issue after training
            # This fix affects usage of trainer.test() after trainer.train()
            # If trainer.train() was done on multiple GPUs, then trainer.test()
            # will try to do ddp, even if its a new Trainer object with just 1 GPU.
            # Temporary patch to fix that
            if 'PL_TRAINER_GPUS' in os.environ:
                os.environ.pop('PL_TRAINER_GPUS')

        super().teardown(stage)

    def add_punctuation(self, queries):
        ds=PunctuationInferenceDataset(
            tokenizer= self.tokenizer,
            queries=queries, 
            max_seq_length=self.hparams.model.dataset.max_seq_length,
            punct_label_ids=self.labels_to_ids,
            label_map=self.label_map,
            attach_label_to_end=self._cfg.model.dataset.attach_label_to_end)
        batch=ds[0]
        attention_mask = batch['attention_mask']
        subtoken_mask = batch['subtoken_mask']
        # punct_labels = batch['labels']
        # domain_labels = batch['domain']
        input_ids = batch['input_ids']

        labelled_mask=(subtoken_mask[:,0]>0)
        # test_loss, punct_logits, domain_logits = self._make_step(batch)

        punct_logits, domain_logits = self(
            input_ids=input_ids, attention_mask=attention_mask, subtoken_mask=subtoken_mask,
        )
        
        punct_preds = self.punctuation_loss.predict(punct_logits, subtoken_mask) \
            if self.hparams.model.punct_head.loss == 'crf' else torch.argmax(punct_logits, axis=-1)*subtoken_mask
        return view_aligned(input_ids,punct_preds, self.tokenizer,self.ids_to_labels)