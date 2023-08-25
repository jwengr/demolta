import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import deepchem as dc
import lightning as L

from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint

from model.modeling_demolta import DeMOLTaCollateFn, DeMOLTaFeaturizer
from model.modeling_demolta import FineTuneCollateFn
from model.modeling_demolta import MOLLA, MOLLACollateFn, MOLAForMolculeRegression
from optim import Lion

from weakref import proxy

class SaveTrainableParamsCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def _save_checkpoint(self, trainer, filepath):
        model = trainer.lightning_module

        # Filter and save trainable parameters
        trainable_state_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

        torch.save(trainable_state_dict, filepath)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

class MOLADataset(IterableDataset):
    def __init__(self, df_path, ignore_smiles = []):
        self.df = pd.read_csv(df_path)
        self.ignore_smiles = ignore_smiles
        self.featurizer = DeMOLTaFeaturizer()

    def __iter__(self):
        for idx, row in self.df.iterrows():
            smiles = row['smiles']
            if smiles in self.ignore_smiles:
                continue
            sample = {
                'mol_feats': self.featurizer(smiles=smiles),
                'query': row['query'],
                'answer': row['answer']
            }
            yield sample

class FineTuneDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.featurizer = DeMOLTaFeaturizer()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['SMILES']
        sample = {
            'mol_feats': self.featurizer(smiles=smiles),
            'label': [row['MLM'], row['HLM']]
        }
        return sample

    def __len__(self):
        return len(self.df)

def get_mola_dataloader(df_path, ignore_smiles, tokenizer_name, batch_size, **kwargs):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id=2

    dataset = MOLADataset(df_path, ignore_smiles)
    collate_fn = MOLLACollateFn(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
    return dataloader

def get_finetune_dataloader(df, batch_size, **kwargs):
    dataset = FineTuneDataset(df)
    collate_fn = FineTuneCollateFn()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
    return dataloader

class LitMOLA(L.LightningModule):
    def __init__(self, demolta_config, text_model_name):
        super().__init__()
        self.save_hyperparameters()
        self.model = MOLLA(demolta_config, text_model_name)

    def training_step(self, batch, batch_idx):
        input_ids=batch['input_ids']
        input_attention_mask=batch['attention_mask']
        atom_feats=batch['mols']['atom_feats']
        bond_feats=batch['mols']['bond_feats']
        attention_matrix_mask=batch['mols']['attention_mask']
        labels=batch['labels']
        outputs = self.model(
            input_ids=input_ids,
            input_attention_mask=input_attention_mask,
            atom_feats=atom_feats,
            bond_feats=bond_feats,
            attention_matrix_mask=attention_matrix_mask,
            labels=labels
        )
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids=batch['input_ids']
        input_attention_mask=batch['attention_mask']
        atom_feats=batch['mols']['atom_feats']
        bond_feats=batch['mols']['bond_feats']
        attention_matrix_mask=batch['mols']['attention_mask']
        labels=batch['labels']
        outputs = self.model(
            input_ids=input_ids,
            input_attention_mask=input_attention_mask,
            atom_feats=atom_feats,
            bond_feats=bond_feats,
            attention_matrix_mask=attention_matrix_mask,
            labels=labels
        )
        loss = outputs[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Lion(self.parameters())
    
class LitMOLAForRegression(L.LightningModule):
    def __init__(self, demolta_config, text_model_name, n_class):
        super().__init__()
        self.model = MOLAForMolculeRegression(demolta_config, text_model_name, n_class)
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        atom_feats=batch['mols']['atom_feats']
        bond_feats=batch['mols']['bond_feats']
        attention_matrix_mask=batch['mols']['attention_mask']
        labels=batch['labels']
        outputs = self.model(
            atom_feats=atom_feats,
            bond_feats=bond_feats,
            attention_matrix_mask=attention_matrix_mask,
            labels=labels
        )
        loss, logits = outputs
        loss = (loss[0]**0.5 + loss[1]**0.5)/2
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        atom_feats=batch['mols']['atom_feats']
        bond_feats=batch['mols']['bond_feats']
        attention_matrix_mask=batch['mols']['attention_mask']
        labels=batch['labels']
        outputs = self.model(
            atom_feats=atom_feats,
            bond_feats=bond_feats,
            attention_matrix_mask=attention_matrix_mask,
            labels=labels
        )
        loss, logits = outputs
        self.validation_step_outputs.append(loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        loss = torch.Tensor(outputs)
        loss1, loss2 = loss[:, 0], loss[:, 1]
        loss = loss1**0.5 + loss2**0.5
        loss = loss.mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs = []
        

def scaffold_split(df, smiles, fraction=0.2, seed=42, k_fold=5, spplitter='scaffold'):
    Xs, ys = np.arange(len(smiles)), np.ones(len(smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=ys,w=np.zeros(len(smiles)),ids=smiles)
    if spplitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif spplitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    elif spplitter == 'fingerprints':
        splitter = dc.splits.FingerprintSplitter()
    folds = splitter.k_fold_split(dataset, k=k_fold, seed=seed)
    dfs = []
    for fold in folds:
        train_indices = fold[0].X
        val_indices = fold[1].X
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        dfs.append((train_df, val_df))
    return dfs