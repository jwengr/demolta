import pandas as pd
import torch
import torch.nn as nn
import lightning as L

from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint

from model.modeling_demolta import MOLLA, DeMOLTaFeaturizer, MOLLACollateFn
from optim import Lion

from weakref import proxy

class SaveTrainableParamsCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def _save_checkpoint(self, trainer, filepath):
        model = trainer.lightning_module
        optimizer = trainer.optimizers[0]

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


class LitMOLA(L.LightningModule):
    def __init__(self, demolta_config, text_model_name):
        super().__init__()
        self.save_hyperparameters()
        self.model = MOLLA(demolta_config, text_model_name)

    def forward(self, x):
        return self.model(x)

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