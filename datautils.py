import numpy as np
import pandas as pd
import lightning as L

from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizer

from model.modeling_demolta import MOLLACollateFn, DeMOLTaFeaturizer, FineTuneCollateFn


def smiles_split(df, smiles, seed=42, k_fold=5, splitter='scaffold'):
    import deepchem as dc
    Xs, ys = np.arange(len(smiles)), np.ones(len(smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=ys,w=np.zeros(len(smiles)),ids=smiles)
    if splitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif splitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    elif splitter == 'fingerprints':
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


class MOLADataset(IterableDataset):
    def __init__(self, df, ignore_smiles = []):
        self.df = df
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


class LitMOLADataModule(L.LightningDataModule):
    def __init__(self, train_df_path, val_df_path, ignore_smiles, tokenizer_name, batch_size, hf_token='', **kwargs):
        super().__init__()
        self.train_df_path = train_df_path
        self.val_df_path = val_df_path
        self.ignore_smiles = ignore_smiles
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.hf_token = hf_token

        try:
            if hf_token:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            if hf_token:
                self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, token=hf_token)
            else:
                self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        
        if not self.tokenizer.pad_token:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id=2

        self.featurizer = DeMOLTaFeaturizer()
        self.collate_fn = MOLLACollateFn(self.tokenizer)

    def setup(self, stage: str):
        if stage == 'fit':
            train_df = pd.read_csv(self.train_df_path).sort_values(by='smiles', key=lambda x: -x.str.len()).reset_index(drop=True)
            self.train_dataset = MOLADataset(train_df, self.ignore_smiles)
            val_df = pd.read_csv(self.val_df_path)
            self.val_dataset = MOLADataset(val_df, self.ignore_smiles)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    

class LitMOLAFintTuneDataModule(L.LightningDataModule):
    def __init__(self, df_path, batch_size, seed=42, k_fold=5, train_fold=0, splitter='fingerprints', hf_token='', **kwargs):
        super().__init__()
        self.df_path = df_path
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.seed = seed
        self.k_fold = k_fold
        self.train_fold = train_fold
        self.splitter = splitter

        self.featurizer = DeMOLTaFeaturizer()
        self.collate_fn = FineTuneCollateFn()

    def setup(self, stage: str):
        if stage == 'fit':
            df = pd.read_csv(self.df_path)
            smiles = df['SMILES'].tolist()
            dfs = smiles_split(df, smiles, seed=self.seed, k_fold=self.k_fold, splitter=self.splitter)
            train_df, val_df = dfs[self.train_fold]
            self.train_dataset = FineTuneDataset(train_df)
            self.val_dataset = FineTuneDataset(val_df)
            self.dataset = FineTuneDataset(df)
        elif stage == 'predict':
            self.pred_dataset = FineTuneDataset(pd.read_csv(self.df_path))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
