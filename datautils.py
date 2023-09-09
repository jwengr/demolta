import numpy as np
import pandas as pd
import lightning as L

from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizer

from model.modeling_demolta import MOLLACollateFn, DeMOLTaFeaturizer, DeMOLTaFineTuneCollateFn


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


class MOLLADataset(IterableDataset):
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


class DeMOLTaFineTuneDataset(Dataset):
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

class MOLLAFineTuneDataset(Dataset):
    def __init__(self, df, query, col_name):
        self.df = df
        self.query = query
        self.col_name = col_name
        self.featurizer = DeMOLTaFeaturizer()
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['SMILES']
        sample = {
            'mol_feats': self.featurizer(smiles=smiles),
            'query': self.query,
            'answer': '',
            'target': row[self.col_name]
        }
        return sample  

    def __len__(self):
        return len(self.df)


class LitMOLLADataModule(L.LightningDataModule):
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

        self.collate_fn = MOLLACollateFn(self.tokenizer)

    def setup(self, stage: str):
        if stage == 'fit':
            train_df = pd.read_csv(self.train_df_path).sort_values(by='smiles', key=lambda x: -x.str.len()).reset_index(drop=True)
            train_df = train_df.iloc[int(len(train_df)*0.05):, :]
            train_df.iloc[1:, :] = train_df.iloc[1:, :].sample(frac=1).reset_index(drop=True).values
            self.train_dataset = MOLLADataset(train_df, self.ignore_smiles)
            val_df = pd.read_csv(self.val_df_path)
            self.val_dataset = MOLLADataset(val_df, self.ignore_smiles)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    
class LitMOLLAFineTuneDataModule(L.LightningDataModule):
    def __init__(self, df_path, batch_size, query, column_name, tokenizer_name='', seed=42, k_fold=5, train_fold=0, splitter='fingerprints', hf_token='', **kwargs):
        super().__init__()
        self.df_path = df_path
        self.batch_size = batch_size
        self.query = query
        self.column_name = column_name
        self.tokenizer_name = tokenizer_name
        self.hf_token = hf_token
        self.seed = seed
        self.k_fold = k_fold
        self.train_fold = train_fold
        self.splitter = splitter

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

        self.collate_fn = MOLLACollateFn(self.tokenizer, finetune=True)


    def setup(self, stage: str):
        if stage == 'fit':
            df = pd.read_csv(self.df_path)
            smiles = df['SMILES'].tolist()
            dfs = smiles_split(df, smiles, seed=self.seed, k_fold=self.k_fold, splitter=self.splitter)
            train_df, val_df = dfs[self.train_fold]
            train_df = self.preproc_train(train_df)
            self.train_dataset = MOLLAFineTuneDataset(train_df, self.query, self.column_name)
            self.val_dataset = MOLLAFineTuneDataset(val_df, self.query, self.column_name)
        elif stage == 'predict':
            self.pred_dataset = MOLLAFineTuneDataset(pd.read_csv(self.df_path), self.query, self.column_name)

    def preproc_train(self, df):
        df = df.copy()
        df = df.drop('id',axis=1).groupby('SMILES').mean().reset_index()
        df.loc[df['HLM']>=100, 'HLM'] = 100.0
        df.loc[df['MLM']>=100, 'MLM'] = 100.0
        return df

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


class LitDeMOLTaFineTuneDataModule(L.LightningDataModule):
    def __init__(self, df_path, batch_size, seed=42, k_fold=5, train_fold=0, splitter='fingerprints', hf_token='', **kwargs):
        super().__init__()
        self.df_path = df_path
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.seed = seed
        self.k_fold = k_fold
        self.train_fold = train_fold
        self.splitter = splitter

        self.collate_fn = DeMOLTaFineTuneCollateFn()

    def setup(self, stage: str):
        if stage == 'fit':
            df = pd.read_csv(self.df_path)
            smiles = df['SMILES'].tolist()
            dfs = smiles_split(df, smiles, seed=self.seed, k_fold=self.k_fold, splitter=self.splitter)
            train_df, val_df = dfs[self.train_fold]
            train_df = self.preproc_train(train_df)
            self.train_dataset = DeMOLTaFineTuneDataset(train_df)
            self.val_dataset = DeMOLTaFineTuneDataset(val_df)
        elif stage == 'predict':
            self.pred_dataset = DeMOLTaFineTuneDataset(pd.read_csv(self.df_path))

    def preproc_train(self, df):
        df = df.copy()
        df = df.drop('id',axis=1).groupby('SMILES').mean().reset_index()
        df.loc[df['HLM']>=100, 'HLM'] = 100.0
        df.loc[df['MLM']>=100, 'MLM'] = 100.0
        return df

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
