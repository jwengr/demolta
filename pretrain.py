import argparse
import pandas as pd
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from model.modeling_demolta import DeMOLTaConfig
from trainer import LitMOLA, get_mola_dataloader


torch.set_float32_matmul_precision('medium')

def main(
        batch_size,
        max_step,
        text_model_name,
        pretrain_df_path,
        pretrain_val_df_path,
        demolta_size,
        test_df_path
    ):
    smiles_to_filter = pd.read_csv(test_df_path)['SMILES'].tolist()
    train_dataloader = get_mola_dataloader(
        df_path=pretrain_df_path,
        ignore_smiles=smiles_to_filter,
        tokenizer_name=text_model_name,
        batch_size=batch_size,
    )
    val_dataloader = get_mola_dataloader(
        df_path=pretrain_val_df_path,
        ignore_smiles=smiles_to_filter,
        tokenizer_name=text_model_name,
        batch_size=batch_size,
    )
    if demolta_size =='xsmall':
        demolta_config = DeMOLTaConfig(
            num_layers=12,
            hidden_dim=384,
            ff_dim=1536,
            num_heads=6,
        )
    lit_model = LitMOLA(
        demolta_config=demolta_config,
        text_model_name=text_model_name,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoint/',
        filename='mola-pretrain' + f'-{demolta_size}-{text_model_name.split("/")[-1]}' + '-{step}-{val_loss:.2f}',
        save_top_k=3,
    )
    trainer = L.Trainer(
        accelerator='gpu',
        precision='16-mixed',
        max_steps=max_step,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        val_check_interval=10000,
        limit_val_batches=1000,
    )
    trainer.fit(lit_model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int)
    parser.add_argument("max_step", type=int)
    parser.add_argument("text_model_name", type=str)
    parser.add_argument("pretrain_df_path", type=str)
    parser.add_argument("pretrain_val_df_path", type=str)
    parser.add_argument("demolta_size", type=str)
    parser.add_argument("test_df_path", type=str)


    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    main(
        batch_size=args.batch_size,
        max_step=args.max_step,
        text_model_name=args.text_model_name,
        pretrain_df_path=args.pretrain_df_path,
        pretrain_val_df_path=args.pretrain_va_df_path,
        demolta_size=args.demolta_size,
        test_df_path=args.test_df_path
    )