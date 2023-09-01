import argparse
import pandas as pd
import torch
import lightning as L

from torch.utils.data import DataLoader

from model.modeling_demolta import DeMOLTaConfig
from datautils import LitMOLADataModule
from trainer import LitMOLA, SaveTrainableParamsCheckpoint


torch.set_float32_matmul_precision('medium')

def main(
        batch_size,
        max_step,
        text_model_name,
        pretrain_df_path,
        pretrain_val_df_path,
        demolta_size,
        test_df_path,
        accumulate_grad_batches=1,
        hf_token='',
        gcp_credentials_path='',
        bucket_name='',
        destination_blob_name='',
        deepspeed=False,
        device='0'
    ):

    smiles_to_filter = pd.read_csv(test_df_path)['SMILES'].tolist()
    lit_mola_data_module = LitMOLADataModule(
        train_df_path=pretrain_df_path,
        val_df_path=pretrain_val_df_path,
        ignore_smiles=smiles_to_filter,
        tokenizer_name=text_model_name,
        batch_size=batch_size,
        hf_token=hf_token,
    )

    if demolta_size =='xsmall':
        demolta_config = DeMOLTaConfig(
            num_layers=12,
            node_hidden_dim=384,
            edge_hidden_dim=128,
            node_ff_dim=1536,
            edge_ff_dim=768,
            num_heads=6,
        )   
    elif demolta_size =='small':
        demolta_config = DeMOLTaConfig(
            num_layers=6,
            node_hidden_dim=768,
            edge_hidden_dim=256,
            node_ff_dim=3072,
            edge_ff_dim=1536,
            num_heads=12,
        )
    elif demolta_size =='base':
        demolta_config = DeMOLTaConfig(
            num_layers=12,
            node_hidden_dim=768,
            edge_hidden_dim=256,
            node_ff_dim=3072,
            edge_ff_dim=1536,
            num_heads=12,
        )
    elif demolta_size =='large':
        demolta_config = DeMOLTaConfig(
            num_layers=24,
            node_hidden_dim=1024,
            edge_hidden_dim=512,
            node_ff_dim=4096,
            edge_ff_dim=2048,
            num_heads=16,
        )

    lit_model = LitMOLA(
        demolta_config=demolta_config,
        text_model_name=text_model_name,
        hf_token=hf_token,
        deepspeed=deepspeed
    )


    if deepspeed:
        from lightning.pytorch.strategies import DeepSpeedStrategy
        from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

        class CustomDeepSpeedStargy(DeepSpeedStrategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def save_checkpoint(self, checkpoint, filepath, storage_options=None) -> None:
                filepath = self.broadcast(filepath)
                val_loss = self.trainer.callback_metrics("val_loss", None)
                if val_loss is not None:
                    filepath = filepath.replace(".cpkt", f"-val_loss={val_loss:.4f}.ckpt")
                _exclude_keys = ["state_dict", "optimizer_states"]
                checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
                self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint", exclude_frozen_parameters=True)
                convert_zero_checkpoint_to_fp32_state_dict(
                    filepath,
                    os.path.join(filepath,"fp32.pt")
                )

        trainer = L.Trainer(
            accelerator='gpu',
            precision='bf16-mixed',
            max_steps=max_step,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            val_check_interval=10000,
            limit_val_batches=1000,
            strategy=CustomDeepSpeedStargy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
            devices=device
        )
    else:
        checkpoint_callback = SaveTrainableParamsCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoint/',
            filename='mola-pretrain' + f'-{demolta_size}-{text_model_name.split("/")[-1]}' + '-{step}-{train_loss:.4f}-{val_loss:.2f}',
            save_top_k=3,
            save_last=True,
            bucket_name=bucket_name,
            destination_blob_name=destination_blob_name,
            gcp_credentials_path=gcp_credentials_path,
            ddp=True if deepspeed else False
        )

        trainer = L.Trainer(
            accelerator='gpu',
            precision='bf16-mixed',
            max_steps=max_step,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            val_check_interval=10,
            limit_val_batches=10,
            devices=device
        )
    trainer.fit(lit_model, lit_mola_data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_step", type=int)
    parser.add_argument("--text_model_name", type=str)
    parser.add_argument("--pretrain_df_path", type=str)
    parser.add_argument("--pretrain_val_df_path", type=str)
    parser.add_argument("--demolta_size", type=str)
    parser.add_argument("--test_df_path", type=str)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--gcp_credentials_path", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--destination_blob_name", type=str)
    parser.add_argument("--deepspeed", type=lambda x: x=='True', default=False)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    main(
        batch_size=args.batch_size,
        max_step=args.max_step,
        text_model_name=args.text_model_name,
        pretrain_df_path=args.pretrain_df_path,
        pretrain_val_df_path=args.pretrain_val_df_path,
        demolta_size=args.demolta_size,
        test_df_path=args.test_df_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        hf_token=args.hf_token,
        gcp_credentials_path=args.gcp_credentials_path,
        bucket_name=args.bucket_name,
        destination_blob_name=args.destination_blob_name,
        deepspeed=args.deepspeed,
        device=args.device
    )

# pip install deepspeed transformers lightning rdkit-pypi dgl dgllife google-cloud-storage pytorch-metric-learning 
# python pretrain.py --batch_size=4 --max_step=2000000 --text_model_name=facebook/galactica-125m --pretrain_df_path=./preproc/pretrain.csv --pretrain_val_df_path=./preproc/pretrain_val.csv --demolta_size=xsmall --test_df_path=./data/test.csv --accumulate_grad_batches=2 --gcp_credentials_path=./.auth/flowing-banner-391105-04efc2e014a8.json --bucket_name=jinwoo0766 --destination_blob_name=mola_checkpoint --deepspeed=False