import os
import torch
import torch.nn.functional as F
import lightning as L

from google.cloud import storage
from google.oauth2 import service_account
from lightning.pytorch.callbacks import ModelCheckpoint

from model.modeling_demolta import MOLLA, MOLAForMolculeRegression
from optim import Lion

from weakref import proxy

    
def upload_file_to_gcs(bucket_name, destination_blob_name, local_file_path, gcp_credentials_path):
    credentials = service_account.Credentials.from_service_account_file(gcp_credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {destination_blob_name} in {bucket_name}.")


class SaveTrainableParamsCheckpoint(ModelCheckpoint):
    def __init__(self, bucket_name='', destination_blob_name='', gcp_credentials_path='', ddp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcp_bucket_name = bucket_name
        self.destination_blob_name = destination_blob_name
        self.gcp_credentials_path=gcp_credentials_path
        self.ddp=ddp
        

    def _save_checkpoint(self, trainer, filepath):
        if not self.ddp:
            model = trainer.lightning_module
        else:
            model = trainer.lightning_module.module

        # Filter and save trainable parameters
        trainable_state_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

        torch.save(trainable_state_dict, filepath)
        file_name = filepath.split('/')[-1]
        if self.gcp_bucket_name and self.destination_blob_name:
            upload_file_to_gcs(self.gcp_bucket_name, f'{self.destination_blob_name}/{os.path.basename(file_name)}', filepath, self.gcp_credentials_path)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class LitMOLA(L.LightningModule):
    def __init__(self, demolta_config, text_model_name, hf_token=None, deepspeed=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MOLLA(demolta_config, text_model_name, hf_token)
        self.validation_step_outputs = []

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
        loss = outputs[0].detach().cpu()
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        loss = torch.Tensor(self.validation_step_outputs)
        loss = loss.mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            return DeepSpeedCPUAdam(self.parameters(), lr=1e-6)
        else:
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
        labels=batch['labels']/100.0
        outputs = self.model(
            atom_feats=atom_feats,
            bond_feats=bond_feats,
            attention_matrix_mask=attention_matrix_mask,
            labels=labels
        )
        loss, logits = outputs
        loss1, loss2 = loss
        loss = (loss1**0.5 + loss2**0.5)/2
        self.log("train_loss", loss*100, on_epoch=True, prog_bar=True, logger=True)
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
        )
        logits = outputs[1]*100.0
        loss1 = F.mse_loss(logits[:,0].flatten(), labels[:,0].flatten())
        loss2 = F.mse_loss(logits[:,1].flatten(), labels[:,1].flatten())
        loss = (loss1, loss2)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        loss = torch.Tensor(self.validation_step_outputs)
        loss1, loss2 = loss[:, 0], loss[:, 1]
        loss = ((loss1.mean())**0.5 + (loss2.mean())**0.5)/2
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return Lion(self.parameters(), lr=1e-3)

