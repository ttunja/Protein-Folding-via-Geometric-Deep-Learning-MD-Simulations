import os
import copy
import hydra
import numpy as np
import random
import tqdm
import wandb
from omegaconf import DictConfig
import torch
import torch.nn
from torch.utils.data import DataLoader
from preprocessing.preprocessing_utils import dict_collate_fn
from utils.visualizations import plot_3d_graph
import models.simulator as simulator
from models.TINGnn import EMA
import preprocessing.data as data
from utils.custom_losses import MatrixLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_hyperparam(model):
    hparams = {}
    hparams["params_total"] = sum(p.numel() for p in model.parameters())
    hparams["params_trainable"] = sum(
      p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["params_not_trainable"] = sum(
      p.numel() for p in model.parameters() if not p.requires_grad
    )
    print("----------------------------MODEL PARAMS-------------------------------")
    for param in hparams.keys():
        print("-----------------------------------------------------------------------")
        print(f"{param} equals {hparams[param]/1000000:.2f}M")
    print("-----------------------------------------------------------------------")


def train_loop(dataloader, model, ema, ema_model, loss_fn, optimizer, epoch, device="cuda"):
    size = len(dataloader.dataset)
    losses = []
    loss_1000 = []
    all_target_acc = []
    first_batch=True
    for batch_num, batch in enumerate(tqdm.tqdm(dataloader)):
        for feat_args in batch.keys():
            batch[feat_args] = batch[feat_args].to(device)
        # Compute prediction and loss
        acceleration, target_acceleration = model(batch['position'], batch['target'], 
                                                  batch['particle_types'], batch['n_particle'], 
                                                  batch['bonds'], batch.get('bond_types'))
            
        # loss = loss_fn(acceleration, target_acceleration)
        new_position_target = model._decoder_postprocessor(target_acceleration, batch['position'])
        new_position_pred = model._decoder_postprocessor(acceleration, batch['position'])
        loss = loss_fn(new_position_pred, new_position_target)
        
        # loss = -acceleration.log_prob(target_acceleration).mean()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema.step_ema(ema_model, model)
        all_target_acc.append(target_acceleration.detach().cpu().numpy())
        losses.append(loss.item())
        loss_1000.append(loss)
        if batch_num % 100 == 0:
            new_position_pred = model._decoder_postprocessor(acceleration, batch['position'])
            name_pred = f"pred__huberloss_batch_{batch_num}_epoch_{epoch}"
            plot_3d_graph(new_position_pred, batch, name=name_pred)
            new_position_target = model._decoder_postprocessor(target_acceleration, batch['position'])
            name_target = f"target_batch_{batch_num}_epoch_{epoch}"
            plot_3d_graph(new_position_target, batch, name=name_target)

            loss, current = loss.item(), (batch_num + 1) * batch['n_particle'].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if batch_num % 900 == 0 and batch_num != 0:
            print(f"Summed loss equals {sum(loss_1000).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000 = []

        if batch_num % 900 == 0 and batch_num != 0:
            torch.save(model, f"D:/PROTEIN-FOLDING-MODELS/model_overfit_{epoch}_batch_{batch_num}")


    print(f"Epoch loss={sum(losses):.3f}")
    return losses, all_target_acc


@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg: DictConfig) -> None:

    seed_everything()  
    training_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.train.split, 
                                        directory= cfg.data.dir)
    train_dataloader = DataLoader(training_data, 
                                  batch_size=cfg.data.batch_size, 
                                  shuffle=cfg.data.shuffle, 
                                  collate_fn=dict_collate_fn, 
                                  num_workers=cfg.data.datamodule.num_workers.train)

    model = simulator.LearnedSimulator(connectivity_radius=cfg.model.radius,
                                       graph_network_kwargs=cfg.model.gn_hparams,
                                       num_particle_types=cfg.model.num_particle_types,
                                       num_bond_types=cfg.model.num_bond_types,
                                       particle_type_embedding_size=cfg.model.type_emb_size,
                                       edge_type_embedding_size=cfg.model.type_emb_size,
                                       sequence_len=cfg.model.seq_len,
                                       noise_std=cfg.model.noise_std)

    model = model.to(device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.HuberLoss()
    loss_fn = MatrixLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.optimizer.lr)   

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, ema, ema_model, loss_fn, optimizer, epoch=t, device=device)

if __name__=="__main__":
    import ipdb; ipdb.set_trace()
    main()
