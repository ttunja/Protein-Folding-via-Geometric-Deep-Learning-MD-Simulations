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
from utils.visualizations import plot_3d_graph,  target_plot_3d_graph, prev_pred_target_plot_3d_graph
import models.simulator_diff as simulator
from models.simulator_diff import CustomScheduleLR
# from models.TINGnn import EMA
import preprocessing.data as data
from utils.custom_losses import MatrixLoss

# import pytorch_lightning as pl
# from omegaconf import DictConfig  # , OmegaConf
# from pytorch_lightning import seed_everything, Callback
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.callbacks import (
#     EarlyStopping,
#     LearningRateMonitor,
#     ModelCheckpoint,
# )
# from pytorch_lightning.loggers import WandbLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_hyperparam(model) -> None:
    print("Model Summary:")
    print("------------------------------------------------------------")
    module_params = {}

    for name, module in model.named_modules():
        # Split the name of the module to check its depth
        name_split = name.split('.')
        if len(name_split) == 1 and name_split[0] != '':  # Top-level modules
            module_params[name] = sum(p.numel() for p in module.parameters())

    total_params = sum(module_params.values())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_non_trainable_params = total_params - total_trainable_params

    for module_name, params in module_params.items():
        print(f"{module_name:20} | Params: {params:10}")

    print("------------------------------------------------------------")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {total_trainable_params}")
    print(f"Non-trainable params: {total_non_trainable_params}")
    print("------------------------------------------------------------")
    print(f"Total estimated model params size (MB): {total_params * 4 / 1024 / 1024:.3f}")


def train_loop(dataloader, model, optimizer, scheduler, epoch, device="cuda"):
    size = len(dataloader.dataset)
    losses = []
    loss_1000 = []
    for batch_num, batch in enumerate(tqdm.tqdm(dataloader)):
        for feat_args in batch.keys():
            batch[feat_args] = batch[feat_args].to(device)

        acceleration, target_acceleration, position_seq, loss = model(batch['position'], batch['target'], 
                                                  batch['particle_types'], batch['n_particle'], 
                                                  batch['bonds'], batch.get('bond_types'),
                                                  batch['cg_bonds'], batch['rgs'],
                                                  batch['n_keypoint'], batch['cluster'])


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.detach().item())
        loss_1000.append(loss.detach())
        if batch_num % 1000 == 0: # and batch_num != 0:
            print(f"Summed loss equals {sum(loss_1000).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000 = []
    
    torch.save(model, f"model_epoch_{epoch}")
    print(f"Epoch loss={sum(losses):.3f}")
    return losses


@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg: DictConfig) -> None:

    # seed_everything()  
    training_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.train.split, 
                                        directory=cfg.data.dir,
                                        dilation = 1,
                                        grouping=100)
    train_dataloader = DataLoader(training_data, 
                                  batch_size=cfg.data.batch_size, 
                                  shuffle=True,
                                  collate_fn=dict_collate_fn, 
                                  num_workers=cfg.data.datamodule.num_workers.train)

    model = simulator.LearnedSimulatorDiff(connectivity_radius=cfg.model.radius,
                                        graph_network_kwargs=cfg.model.gn_hparams,
                                        embedding_network_kwargs=cfg.model.embedding_gn_hparams,
                                        num_particle_types=cfg.model.num_particle_types,
                                        num_bond_types=cfg.model.num_bond_types,
                                        particle_type_embedding_size=cfg.model.type_emb_size,
                                        edge_type_embedding_size=cfg.model.type_emb_size,
                                        sequence_len=cfg.model.seq_len,
                                        noise_std=cfg.model.noise_std)

    model = model.to(device)
    get_hyperparam(model)
    # ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.optimizer.lr)   

    scheduler = CustomScheduleLR(optimizer, 
                                 min_lr=2e-6, 
                                 decay_steps=2e6, 
                                 decay_rate=0.1)

    epochs = 18
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, scheduler, epoch=t, device=device)

        # Step the scheduler
        # scheduler.step()

if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    main()
