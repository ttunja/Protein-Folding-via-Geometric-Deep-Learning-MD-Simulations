import os
import copy
import hydra
import numpy as np
import random
import tqdm
import wandb
import logging
import argparse
from omegaconf import DictConfig, OmegaConf
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn
from torch.utils.data import DataLoader
from preprocessing.preprocessing_utils import dict_collate_fn
from utils.visualizations import plot_3d_graph,  target_plot_3d_graph, prev_pred_target_plot_3d_graph
import models.simulator_diff as simulator
from models.TINGnn import EMA
import preprocessing.data as data
from utils.custom_losses import MatrixLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(cfg):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        handlers=[logging.FileHandler(os.path.join(cfg.workdir, 'train.log'), mode='a'),
                                  logging.StreamHandler()])

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,  warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hyperparam(model):
    hparams = {}
    hparams["params_total"] = sum(p.numel() for p in model.parameters())
    hparams["params_trainable"] = sum(
      p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["params_not_trainable"] = sum(
      p.numel() for p in model.parameters() if not p.requires_grad
    )
    logging.info("----------------------------MODEL PARAMS-------------------------------")
    for param in hparams.keys():
        logging.info("-----------------------------------------------------------------------")
        logging.info(f"{param} equals {hparams[param]/1000000:.2f}M")
    logging.info("-----------------------------------------------------------------------")


def train_loop(dataloader, model, ema, ema_model, loss_fn, optimizer, epoch, cfg, device="cuda"):
    size = len(dataloader.dataset)
    losses = []
    losses_wo = []
    loss_1000 = []
    loss_1000_wo = []
    all_target_acc = []
    first_batch=True
    for batch_num, batch in enumerate(tqdm.tqdm(dataloader)):
        for feat_args in batch.keys():
            batch[feat_args] = batch[feat_args].to(device)

        acceleration, target_acceleration, position_seq, loss = model(batch['position'], batch['target'], 
                                                  batch['particle_types'], batch['n_particle'], 
                                                  batch['bonds'], batch.get('bond_types'),
                                                  batch['cg_bonds'], batch['rgs'],
                                                  batch['n_keypoint'], batch['cluster'])
            
        # loss = loss_fn(acceleration, target_acceleration)
        new_position_target = model._decoder_postprocessor(target_acceleration, position_seq)
        new_position_pred = model._decoder_postprocessor(acceleration, position_seq)
        all_atom_position_target = batch['target']
        # loss = loss_fn(new_position_pred, new_position_target)

        ########## calc distances ###################################################################
        bonds = batch['cg_bonds']
        bonds_set = set([tuple(sorted(pair)) for pair in bonds.tolist()])
        non_duplicate_bonds = torch.tensor(list(bonds_set), device=bonds.device)
        def calculate_bond_distances(bonds, positions):
            """
            Calculate the Euclidean distances between bonded beads.

            :param bonds: Tensor of shape [n_bonds, 2], where each row is a pair of indices representing a bond.
            :param positions: Tensor of shape [n_beads, 3], where each row is the 3D position of a bead.
            :return: Tensor of distances for each bond.
            """
            # Extract the positions of the first and second atom in each bond
            start_positions = positions[bonds[:, 0]]
            end_positions = positions[bonds[:, 1]]

            # Calculate the distances
            distances = torch.sqrt(torch.sum((start_positions - end_positions) ** 2, dim=1))
            return distances

        target_bead_dist = calculate_bond_distances(non_duplicate_bonds, new_position_target)
        pred_bead_dist = calculate_bond_distances(non_duplicate_bonds, new_position_pred)
        ############################################################################################
        # loss_dist = loss_fn(pred_bead_dist, target_bead_dist)
        # acc_loss = loss_fn(acceleration, target_acceleration)
        # # loss = acc_loss + score_loss + loss_dist + property_loss
        # loss = acc_loss + score_loss + property_loss

        most_recent_velocity = position_seq[:, -1] - position_seq[:, -2]
        zero = torch.zeros(size=acceleration.shape).cuda()
        loss_wo_pred = loss_fn(zero, target_acceleration)
        # loss = -acceleration.log_prob(target_acceleration).mean()
        # first_batch = False
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema.step_ema(ema_model, model)
        all_target_acc.append(target_acceleration.detach().cpu().numpy())
        losses.append(loss.item())
        losses_wo.append(loss_wo_pred.item())
        loss_1000.append(loss)
        loss_1000_wo.append(loss_wo_pred)
        if batch_num % 1000 == 0:
            name_coord = f"pred_vs_prev_vs_target_batch_{batch_num}_epoch_{epoch}"
            # prev_pred_target_plot_3d_graph(position_seq[:][:,-1], new_position_pred, new_position_target, batch, name=name_coord)
            prev_pred_target_plot_3d_graph(position_seq[:][:, -1], new_position_pred, new_position_target, batch, name=name_coord)

        # if batch_num % 900 == 0:# and batch_num != 0:
            logging.info(f"Summed loss equals {sum(loss_1000).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000 = []
            logging.info(f"Summed Baseline loss equals {sum(loss_1000_wo).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000_wo = []
            # model_dir = os.getenv('MODEL_DIR')
            # torch.save(model, f"{model_dir}/model_{epoch}")

    # model_dir = os.getenv('MODEL_DIR')
    model_dir = cfg.workdir
    torch.save(model, f"{model_dir}/model_{epoch}")
    print(f"Epoch loss={sum(losses):.3f}")
    return losses, all_target_acc

# TODO this cannot be hard coded
@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg: DictConfig) -> None:

    setup_logging(cfg)
    logging.info("Training started.")

    # Define the path for the configuration file to be saved
    config_save_path = os.path.join(cfg.workdir, 'train_config.yaml')

    # Save the configuration
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    seed_everything()  
    training_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.train.split, 
                                        directory=cfg.data.dir,
                                        dilation = 1,
                                        grouping=10)
    train_dataloader = DataLoader(training_data, 
                                  batch_size=cfg.data.batch_size, 
                                #   batch_size=1,
                                  shuffle=cfg.data.shuffle, 
                                #   shuffle=False,
                                  collate_fn=dict_collate_fn, 
                                #   num_workers=0)
                                  num_workers=cfg.data.datamodule.num_workers.train)

    model = simulator.LearnedSimulatorDiff(connectivity_radius=cfg.model.radius,
                                          graph_network_kwargs=cfg.model.gn_hparams,
                                          embedding_network_kwargs=cfg.model.embedding_gn_hparams,
                                          property_network_kwargs=cfg.model.property_net_hparams,
                                          num_particle_types=cfg.model.num_particle_types,
                                          num_bond_types=cfg.model.num_bond_types,
                                          particle_type_embedding_size=cfg.model.type_emb_size,
                                          edge_type_embedding_size=cfg.model.type_emb_size,
                                          sequence_len=cfg.model.seq_len,
                                          noise_std=cfg.model.noise_std,
                                          deterministic=cfg.model.deterministic)

    model = model.to(device)
    # model_path = "/home/ttunjic/PROTEIN-FOLDING-MODELS/model_overfit_5_batch_98100"
    # model = torch.load(model_path)
    get_hyperparam(model)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()
    loss_fn = torch.nn.HuberLoss()
    # loss_fn = MatrixLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.optimizer.lr)   

    epochs = 19
    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, ema, ema_model, loss_fn, optimizer, epoch=t, cfg=cfg, device=device)

if __name__=="__main__":
    import ipdb; ipdb.set_trace()
    main()
