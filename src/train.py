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

# def get_hyperparam(model):
#     hparams = {}
#     hparams["params_total"] = sum(p.numel() for p in model.parameters())
#     hparams["params_trainable"] = sum(
#       p.numel() for p in model.parameters() if p.requires_grad
#     )
#     hparams["params_not_trainable"] = sum(
#       p.numel() for p in model.parameters() if not p.requires_grad
#     )
#     print("----------------------------MODEL PARAMS-------------------------------")
#     for param in hparams.keys():
#         print("-----------------------------------------------------------------------")
#         print(f"{param} equals {hparams[param]/1000000:.2f}M")
#     print("-----------------------------------------------------------------------")

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


def train_loop(dataloader, model, ema, ema_model, loss_fn, optimizer, epoch, device="cuda"):
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
        # Compute prediction and loss
        # if first_batch:
        #     first_batch_ = batch
        # batch = first_batch_
        acceleration, target_acceleration, position_seq, acc_loss = model(batch['position'], batch['target'], 
                                                  batch['particle_types'], batch['n_particle'], 
                                                  batch['bonds'], batch.get('bond_types'),
                                                  batch['cg_bonds'], batch['rgs'],
                                                  batch['n_keypoint'], batch['cluster'])#, batch['keypoint'])
            
        # if batch['position'].shape[0]!=800:
        #     print("Tin")

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
        # loss = loss_fn(acceleration, target_acceleration)
        # loss_dist = loss_fn(pred_bead_dist, target_bead_dist)
        # loss = loss + loss_dist

        # loss = acc_loss

        most_recent_velocity = position_seq[:, -1] - position_seq[:, -2]
        zero = torch.zeros(size=acceleration.shape).cuda()
        loss_wo_pred = loss_fn(zero, target_acceleration)
        # loss = -acceleration.log_prob(target_acceleration).mean()
        # first_batch = False
        # Backpropagation
        optimizer.zero_grad()
        acc_loss.backward()
        optimizer.step()
        # ema.step_ema(ema_model, model)
        all_target_acc.append(target_acceleration.detach().cpu().numpy())
        losses.append(acc_loss.detach().item())
        losses_wo.append(loss_wo_pred.item())
        loss_1000.append(acc_loss.detach())
        loss_1000_wo.append(loss_wo_pred)
        # if batch_num % 1000 == 0:
            # # new_position_pred = model._decoder_postprocessor(acceleration, position_seq)
            # name_pred = f"target_vs_prev_huberloss_batch_{batch_num}_epoch_{epoch}"
            # # # plot_3d_graph(new_position_pred, batch, all_atom_position_target, name=name_pred)
            # # new_position_target = model._decoder_postprocessor(target_acceleration, position_seq)
            # # target_plot_3d_graph(position_seq[:][:,-1], batch, new_position_target, name=name_pred)
            # target_plot_3d_graph(new_position_pred, batch, new_position_target, name=name_pred)
            # # name_target = f"pred_vs_prev_batch_{batch_num}_epoch_{epoch}"
            # # target_plot_3d_graph(new_position_pred, batch, position_seq[:][:,-1], name=name_target)
            # # # plot_3d_graph(new_position_target, batch, all_atom_position_target, name=name_target)

            # name_coord = f"pred_vs_prev_vs_target_batch_{batch_num}_epoch_{epoch}"
            # # prev_pred_target_plot_3d_graph(position_seq[:][:,-1], new_position_pred, new_position_target, batch, name=name_coord)
            # prev_pred_target_plot_3d_graph(position_seq[:][:, -1], new_position_pred, new_position_target, batch, name=name_coord)

            # loss, current = loss.item(), (batch_num + 1) * batch['n_particle'].shape[0]
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # loss_wo_pred, current = loss_wo_pred.item(), (batch_num + 1) * batch['n_particle'].shape[0]
            # print(f"loss_wo: {loss_wo_pred:>7f}  [{current:>5d}/{size:>5d}]")
        
        if batch_num % 900 == 0: # and batch_num != 0:
            print(f"Summed loss equals {sum(loss_1000).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000 = []
            print(f"Summed Baseline loss equals {sum(loss_1000_wo).item():.3f}")
            # torch.save(model, f"model_epoch_{epoch}_batch_{batch_num}")
            loss_1000_wo = []

        if batch_num % 900 == 0 and batch_num != 0:
            model_dir = os.getenv('MODEL_DIR')
            torch.save(model, f"{model_dir}/model_overfit_{epoch}_batch_{batch_num}")


    print(f"Epoch loss={sum(losses):.3f}")
    return losses, all_target_acc


@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg: DictConfig) -> None:

    seed_everything()  
    training_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.train.split, 
                                        directory=cfg.data.dir,
                                        dilation = 1,
                                        grouping=100)
    train_dataloader = DataLoader(training_data, 
                                  batch_size=cfg.data.batch_size, 
                                #   batch_size=1,
                                #   shuffle=cfg.data.shuffle, 
                                  shuffle=True,
                                  collate_fn=dict_collate_fn, 
                                #   num_workers=0)
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
    # model_path = "/home/ttunjic/MY_POLYMER_MODELS/model_overfit_17_batch_27900"
    # model = torch.load(model_path)
    get_hyperparam(model)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.HuberLoss()
    # loss_fn = MatrixLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.optimizer.lr)   

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, ema, ema_model, loss_fn, optimizer, epoch=t, device=device)

if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    main()
