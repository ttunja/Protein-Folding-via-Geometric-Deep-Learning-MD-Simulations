import os
import copy
import hydra
import numpy as np
import random
import tqdm
import logging
from omegaconf import DictConfig, OmegaConf
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn
from torch.utils.data import DataLoader
from preprocessing.preprocessing_utils import dict_collate_fn
from utils.visualizations import prev_pred_target_plot_3d_graph
import models.simulator_diff as simulator
from models.TINGnn import EMA
import preprocessing.data as data

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
    model.train()
    losses = []
    losses_wo = []
    loss_1000 = []
    loss_1000_wo = []
    all_target_acc = []
    grad_clip = cfg.train.pl_trainer.gradient_clip_val

    for batch_num, batch in enumerate(tqdm.tqdm(dataloader)):
        for feat_args in batch.keys():
            batch[feat_args] = batch[feat_args].to(device)

        rollout_steps = cfg.model.get('rollout_steps', 1)
        if rollout_steps > 1 and 'target_seq' in batch:
            acceleration, target_acceleration, position_seq, loss = model.forward_multistep(
                                                      batch['position'], batch['target_seq'],
                                                      batch['particle_types'], batch['n_particle'],
                                                      batch['bonds'], batch.get('bond_types'),
                                                      batch['cg_bonds'], batch['rgs'],
                                                      batch['n_keypoint'], batch['cluster'])
        else:
            acceleration, target_acceleration, position_seq, loss = model(batch['position'], batch['target'],
                                                      batch['particle_types'], batch['n_particle'],
                                                      batch['bonds'], batch.get('bond_types'),
                                                      batch['cg_bonds'], batch['rgs'],
                                                      batch['n_keypoint'], batch['cluster'])

        new_position_target = model._decoder_postprocessor(target_acceleration, position_seq)
        new_position_pred = model._decoder_postprocessor(acceleration, position_seq)

        # baseline loss (zero prediction) for comparison
        zero = torch.zeros(size=acceleration.shape, device=device)
        loss_wo_pred = loss_fn(zero, target_acceleration)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.step_ema(ema_model, model)

        all_target_acc.append(target_acceleration.detach().cpu().numpy())
        losses.append(loss.item())
        losses_wo.append(loss_wo_pred.item())
        loss_1000.append(loss)
        loss_1000_wo.append(loss_wo_pred)

        if batch_num % 1000 == 0:
            name_coord = f"pred_vs_prev_vs_target_batch_{batch_num}_epoch_{epoch}"
            try:
                prev_pred_target_plot_3d_graph(position_seq[:][:, -1], new_position_pred, new_position_target, batch, name=name_coord)
            except (IndexError, RuntimeError):
                pass  # skip plot if bond indices don't match (e.g. mixed-protein batches)
            logging.info(f"Summed loss equals {sum(loss_1000).item():.3f}")
            loss_1000 = []
            logging.info(f"Summed Baseline loss equals {sum(loss_1000_wo).item():.3f}")
            loss_1000_wo = []

    model_dir = cfg.workdir
    torch.save(model, f"{model_dir}/model_{epoch}")
    torch.save(ema_model, f"{model_dir}/ema_model_{epoch}")
    logging.info(f"Epoch loss={sum(losses):.3f}")
    return losses, all_target_acc


def val_loop(dataloader, model, loss_fn, device="cuda"):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Validation"):
            for feat_args in batch.keys():
                batch[feat_args] = batch[feat_args].to(device)

            acceleration, target_acceleration, position_seq, loss = model(batch['position'], batch['target'],
                                                      batch['particle_types'], batch['n_particle'],
                                                      batch['bonds'], batch.get('bond_types'),
                                                      batch['cg_bonds'], batch['rgs'],
                                                      batch['n_keypoint'], batch['cluster'])
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    logging.info(f"Validation loss={avg_val_loss:.4f}")
    return avg_val_loss

@hydra.main(config_path="conf", config_name="chignolin", version_base=None)
def main(cfg: DictConfig) -> None:

    setup_logging(cfg)
    logging.info("Training started.")

    # Define the path for the configuration file to be saved
    config_save_path = os.path.join(cfg.workdir, 'train_config.yaml')

    # Save the configuration
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    seed_everything()
    rollout_steps = cfg.model.get('rollout_steps', 1)
    training_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.train.split,
                                        directory=cfg.data.dir,
                                        dilation=cfg.model.dilation,
                                        grouping=10,
                                        rollout_steps=rollout_steps)
    train_dataloader = DataLoader(training_data,
                                  batch_size=cfg.data.batch_size,
                                  shuffle=cfg.data.shuffle,
                                  collate_fn=dict_collate_fn,
                                  num_workers=cfg.data.datamodule.num_workers.train)

    # validation set (if split file exists)
    val_dataloader = None
    val_split = cfg.data.datamodule.datasets.val.split
    if os.path.exists(val_split):
        val_data = data.ProteinDataset(split=val_split,
                                       directory=cfg.data.dir,
                                       dilation=cfg.model.dilation,
                                       grouping=10)
        val_dataloader = DataLoader(val_data,
                                    batch_size=cfg.data.batch_size,
                                    shuffle=False,
                                    collate_fn=dict_collate_fn,
                                    num_workers=cfg.data.datamodule.num_workers.val)

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
                                          deterministic=cfg.model.deterministic,
                                          use_gru=cfg.model.get('use_gru', True),
                                          use_temporal=cfg.model.get('use_temporal', True))

    model = model.to(device)
    get_hyperparam(model)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    loss_fn = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.optimizer.lr)

    epochs = cfg.train.pl_trainer.max_epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_loss = float('inf')

    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        losses, _ = train_loop(train_dataloader, model, ema, ema_model, loss_fn, optimizer, epoch=t, cfg=cfg, device=device)

        # validation
        if val_dataloader is not None:
            val_loss = val_loop(val_dataloader, model, loss_fn, device=device)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f"{cfg.workdir}/model_best")
                torch.save(ema_model, f"{cfg.workdir}/ema_model_best")
                logging.info(f"New best model saved (val_loss={val_loss:.4f})")
        else:
            epoch_loss = sum(losses) / len(losses)
            scheduler.step(epoch_loss)

        logging.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    main()
