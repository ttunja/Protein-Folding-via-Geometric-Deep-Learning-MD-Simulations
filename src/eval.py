import os
import time
import logging
from pathlib import Path

# External libraries
import hydra
from omegaconf import DictConfig
import numpy as np
import prody
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

# Internal project-specific imports
import preprocessing.data as data
from preprocessing.preprocessing_utils import dict_collate_fn, load_data
from utils.utils import coarse_graining, compute_com, compute_autocorrelation, compute_ramachandran_angles
from utils.utils import compute_distance_matrix, compute_weighted_rgs, KL_divergence
from utils.visualizations import plot_scatter, KDE_scatter2d, plot_ramachandran
from train import seed_everything

def simulate_trajectory(cfg, dataloader, model):
    seed_everything(1)  # Set random seed for reproducibility
    rollout_len = cfg.rollout_length  # Use config for rollout length
    bead_collision_dist = 5
    eval_dir = Path(cfg.model_dir) / 'eval'
    os.makedirs(eval_dir, exist_ok=True)

    last_cluster = 0
    outputs = []
    
    logging.info("Starting trajectory simulation...")

    now = time.time()

    # Prevent gradient calculation for efficiency
    with torch.no_grad():
       # Extract only the first batch from dataloader
        batch = next(iter(dataloader))
        batch = {feat_args: tensor.to('cuda') for feat_args, tensor in batch.items()}

        # Simulate trajectory
        output = model.simulate(batch, rollout_len - 30, 
                                save_positions=cfg.save_pos, deter=False, 
                                save_frequency=cfg.save_frequency)
        output.update({k: v.detach().cpu() for k, v in batch.items()})

        # Adjust cluster indices
        output['cluster'] += last_cluster
        last_cluster = output['cluster'].max().item() + 1
        outputs.append(output)

    elapsed = time.time() - now
    logging.info(f"Simulation finished in {elapsed:.2f} seconds.")

    # Concatenate the outputs from all batches
    outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[-1].keys()}
    outputs['time_elapsed'] = elapsed
    outputs['eval_cfg'] = cfg

    # Create visualization directories
    create_visualization_directories(cfg.save_dir)

    # Save the outputs to the eval directory
    save_path = Path(cfg.save_dir) / 'eval' / f'seed_{cfg.random_seed}.pt'
    torch.save(outputs, save_path)
    logging.info(f'Simulation outputs saved to {save_path}')

    # Perform evaluation (Rgs, MSD, collisions, etc.)
    evaluate_simulation(cfg, dataloader, outputs, bead_collision_dist, eval_dir)

    return outputs

def create_visualization_directories(save_dir):
    """Create necessary directories for saving visualizations."""
    visualization_paths = [
        'time_rgs',
        'time_rgs_autocorrelation',
        'time_acc_error',
        'KDE_acceleration',
        'time_acc_autocorrelation',
        'bead_collisions',
        'MSD_error',
        'MSD_folded_pred_real',
        'Ramachandran_real_plot',
        'Ramachandran_pred_plot',
        '3d_graphs',  # Additional folder for 3D evaluation graphs
    ]
    
    for path in visualization_paths:
        os.makedirs(Path(save_dir) / 'eval' / path, exist_ok=True)


def evaluate_simulation(cfg, dataloader, outputs, bead_collision_dist, eval_dir):
    """Perform various evaluations such as Rgs, MSD, bead collisions, etc."""
    logging.info("Starting evaluation...")

    # Initialize variables for storing evaluations
    position_list, velocity_list, acceleration_list = [], [], []
    msd_list, msd_fold_pred_list, msd_fold_real_list = [], [], []
    distance_list, collisions_list, pred_distance_list, pred_collisions_list = [], [], [], []
    pred_rgs_list, CG_rgs_list, rgs_list, rgs_list_mean = [], [], [], []
    pred_phi_angles_list, pred_psi_angles_list, real_phi_angles_list, real_psi_angles_list = [], [], [], []

    # process folded protein and coarse grain it
    fold_position, fold_weights, fold_cluster = process_folded_protein(cfg)
    fold_cg_position, fold_cg_weights = compute_com(fold_position, fold_weights, fold_cluster)
    fold_distance_matrices = compute_distance_matrix(fold_cg_position[:, 0, :])

    # Loop over all trajectory paths in the dataset
    for ind, path_index in enumerate(dataloader.dataset.traj_index):
        # Load position data
        position_path = os.path.join(path_index, "position.h5")
        position = np.array(load_data(['position'], position_path)[0], dtype=np.float32)
        position = position.transpose(1, 0, 2)[:, 30:cfg.rollout_length]  # Slice to match rollout length

        # Compute coarse-grained position and weights
        residue_path = os.path.join(path_index, "residue_num.h5")
        residue = load_data(['residue_num'], residue_path)[0]
        cluster = torch.tensor(residue - 1)
        weights = torch.ones(outputs['n_particle'][0], device=outputs['n_particle'][0].device)
        cg_position, cg_weights = compute_com(position, weights, cluster)
        position_list.append(cg_position)

        # Calculate velocity and acceleration
        velocity = cg_position[:, 1:] - cg_position[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        velocity_list.append(velocity)
        acceleration_list.append(acceleration)

        # Perform Rg and distance matrix calculations
        traj_CG_rgs_list, distance_matrices_list, frame_collisions_list = evaluate_framewise_metrics(cg_position, cg_weights, bead_collision_dist)

        # Append results
        CG_rgs_list.append(np.array(traj_CG_rgs_list))
        distance_list.append(torch.stack(distance_matrices_list, dim=-1))
        collisions_list.append(np.array(frame_collisions_list))

        # Calculate and store MSD, bead collisions for the predicted simulation
        pred_metrics = evaluate_framewise_predicted_metrics(cg_position, cg_weights, outputs, bead_collision_dist, ind, fold_distance_matrices)
        pred_rgs_list.append(pred_metrics['pred_CG_rgs'])
        pred_distance_list.append(pred_metrics['pred_distance_matrices'])
        pred_collisions_list.append(pred_metrics['pred_collisions'])
        msd_list.append(pred_metrics['msd'])
        msd_fold_pred_list.append(pred_metrics['msd_fold_pred'])
        msd_fold_real_list.append(pred_metrics['msd_fold_real'])

        # Load real Rg data and append
        rgs_path = os.path.join(path_index, "rgs.h5")
        rgs = load_data(['rgs'], rgs_path)[0][30:cfg.rollout_length]
        rgs_list.append(rgs)
        rgs_list_mean.append(rgs.mean())

        # compute pseudo phi and psi angles
        real_bead_positions = np.array(cg_position)
        real_phi_angles, real_psi_angles = compute_ramachandran_angles(real_bead_positions)
        real_phi_angles_list.append(real_phi_angles)
        real_psi_angles_list.append(real_psi_angles)

        # compute prediction pseudo phi and psi angles
        k = ind * cg_position.shape[0]
        k_plus_1 = (ind + 1) * cg_position.shape[0]
        pred_bead_positions = np.array(outputs['rollout_u_pos'][k:k_plus_1,:])
        pred_phi_angles, pred_psi_angles = compute_ramachandran_angles(pred_bead_positions)
        pred_phi_angles_list.append(pred_phi_angles)
        pred_psi_angles_list.append(pred_psi_angles)

    # Perform final evaluations (e.g., KL divergence, error plots, etc.)
    finalize_evaluations(outputs, CG_rgs_list, pred_rgs_list, eval_dir, msd_list, 
                        msd_fold_pred_list, msd_fold_real_list, collisions_list, 
                        pred_collisions_list, acceleration_list, position_list,
                        pred_phi_angles_list, pred_psi_angles_list, real_phi_angles_list,
                        real_psi_angles_list)

    logging.info("Evaluation complete.")

def process_folded_protein(cfg):
    pdb_file = cfg.folded_protein
    structure = prody.parsePDB(pdb_file)

    # Extract atomic coordinates (this is a NumPy array of shape (N_atoms, 3))
    fold_position = torch.tensor(structure.getCoords()).unsqueeze(1)
    fold_weights = torch.ones(fold_position.shape[0])
    fold_cluster = torch.tensor(structure.getResnums()-1)

    return fold_position, fold_weights, fold_cluster

def evaluate_framewise_metrics(cg_position, cg_weights, bead_collision_dist):
    """Calculate radius of gyration, distance matrices, and bead collisions for each frame."""
    traj_CG_rgs_list, distance_matrices_list, frame_collisions_list = [], [], []

    for frame in range(cg_position.shape[1]):
        # Compute Rg
        CG_rgs = compute_weighted_rgs(cg_position[:, frame], cg_weights, torch.tensor([cg_position.shape[0]]))
        traj_CG_rgs_list.append(CG_rgs.item())

        # Compute distance matrix
        distance_matrices = compute_distance_matrix(cg_position[:, frame, :])
        distance_matrices_list.append(distance_matrices)

        # Calculate bead collisions
        collisions = (distance_matrices.flatten()[distance_matrices.flatten() < bead_collision_dist].shape[0] - cg_position.shape[0]) / 2
        frame_collisions_list.append(collisions)

    return traj_CG_rgs_list, distance_matrices_list, frame_collisions_list


def evaluate_framewise_predicted_metrics(cg_position, cg_weights, outputs,
                                         bead_collision_dist, ind, fold_distance_matrices):
    """Calculate predicted distance matrices, collisions, and MSD."""
    pred_CG_rgs_list, pred_distance_matrices_list, pred_frame_collisions_list = [], [], []
    msd_frame_list, msd_frame_folded_pred, msd_frame_folded_real = [], [], []

    for frame in range(cg_position.shape[1]):
        k = ind * cg_position.shape[0]
        k_plus_1 = (ind + 1) * cg_position.shape[0]
        
        pred_CG_rgs = compute_weighted_rgs(outputs['rollout_u_pos'][k:k_plus_1, frame], cg_weights, torch.tensor([cg_position.shape[0]]))
        pred_CG_rgs_list.append(pred_CG_rgs.item())

        pred_distance_matrices = compute_distance_matrix(outputs['rollout_u_pos'][k:k_plus_1, frame])
        pred_distance_matrices_list.append(pred_distance_matrices)

        # Calculate bead collisions for predicted
        pred_collisions = (pred_distance_matrices.flatten()[pred_distance_matrices.flatten() < bead_collision_dist].shape[0] - cg_position.shape[0]) / 2
        pred_frame_collisions_list.append(pred_collisions)

        # Mean Squared Displacement (MSD) between predicted and real
        msd = torch.mean((pred_distance_matrices - compute_distance_matrix(cg_position[:, frame, :])) ** 2)
        msd_frame_list.append(msd.item())

        # Mean Squared Displacement (MSD) between predicted and folded distances
        msd_fold1 = torch.mean((pred_distance_matrices - fold_distance_matrices) ** 2)
        msd_frame_folded_pred.append(msd_fold1.item())

        # Mean Squared Displacement (MSD) between real and folded distances
        msd_fold2 = torch.mean((compute_distance_matrix(cg_position[:, frame, :]) - fold_distance_matrices) ** 2)
        msd_frame_folded_real.append(msd_fold2.item())

    return {
        'pred_CG_rgs': np.array(pred_CG_rgs_list),
        'pred_distance_matrices': torch.stack(pred_distance_matrices_list, dim=-1),
        'pred_collisions': np.array(pred_frame_collisions_list),
        'msd': np.array(msd_frame_list),
        'msd_fold_pred': np.array(msd_frame_folded_pred),
        'msd_fold_real': np.array(msd_frame_folded_real),
    }


def finalize_evaluations(outputs, CG_rgs_list, pred_rgs_list, eval_dir, msd_list,
                         msd_fold_pred_list, msd_fold_real_list, collisions_list, 
                         pred_collisions_list, acceleration_list, position_list,
                         pred_phi_angles_list, pred_psi_angles_list, real_phi_angles_list,
                         real_psi_angles_list):
    """Finalize evaluations, including plotting and calculating errors."""
    kl_div_list = []  # Initialize list to store KL divergence values

    # Loop through each trajectory to plot and evaluate
    for ind in range(outputs['rollout_prop'].shape[0]):
    # for ind in range(2):
        plot_and_save_evaluation(ind, outputs, CG_rgs_list, pred_rgs_list, 
                                eval_dir, msd_list, msd_fold_pred_list, msd_fold_real_list, 
                                collisions_list, pred_collisions_list, acceleration_list, 
                                kl_div_list, pred_phi_angles_list, pred_psi_angles_list, 
                                real_phi_angles_list, real_psi_angles_list)

        # # Generate 3D evaluation plots
        # for i in tqdm.tqdm(range(outputs['rollout_u_pos'].shape[1])):
        #     eval_plot_3d_graph(outputs['rollout_u_pos'][:, i].squeeze(), None, f"{eval_dir}/{str(i)}")
    
    # # Create video from the saved 3D evaluations
    # visualization_dir = os.getenv("VISUALIZATION_DIR")
    # create_video(f"{visualization_dir}/eval")

    # Calculate average KL divergence
    mean_kl_div = sum(kl_div_list) / len(kl_div_list) if kl_div_list else 0.0

    # Compute final metrics like MAE and R-squared
    avg_real_rgs = torch.tensor([x.mean() for x in CG_rgs_list]) * 100
    avg_pred_rgs = torch.tensor([x.mean() for x in pred_rgs_list]) * 100
    # avg_pred_rgs = outputs['rollout_prop'].mean(dim=1, keepdim=True).squeeze() * 100

    mae = torch.mean(torch.abs(avg_pred_rgs - avg_real_rgs))
    ss_res = torch.sum((avg_real_rgs - avg_pred_rgs) ** 2)
    ss_tot = torch.sum((avg_real_rgs - torch.mean(avg_real_rgs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    logging.info(f"MAE: {mae.item()}")
    logging.info(f"R squared: {r_squared.item()}")
    logging.info(f"Mean KL Divergence: {mean_kl_div}")

def plot_and_save_evaluation(ind, outputs, CG_rgs_list, pred_rgs_list, eval_dir,
                             msd_list, msd_fold_pred_list, msd_fold_real_list, 
                             collisions_list, pred_collisions_list, acceleration_list, 
                             kl_div_list, pred_phi_angles_list, pred_psi_angles_list, 
                             real_phi_angles_list, real_psi_angles_list):
    """Plot and save results for each trajectory."""
    
    # Plot Rgs vs Time
    plot_name = os.path.join(eval_dir, f'time_rgs/rgs_vs_time_{ind}.png')
    # pred_rgs = np.array(outputs['rollout_prop'][ind])
    pred_rgs = pred_rgs_list[ind]
    real_rgs = CG_rgs_list[ind]
    plot_scatter(pred_rgs, real_rgs, plot_name)

    # Plot autocorrelation of Rgs
    pred_rgs_autocorrelation = compute_autocorrelation(torch.tensor(pred_rgs).unsqueeze(0).unsqueeze(2))
    real_rgs_autocorrelation = compute_autocorrelation(torch.tensor(real_rgs).unsqueeze(0).unsqueeze(2))
    plot_name = os.path.join(eval_dir, f'time_rgs_autocorrelation/rgs_autocorrelation_vs_time_{ind}.png')
    plot_scatter(pred_rgs_autocorrelation, real_rgs_autocorrelation, plot_name)

    # Plot acceleration error (True vs Predicted)
    k = ind * acceleration_list[ind].shape[0]
    k_plus_1 = (ind + 1) * acceleration_list[ind].shape[0]
    pred_velocity = outputs['rollout_u_pos'][k:k_plus_1, 1:] - outputs['rollout_u_pos'][k:k_plus_1, :-1]
    pred_acc = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    real_acc = acceleration_list[ind]
    errors = torch.sqrt(((pred_acc - real_acc) ** 2).sum(dim=2))
    avg_error_per_frame = errors.mean(dim=0)
    plot_name = os.path.join(eval_dir, f'time_acc_error/acc_error_vs_time_{ind}.png')
    plot_scatter(avg_error_per_frame, None, plot_name)

    # Plot KDE of accelerations
    plot_name = os.path.join(eval_dir, f'KDE_acceleration/KDE_real_vs_pred_{ind}.png')
    true_density, pred_density = KDE_scatter2d(pred_acc, real_acc, plot_name)
    
    # Compute KL divergence and append to kl_div_list
    kl_div = KL_divergence(true_density, pred_density)
    kl_div_list.append(kl_div)
    
    logging.info(f"KL Divergence between true and predicted accelerations for trajectory {ind}: {kl_div}")

    # Plot autocorrelation of accelerations
    pred_acc_autocorrelation = compute_autocorrelation(pred_acc)
    real_acc_autocorrelation = compute_autocorrelation(real_acc)
    plot_name = os.path.join(eval_dir, f'time_acc_autocorrelation/acc_autocorrelation_vs_time_{ind}.png')
    plot_scatter(pred_acc_autocorrelation, real_acc_autocorrelation, plot_name)

    # Plot bead collisions over time
    pred_bead_collisions = pred_collisions_list[ind]
    real_bead_collisions = collisions_list[ind]
    plot_name = os.path.join(eval_dir, f'bead_collisions/bead_collisions_vs_time_{ind}.png')
    plot_scatter(pred_bead_collisions, real_bead_collisions, plot_name)

    # Plot MSD (Mean Squared Displacement) over time
    msd_error = msd_list[ind]
    plot_name = os.path.join(eval_dir, f'MSD_error/msd_vs_time_{ind}.png')
    plot_scatter(msd_error, None, plot_name)

    # Plot MSD (Mean Squared Displacement) over time
    msd_fold_pred_error = msd_fold_pred_list[ind]
    msd_fold_real_error = msd_fold_real_list[ind]
    plot_name = os.path.join(eval_dir, f'MSD_folded_pred_real/msd_vs_time_{ind}.png')
    plot_scatter(msd_fold_pred_error, msd_fold_real_error, plot_name)

    # Plot Pseudo Ramachandran plot for real data
    real_phi_angles = real_phi_angles_list[ind]
    real_psi_angles = real_psi_angles_list[ind]
    plot_name = os.path.join(eval_dir, f'Ramachandran_real_plot/Ramachandran_real_{ind}.png')
    plot_ramachandran(real_phi_angles, real_psi_angles, plot_name)

    # Plot Pseudo Ramachandran plot for predicted data
    pred_phi_angles = pred_phi_angles_list[ind]
    pred_psi_angles = pred_psi_angles_list[ind]
    plot_name = os.path.join(eval_dir, f'Ramachandran_pred_plot/Ramachandran_pred_{ind}.png')
    plot_ramachandran(pred_phi_angles, pred_psi_angles, plot_name)


@hydra.main(config_path="conf", config_name="eval_chignolin")        
def main(cfg: DictConfig) -> None:
    test_data = data.ProteinDataset(split=cfg.data.split, 
                                    directory=cfg.data.directory,
                                    mode=cfg.data.mode,
                                    grouping=cfg.grouping)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=cfg.batch_size, 
                                 shuffle=cfg.shuffle, 
                                 collate_fn=dict_collate_fn, 
                                 num_workers=0)
    # model_dir = os.getenv('MODEL_DIR')
    model_dir = cfg.model_dir
    # model_dir = "/home/ttunjic/PROTEIN-FOLDING-MODELS/Chignolin/less_hparams_more_mp_steps"
    model_path = f"{model_dir}/model_11"
    model = torch.load(model_path)
    # model = model.to('cuda')
    # model = model.cpu()
    model.eval()

    start_time = time.time()  # Start timing
    rollout = simulate_trajectory(cfg, test_dataloader, model)
    end_time = time.time()  # End timing

    # Calculate the elapsed time in minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    # Log the time taken in minutes
    logging.info(f"Time taken to run simulate_trajectory: {elapsed_time_minutes:.2f} minutes")

if __name__=="__main__":
    import ipdb;ipdb.set_trace()
    main()