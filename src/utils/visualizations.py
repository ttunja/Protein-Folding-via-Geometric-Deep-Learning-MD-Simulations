
import os
import pickle
import matplotlib.pyplot as plt
from chart_studio import plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.io import write_image
from plotly.subplots import make_subplots
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy.stats import gaussian_kde
# with open ("C:/Users/ttunj/TINGnn/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/corr_matrix.pkl", "rb") as fp:
#     cor = pickle.load(fp)

# cor.cpu().detach().numpy()

# plt.matshow(cor.cpu().detach().numpy())
# plt.show()

def plot_scatter(y1, y2=None, name=None):

    x_array = np.arange(1, y1.shape[0]+1)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_array, y=y1,
                        mode='lines',
                        name='Predicted'))
    if y2 is not None:
        assert y1.shape[0]==y2.shape[0], "Two arrays are not same in length"
        fig.add_trace(go.Scatter(x=x_array, y=y2,
                            mode='lines',
                            name='Real'))

    fig.write_image(name)

def KDE_scatter2d(pred_acceleration, acceleration, name):
 # Assuming you have `true_acceleration` and `predicted_acceleration` of shape (N_beads, N_frames, 3)

    # Step 1: Flatten the acceleration data into a 2D array of shape (N_frames*N_beads, 3)
    true_acc_flat = np.array(acceleration.reshape(-1, 3))
    pred_acc_flat = np.array(pred_acceleration.reshape(-1, 3))

    # Step 2: Estimate the Kernel Density for both true and predicted accelerations
    kde_true = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(true_acc_flat)
    kde_pred = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(pred_acc_flat)

    # Adjust grid limits to cover both true and predicted data
    x_min, y_min, z_min = np.min(true_acc_flat, axis=0)
    x_max, y_max, z_max = np.max(true_acc_flat, axis=0)

    # Create a grid over the 3D space (you can adjust the range/resolution)
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    z_grid = np.linspace(z_min, z_max, 100)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    # Combine grid points into a list of 3D coordinates
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Evaluate the densities on the grid points
    true_density = np.exp(kde_true.score_samples(grid_points))
    pred_density = np.exp(kde_pred.score_samples(grid_points))

    # Reshape densities to match the grid shape
    true_density = true_density.reshape(X.shape)
    pred_density = pred_density.reshape(X.shape)

    z_plane_idx = 50
    x_slice = X[:, :, z_plane_idx]
    y_slice = Y[:, :, z_plane_idx]
    true_density_slice = true_density[:, :, z_plane_idx]
    pred_density_slice = pred_density[:, :, z_plane_idx]

    # Create a figure with two subplots: one for true KDE, one for predicted KDE
    fig = make_subplots(rows=1, cols=2, subplot_titles=("True Acceleration KDE", "Predicted Acceleration KDE"))

    # Plot true density slice
    fig.add_trace(go.Contour(
        z=true_density_slice,
        x=x_slice[0],  # The x coordinates
        y=y_slice[:, 0],  # The y coordinates
        colorscale='Blues',
        contours=dict(showlabels=False),
    ), row=1, col=1)

    # Plot predicted density slice
    fig.add_trace(go.Contour(
        z=pred_density_slice,
        x=x_slice[0],  # The x coordinates
        y=y_slice[:, 0],  # The y coordinates
        colorscale='Reds',
        contours=dict(showlabels=False),
    ), row=1, col=2)

    # Update layout
    fig.update_layout(height=600, width=1000, title_text="KDE Comparison of True and Predicted Accelerations at z-plane 50")
    fig.write_image(name)

    return true_density, pred_density

def KDE_scatter3d(pred_acceleration, acceleration, name):
    # Step 1: Flatten the acceleration data into a 2D array of shape (N_frames*N_beads, 3)
    true_acc_flat = np.array(acceleration.reshape(-1, 3))
    pred_acc_flat = np.array(pred_acceleration.reshape(-1, 3))

    # Step 2: Estimate the Kernel Density for both true and predicted accelerations
    kde_true = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(true_acc_flat)
    kde_pred = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(pred_acc_flat)

    # Generate a grid of points over the range of data to evaluate the densities
    x_min, y_min, z_min = np.min(true_acc_flat, axis=0)
    x_max, y_max, z_max = np.max(true_acc_flat, axis=0)

    # Create a grid over the 3D space (you can adjust the range/resolution)
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    z_grid = np.linspace(z_min, z_max, 100)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

    # Combine grid points into a list of 3D coordinates
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Evaluate the densities on the grid points
    true_density = np.exp(kde_true.score_samples(grid_points))
    pred_density = np.exp(kde_pred.score_samples(grid_points))

    # Reshape densities to match the grid shape
    true_density = true_density.reshape(X.shape)
    pred_density = pred_density.reshape(X.shape)
    # Select a threshold to display only points with significant density values
    threshold = 0.01  # You can adjust this to control the sparsity of the plot

    # Flatten the grid and densities to match the scatter3d input
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    true_density_flat = true_density.ravel()
    pred_density_flat = pred_density.ravel()

    # Filter points where the density is above the threshold
    true_filter = true_density_flat > threshold
    pred_filter = pred_density_flat > threshold

    # True acceleration KDE 3D scatter plot
    true_scatter = go.Scatter3d(
        x=x_flat[true_filter],
        y=y_flat[true_filter],
        z=z_flat[true_filter],
        mode='markers',
        marker=dict(
            size=5,
            color=true_density_flat[true_filter],  # Use density values to color the points
            colorscale='Blues',
            opacity=0.8,
            colorbar=dict(title='True Density'),
        ),
        name='True KDE'
    )

    # Predicted acceleration KDE 3D scatter plot
    pred_scatter = go.Scatter3d(
        x=x_flat[pred_filter],
        y=y_flat[pred_filter],
        z=z_flat[pred_filter],
        mode='markers',
        marker=dict(
            size=5,
            color=pred_density_flat[pred_filter],  # Use density values to color the points
            colorscale='Reds',
            opacity=0.8,
            colorbar=dict(title='Predicted Density'),
        ),
        name='Predicted KDE'
    )

    # Create the figure with both true and predicted scatter plots
    fig = go.Figure(data=[true_scatter, pred_scatter])

    # Update layout
    fig.update_layout(
        title='3D Scatter Plot of True and Predicted KDE',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(x=0.85, y=0.9)
    )

    # save the image
    fig.write_image(name)


def plot_ramachandran(phi_angles, psi_angles, name):
    """
    Plot a high-quality Ramachandran plot with density estimation.
    """
    plt.figure(figsize=(8, 6))
    
    # Perform 2D Kernel Density Estimation (KDE)
    xy = np.vstack([phi_angles, psi_angles])
    kde = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so high-density points are plotted on top
    idx = kde.argsort()
    phi_angles, psi_angles, kde = phi_angles[idx], psi_angles[idx], kde[idx]
    
    # Add contour lines to emphasize high-density regions
    sns.kdeplot(x=phi_angles, y=psi_angles, cmap='Blues', shade=True, levels=10, linewidths=1.5)

    # Scatter plot with density coloring
    plt.scatter(phi_angles, psi_angles, c=kde, s=10, cmap='coolwarm', edgecolor='black')

    # Set limits for the axes
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    # Label the axes and set the title
    plt.xlabel("Phi angles (degrees)", fontsize=14)
    plt.ylabel("Psi angles (degrees)", fontsize=14)
    plt.title("Pseduo-Ramachandran Plot", fontsize=16)

    # Adjust tick marks and grid
    plt.xticks(np.arange(-180, 181, 60), fontsize=12)
    plt.yticks(np.arange(-180, 181, 60), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add colorbar to indicate density
    cbar = plt.colorbar()
    cbar.set_label('Density', fontsize=12)

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(name, dpi=300)


def plot_3d_graph(new_position, batch, real_position, name):

    # assert new_position
    visualization_dir = os.getenv('VISUALIZATION_DIR')
    save_dir = f"{visualization_dir}/{name}.html"
    save_img = f"{visualization_dir}/{name}.png"
    
    # bonds = batch['bonds']
    bonds = batch['cg_bonds']
    real_bonds = batch['bonds']
    particle_types = batch['particle_types']
    if batch['n_particle'].shape[0]!=1:
        particle_num = new_position.shape[0]//batch['n_particle'].shape[0]
        bond_num = bonds.shape[0]//batch['n_particle'].shape[0]
        new_position = new_position[:particle_num]
        # particle_types = particle_types[:particle_num]
        particle_types = torch.tensor([i for i in range(particle_num)])
        bonds = bonds[:bond_num]

        particle_num_real = real_position.shape[0]//batch['n_particle'].shape[0]
        bond_num_real = real_bonds.shape[0]//batch['n_particle'].shape[0]
        real_position = real_position[:particle_num_real]
        real_bonds = real_bonds[:bond_num_real]

    new_position = new_position.cpu().detach().numpy()
    real_position = real_position.cpu().detach().numpy()
    non_duplicate_bonds = bonds[0:int(bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
    non_duplicate_bonds = non_duplicate_bonds.cpu().detach().numpy()
    real_non_duplicate_bonds = real_bonds[0:int(real_bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
    real_non_duplicate_bonds = real_non_duplicate_bonds.cpu().detach().numpy()
    particle_types = particle_types.cpu().detach().numpy()

    Xn=[new_position[k][0] for k in range(new_position.shape[0])]# x-coordinates of nodes
    Yn=[new_position[k][1] for k in range(new_position.shape[0])]# y-coordinates
    Zn=[new_position[k][2] for k in range(new_position.shape[0])]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in non_duplicate_bonds:
        Xe+=[new_position[e[0]][0],new_position[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[new_position[e[0]][1],new_position[e[1]][1], None]# y-coordinates
        Ze+=[new_position[e[0]][2],new_position[e[1]][2], None]# z-coordinates


    Xn2=[real_position[k][0] for k in range(real_position.shape[0])]# x-coordinates of nodes
    Yn2=[real_position[k][1] for k in range(real_position.shape[0])]# y-coordinates
    Zn2=[real_position[k][2] for k in range(real_position.shape[0])]# z-coordinates
    Xe2=[]
    Ye2=[]
    Ze2=[]
    for e in real_non_duplicate_bonds:
        Xe2+=[real_position[e[0]][0],real_position[e[1]][0], None]# x-coordinates of edge ends
        Ye2+=[real_position[e[0]][1],real_position[e[1]][1], None]# y-coordinates
        Ze2+=[real_position[e[0]][2],real_position[e[1]][2], None]# z-coordinates

    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color=particle_types,
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )
    
    trace3=go.Scatter3d(x=Xe2,
                y=Ye2,
                z=Ze2,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace4=go.Scatter3d(x=Xn2,
                y=Yn2,
                z=Zn2,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color="red",
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title="Positions of the atoms",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        )

    data=[trace1, trace2, trace3, trace4]
    fig=go.Figure(data=data, layout=layout)
    fig.write_html(save_dir)
    write_image(fig, save_img)



def target_plot_3d_graph(new_position, batch, real_position, name):

    # assert new_position
    visualization_dir = os.getenv('VISUALIZATION_DIR')
    save_dir = f"{visualization_dir}/{name}.html"
    save_img = f"{visualization_dir}/{name}.png"
    
    # bonds = batch['bonds']
    bonds = batch['cg_bonds']
    particle_types = batch['particle_types']
    if batch['n_particle'].shape[0]!=1:
        particle_num = new_position.shape[0]//batch['n_particle'].shape[0]
        bond_num = bonds.shape[0]//batch['n_particle'].shape[0]
        new_position = new_position[:particle_num]
        # particle_types = particle_types[:particle_num]
        particle_types = torch.tensor([i for i in range(particle_num)])
        bonds = bonds[:bond_num]

        real_position = real_position[:particle_num]
    

    new_position = new_position.cpu().detach().numpy()
    real_position = real_position.cpu().detach().numpy()
    bonds_set = set([tuple(sorted(pair)) for pair in bonds.tolist()])
    non_duplicate_bonds = torch.tensor(list(bonds_set), device=bonds.device)
    # non_duplicate_bonds = bonds[0:int(bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
    non_duplicate_bonds = non_duplicate_bonds.cpu().detach().numpy()
    particle_types = particle_types.cpu().detach().numpy()

    Xn=[new_position[k][0] for k in range(new_position.shape[0])]# x-coordinates of nodes
    Yn=[new_position[k][1] for k in range(new_position.shape[0])]# y-coordinates
    Zn=[new_position[k][2] for k in range(new_position.shape[0])]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in non_duplicate_bonds:
        Xe+=[new_position[e[0]][0],new_position[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[new_position[e[0]][1],new_position[e[1]][1], None]# y-coordinates
        Ze+=[new_position[e[0]][2],new_position[e[1]][2], None]# z-coordinates


    Xn2=[real_position[k][0] for k in range(real_position.shape[0])]# x-coordinates of nodes
    Yn2=[real_position[k][1] for k in range(real_position.shape[0])]# y-coordinates
    Zn2=[real_position[k][2] for k in range(real_position.shape[0])]# z-coordinates
    Xe2=[]
    Ye2=[]
    Ze2=[]
    for e in non_duplicate_bonds:
        Xe2+=[real_position[e[0]][0],real_position[e[1]][0], None]# x-coordinates of edge ends
        Ye2+=[real_position[e[0]][1],real_position[e[1]][1], None]# y-coordinates
        Ze2+=[real_position[e[0]][2],real_position[e[1]][2], None]# z-coordinates

    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color=particle_types,
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )
    
    trace3=go.Scatter3d(x=Xe2,
                y=Ye2,
                z=Ze2,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace4=go.Scatter3d(x=Xn2,
                y=Yn2,
                z=Zn2,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color="red",
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title="Positions of the atoms",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        )

    data=[trace1, trace2, trace3, trace4]
    fig=go.Figure(data=data, layout=layout)
    fig.write_html(save_dir)
    write_image(fig, save_img)
#########################################################################################################################
################################################# EVALUATION ############################################################
#########################################################################################################################
def eval_plot_3d_graph(new_position, batch,  name):

    # assert new_position
    visualization_dir = os.getenv('VISUALIZATION_DIR')
    save_dir = f"{visualization_dir}/{name}.html"
    save_img = f"{visualization_dir}/{name}.png"
    
    # bonds = batch['bonds']
    bonds = batch['cg_bonds']
    particle_types = batch['particle_types']
    if batch['n_particle'].shape[0]!=1:
        particle_num = new_position.shape[0]//batch['n_particle'].shape[0]
        bond_num = bonds.shape[0]//batch['n_particle'].shape[0]
        new_position = new_position[:particle_num]
        # particle_types = particle_types[:particle_num]
        particle_types = torch.tensor([i for i in range(particle_num)])
        bonds = bonds[:bond_num]

    new_position = new_position.cpu().detach().numpy()
    bonds_set = set([tuple(sorted(pair)) for pair in bonds.tolist()])
    non_duplicate_bonds = torch.tensor(list(bonds_set), device=bonds.device)
    # non_duplicate_bonds = bonds[0:int(bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
    non_duplicate_bonds = non_duplicate_bonds.cpu().detach().numpy()
    particle_types = particle_types.cpu().detach().numpy()

    Xn=[new_position[k][0] for k in range(new_position.shape[0])]# x-coordinates of nodes
    Yn=[new_position[k][1] for k in range(new_position.shape[0])]# y-coordinates
    Zn=[new_position[k][2] for k in range(new_position.shape[0])]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in non_duplicate_bonds:
        Xe+=[new_position[e[0]][0],new_position[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[new_position[e[0]][1],new_position[e[1]][1], None]# y-coordinates
        Ze+=[new_position[e[0]][2],new_position[e[1]][2], None]# z-coordinates


    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color=particle_types,
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )


    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title="Positions of the atoms",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        )

    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    # fig.write_html(save_dir)
    write_image(fig, save_img)


def prev_pred_target_plot_3d_graph(old_position, new_position, real_position, batch, name):

    # assert new_position
    visualization_dir = os.getenv('VISUALIZATION_DIR')
    save_dir = f"{visualization_dir}/{name}.html"
    save_img = f"{visualization_dir}/{name}.png"
    
    # bonds = batch['bonds']
    bonds = batch['cg_bonds']
    particle_types = batch['particle_types']
    if batch['n_particle'].shape[0]!=1:
        particle_num = new_position.shape[0]//batch['n_particle'].shape[0]
        bond_num = bonds.shape[0]//batch['n_particle'].shape[0]
        new_position = new_position[:particle_num]
        # particle_types = particle_types[:particle_num]
        particle_types = torch.tensor([i for i in range(particle_num)])
        bonds = bonds[:bond_num]

        real_position = real_position[:particle_num]
        old_position = old_position[:particle_num]
    
    new_position = new_position.cpu().detach().numpy()
    real_position = real_position.cpu().detach().numpy()
    old_position = old_position.cpu().detach().numpy()
    bonds_set = set([tuple(sorted(pair)) for pair in bonds.tolist()])
    non_duplicate_bonds = torch.tensor(list(bonds_set), device=bonds.device)
    # non_duplicate_bonds = bonds[0:int(bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
    non_duplicate_bonds = non_duplicate_bonds.cpu().detach().numpy()
    particle_types = particle_types.cpu().detach().numpy()

    Xn=[new_position[k][0] for k in range(new_position.shape[0])]# x-coordinates of nodes
    Yn=[new_position[k][1] for k in range(new_position.shape[0])]# y-coordinates
    Zn=[new_position[k][2] for k in range(new_position.shape[0])]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in non_duplicate_bonds:
        Xe+=[new_position[e[0]][0],new_position[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[new_position[e[0]][1],new_position[e[1]][1], None]# y-coordinates
        Ze+=[new_position[e[0]][2],new_position[e[1]][2], None]# z-coordinates


    Xn2=[real_position[k][0] for k in range(real_position.shape[0])]# x-coordinates of nodes
    Yn2=[real_position[k][1] for k in range(real_position.shape[0])]# y-coordinates
    Zn2=[real_position[k][2] for k in range(real_position.shape[0])]# z-coordinates
    Xe2=[]
    Ye2=[]
    Ze2=[]
    for e in non_duplicate_bonds:
        Xe2+=[real_position[e[0]][0],real_position[e[1]][0], None]# x-coordinates of edge ends
        Ye2+=[real_position[e[0]][1],real_position[e[1]][1], None]# y-coordinates
        Ze2+=[real_position[e[0]][2],real_position[e[1]][2], None]# z-coordinates

    Xn3=[old_position[k][0] for k in range(old_position.shape[0])]# x-coordinates of nodes
    Yn3=[old_position[k][1] for k in range(old_position.shape[0])]# y-coordinates
    Zn3=[old_position[k][2] for k in range(old_position.shape[0])]# z-coordinates
    Xe3=[]
    Ye3=[]
    Ze3=[]
    for e in non_duplicate_bonds:
        Xe3+=[old_position[e[0]][0],old_position[e[1]][0], None]# x-coordinates of edge ends
        Ye3+=[old_position[e[0]][1],old_position[e[1]][1], None]# y-coordinates
        Ze3+=[old_position[e[0]][2],old_position[e[1]][2], None]# z-coordinates

    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color=particle_types,
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )
    
    trace3=go.Scatter3d(x=Xe2,
                y=Ye2,
                z=Ze2,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace4=go.Scatter3d(x=Xn2,
                y=Yn2,
                z=Zn2,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color="red",
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )

    trace5=go.Scatter3d(x=Xe3,
                y=Ye3,
                z=Ze3,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='none'
                )

    trace6=go.Scatter3d(x=Xn3,
                y=Yn3,
                z=Zn3,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=9,
                            color="blue",
                            colorscale='Viridis',
                            line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title="Positions of the atoms",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        )

    data=[trace1, trace2, trace3, trace4, trace5, trace6]
    fig=go.Figure(data=data, layout=layout)
    fig.write_html(save_dir)



def create_video(path):
    import os
    import imgkit
    import glob
    from moviepy.editor import ImageSequenceClip

    # 1. Render HTML files to images
    html_files = glob.glob(path + '/*.png')  # specify your path
    html_files = sorted(html_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    image_files = []

    for i, file in enumerate(html_files):
        image_files.append(file)

    # 2. Combine images into a video
    clip = ImageSequenceClip(image_files, fps=2)  # adjust fps (frames per second) as needed
    clip.write_videofile("output.mp4")


if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    visualization_dir = os.getenv('VISUALIZATION_DIR')
    create_video(f"{visualization_dir}/eval")
    