# cor_matrix=torch.corrcoef(positions[:166])
import os
import pickle
import matplotlib.pyplot as plt
from chart_studio import plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.io import write_image
import torch
# with open ("C:/Users/ttunj/TINGnn/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/corr_matrix.pkl", "rb") as fp:
#     cor = pickle.load(fp)

# cor.cpu().detach().numpy()

# plt.matshow(cor.cpu().detach().numpy())
# plt.show()

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
    # write_image(fig, save_img)
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
    