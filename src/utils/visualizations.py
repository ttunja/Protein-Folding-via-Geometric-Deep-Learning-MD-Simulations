# cor_matrix=torch.corrcoef(positions[:166])
import pickle
import matplotlib.pyplot as plt
from chart_studio import plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go
# with open ("C:/Users/ttunj/TINGnn/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/corr_matrix.pkl", "rb") as fp:
#     cor = pickle.load(fp)

# cor.cpu().detach().numpy()

# plt.matshow(cor.cpu().detach().numpy())
# plt.show()

def plot_3d_graph(new_position, batch, name):

    # assert new_position
    save_dir = f"D:/PROTEIN-FOLDING-VISUALIZATIONS/{name}.html"
    
    bonds = batch['bonds']
    particle_types = batch['particle_types']
    if batch['n_particle'].shape[0]!=1:
        particle_num = new_position.shape[0]//batch['n_particle'].shape[0]
        bond_num = batch['bonds'].shape[0]//batch['n_particle'].shape[0]
        new_position = new_position[:particle_num]
        particle_types = particle_types[:particle_num]
        bonds = bonds[:bond_num]
    new_position = new_position.cpu().detach().numpy()
    non_duplicate_bonds = bonds[0:int(bonds.shape[0]/2)] # All bonds have been duplicated, since receivers=senders
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
                line=dict(color='rgb(125,125,125)', width=1),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                            size=6,
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
    fig.write_html(save_dir)
