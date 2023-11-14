#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


def activation_func(act:str="relu"):
    """The possible three activation functions are relu,
    leaky relu and softplus.
    """
    assert act in {'relu', 'leaky_relu', 'softplus'}, \
    "act must be one of {'relu', 'leaky_relu', 'softplus'}"
    if act == "relu":
        return F.relu()
    elif act == "leaky_relu":
        return F.leaky_relu()
    else:
        return F.softplus(beta=1)

def build_mlp(input_size, hidden_size, output_size, num_hidden_layers, 
              activation="relu", use_layer_norm=False):
    # Add input layer
    # layers = [nn.Linear(input_size, hidden_size), activation_func(activation)]
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    
    # Add hidden layers
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        # layers.append(activation_func(activation))
        layers.append(nn.ReLU())
    
    # Add output layer
    layers.append(nn.Linear(hidden_size, output_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(output_size))
    # Create the MLP model as a sequential container
    model = nn.Sequential(*layers)
    
    return model

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=200):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class EdgeBlock(nn.Module):

    def __init__(self, net_kwargs):
        super().__init__()
        self._edge_mlp = build_mlp(**net_kwargs)
    
    def forward(self, graph):
        collected_edges = torch.cat([graph.edges, 
                                     graph.nodes[graph.receivers], 
                                     graph.nodes[graph.senders]], dim=1)
        updated_edges = self._edge_mlp(collected_edges)
    
        return graph.replace(edges=updated_edges)
         
class NodeNlock(nn.Module):
    def __init__(self,
               net_kwargs,
               reducer='sum'):
        """
        use node features and receiver messages. 
        """
        super().__init__()
        self._node_mlp = build_mlp(**net_kwargs)
        self._reducer = reducer
    
    def forward(self, graph):
        receiver_agg = scatter(graph.edges, graph.receivers, dim=0, 
                            dim_size=graph.nodes.shape[0], reduce=self._reducer)
        nodes_to_collect = [graph.nodes, receiver_agg]
        collected_nodes = torch.cat(nodes_to_collect, dim=1)
        updated_nodes = self._node_mlp(collected_nodes)
        return graph.replace(nodes=updated_nodes)
  
class GraphIndependent(nn.Module):
    def __init__(self, 
                 edge_net_kwargs=None,
                 node_net_kwargs=None):
        super().__init__()
        self._edge_net_kwargs = edge_net_kwargs
        self._node_net_kwargs = node_net_kwargs
        
        if self._edge_net_kwargs:
            self._edge_mlp = build_mlp(**self._edge_net_kwargs)
        if self._node_net_kwargs:
            self._node_mlp = build_mlp(**self._node_net_kwargs)
  
    def forward(self, graph):
        
        if self._edge_net_kwargs:
            updated_edges = self._edge_mlp(graph.edges)
        else:
            updated_edges = graph.edges
        if self._node_net_kwargs:
            updated_nodes = self._node_mlp(graph.nodes)
        else:
            updated_nodes = graph.nodes
        
        return graph.replace(edges=updated_edges, nodes=updated_nodes)
    
class TemporalInteractionNetwork(nn.Module):
    # Add Temporal features to Edge and/or Node block
    def __init__(self,
               edge_net_kwargs,
               node_net_kwargs,
               reducer='sum'):
    # global is concat rather than using a global block
        super().__init__()
        self._edge_net_kwargs = edge_net_kwargs
        self._node_net_kwargs = node_net_kwargs
        self._reducer = reducer
        self._edge_block = EdgeBlock(self._edge_net_kwargs)
        self._node_block = NodeNlock(self._node_net_kwargs, self._reducer)

    def forward(self, graph):
        graph = self._edge_block(graph)
        graph = self._node_block(graph)
        return graph

class EncodeProcessDecode(nn.Module):

    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_size: int, 
                 num_hidden_layers: int, 
                 latent_size: int, 
                 num_message_passing_steps: int, 
                 output_size: int=3):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._latent_size = latent_size
        self._num_message_passing_steps = num_message_passing_steps

        self._net_kwargs = {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'output_size': latent_size,
            'use_layer_norm': True
        }
        self._out_net_kwargs = {
        'input_size': latent_size,
        'hidden_size': hidden_size,
        'num_hidden_layers': num_hidden_layers,
        'output_size': output_size
        }   

        self._encode = GraphIndependent(
            edge_net_kwargs=dict(input_size=edge_dim, **self._net_kwargs), 
            node_net_kwargs=dict(input_size=node_dim, **self._net_kwargs))
        self._process = nn.ModuleList([
            TemporalInteractionNetwork(
                edge_net_kwargs=dict(input_size=latent_size * 3, **self._net_kwargs), 
                node_net_kwargs=dict(input_size=latent_size * 2, **self._net_kwargs))
            for _ in range(self._num_message_passing_steps)
        ])

        self._decode = build_mlp(**self._out_net_kwargs)

    def forward(self, input_graph, return_latent=False):
        
        if input_graph.globals is not None:
            input_graph = input_graph.replace(
                nodes=torch.cat([input_graph.nodes, torch.repeat_interleave(
                                    input_graph.globals, input_graph.n_node, dim=0)], dim=1), 
                globals=None)
        latent_graph = self._encode(input_graph)
        
        # Process
        for process_layer in self._process:
            _latent_graph = process_layer(latent_graph)
            # residual connection
            _latent_graph = _latent_graph.replace(
                nodes=_latent_graph.nodes+latent_graph.nodes,
                edges=_latent_graph.edges+latent_graph.edges)
            latent_graph = _latent_graph

        output = self._decode(latent_graph.nodes)
        if return_latent:
            return output, latent_graph
        else:
            return output
