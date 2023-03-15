class EncodeProcess(nn.Module):
  
  def __init__(self, node_dim, edge_dim, units, layers, latent_dim, mp_steps):
    super().__init__()
    self._units = units
    self._layers = layers
    self._latent_dim = latent_dim
    self._mp_steps = mp_steps
    self._net_kwargs = {
        'units': units,
        'layers': layers,
        'out_dim': latent_dim,
        'layer_norm': True
    }
    self._encode = GraphIndependent(
        edge_net_kwargs=dict(in_dim=edge_dim, **self._net_kwargs), 
        node_net_kwargs=dict(in_dim=node_dim, **self._net_kwargs))
    self._interaction = nn.ModuleList([
        InteractionLayer(
            edge_net_kwargs=dict(in_dim=latent_dim * 3, **self._net_kwargs), 
            node_net_kwargs=dict(in_dim=latent_dim * 2, **self._net_kwargs))
        for _ in range(self._mp_steps)
    ])

  def forward(self, input_graph):
    
    if input_graph.globals is not None:
      input_graph = input_graph.replace(
          nodes=torch.cat([input_graph.nodes, torch.repeat_interleave(
                              input_graph.globals, input_graph.n_node, dim=0)], dim=1), 
          globals=None)
    latent_graph = self._encode(input_graph)
    
    # Process
    for inter_layer in self._interaction:
      _latent_graph = inter_layer(latent_graph)
      # residual connection
      _latent_graph = _latent_graph.replace(
          nodes=_latent_graph.nodes+latent_graph.nodes,
          edges=_latent_graph.edges+latent_graph.edges)
      latent_graph = _latent_graph
    
    return latent_graph
    
class EncodeProcessDecode(EncodeProcess):
  
  def __init__(self, node_dim, edge_dim, units, layers, latent_dim, mp_steps, out_dim):
    super().__init__(node_dim, edge_dim, units, layers, latent_dim, mp_steps)
    self._out_net_kwargs = {
        'in_dim': latent_dim,
        'units': units,
        'layers': layers,
        'out_dim': out_dim
    }
    self._decode = build_mlp(**self._out_net_kwargs)
    
  def forward(self, input_graph, return_latent=False):
    latent_graph = super().forward(input_graph)
    output = self._decode(latent_graph.nodes)
    if return_latent:
      return output, latent_graph
    else:
      return output
    