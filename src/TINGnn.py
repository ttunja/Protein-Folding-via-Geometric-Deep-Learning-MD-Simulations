# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Graph network implementation accompanying ICML 2020 submission.
   "Learning to Simulate Complex Physics with Graph Networks"
   Alvaro Sanchez-Gonzalez*, Jonathan Godwin*, Tobias Pfaff*, Rex Ying,
   Jure Leskovec, Peter W. Battaglia
   https://arxiv.org/abs/2002.09405
The Sonnet `EncodeProcessDecode` module provided here implements the learnable
parts of the model.
It assumes an encoder preprocessor has already built a graph with
connectivity and features as described in the paper, with features normalized
to zero-mean unit-variance.
Dependencies include Tensorflow 1.x, Sonnet 1.x and the Graph Nets 1.1 library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

# gn.graphs.GraphsTuple

import torch
import torch.nn as nn

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
    layers = [nn.Linear(input_size, hidden_size), activation_func(activation)]
    
    # Add hidden layers
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation_func(activation))
    
    # Add output layer
    layers.append(nn.Linear(hidden_size, output_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(output_size))
    # Create the MLP model as a sequential container
    model = nn.Sequential(*layers)
    
    return model


#######################################################################################################

def build_mlp(in_dim, units, layers, out_dim, 
              act='relu', layer_norm=False, act_final=False):
    
    mods = [nn.Linear(in_dim, units), Act(act)]
    for i in range(layers-1):
        mods += [nn.Linear(units, units), Act(act)]
    mods += [nn.Linear(units, out_dim)]
    if act_final:
        mods += [Act(act)]
    if layer_norm:
        mods += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*mods)

class EdgeBlock(nn.module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
         
class NodeNlock(nn.module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
  
class GraphIndependent(nn.module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
class TemporalInteractionNetwork(nn.module):
  
    def __init__(self) -> None:
        super().__init__()
    def forward():



from typing import Callable
import graph_nets as gn
import sonnet as snt
import tensorflow as tf

Reducer = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def build_mlp(
    hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
  """Builds an MLP."""
  return snt.nets.MLP(
      output_sizes=[hidden_size] * num_hidden_layers + [output_size])  # pytype: disable=bad-return-type  # gen-stub-imports


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      reducer: Reducer = tf.math.unsorted_segment_sum,
      name: str = "EncodeProcessDecode"):
    """Inits the model.
    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of message passing steps.
      output_size: Output size of the decode node representations as required
        by the downstream update function.
      reducer: Reduction to be used when aggregating the edges in the nodes in
        the interaction network. This should be a callable whose signature
        matches tf.math.unsorted_segment_sum.
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size
    self._reducer = reducer

    with self._enter_variable_scope():
      self._networks_builder()

  def _build(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Forward pass of the learnable dynamics model."""

    # Encode the input_graph.
    latent_graph_0 = self._encode(input_graph)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)

    # Decode from the last latent graph.
    return self._decode(latent_graph_m)

  def _networks_builder(self):
    """Builds the networks."""

    def build_mlp_with_layer_norm():
      mlp = build_mlp(
          hidden_size=self._mlp_hidden_size,
          num_hidden_layers=self._mlp_num_hidden_layers,
          output_size=self._latent_size)
      return snt.Sequential([mlp, snt.LayerNorm()])

    # The encoder graph network independently encodes edge and node features.
    encoder_kwargs = dict(
        edge_model_fn=build_mlp_with_layer_norm,
        node_model_fn=build_mlp_with_layer_norm)
    self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = []
    for _ in range(self._num_message_passing_steps):
      self._processor_networks.append(
          gn.modules.InteractionNetwork(
              edge_model_fn=build_mlp_with_layer_norm,
              node_model_fn=build_mlp_with_layer_norm,
              reducer=self._reducer))

    # The decoder MLP decodes node latent features into the output size.
    self._decoder_network = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._output_size)

  def _encode(
      self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Encodes the input graph features into a latent graph."""

    # Copy the globals to all of the nodes, if applicable.
    if input_graph.globals is not None:
      broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
      input_graph = input_graph.replace(
          nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
          globals=None)

    # Encode the node and edge features.
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Processes the latent graph with several steps of message passing."""

    # Do `m` message passing steps in the latent graphs.
    # (In the shared parameters case, just reuse the same `processor_network`)
    latent_graph_prev_k = latent_graph_0
    latent_graph_k = latent_graph_0
    for processor_network_k in self._processor_networks:
      latent_graph_k = self._process_step(
          processor_network_k, latent_graph_prev_k)
      latent_graph_prev_k = latent_graph_k

    latent_graph_m = latent_graph_k
    return latent_graph_m

  def _process_step(
      self, processor_network_k: snt.Module,
      latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Single step of message passing with node/edge residual connections."""

    # One step of message passing.
    latent_graph_k = processor_network_k(latent_graph_prev_k)

    # Add residuals.
    latent_graph_k = latent_graph_k.replace(
        nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
        edges=latent_graph_k.edges+latent_graph_prev_k.edges)
    return latent_graph_k

  def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Decodes from the latent graph."""
    return self._decoder_network(latent_graph.nodes)
  