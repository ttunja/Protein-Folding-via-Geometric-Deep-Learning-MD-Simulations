import os
import json
import numpy as np
import torch
import h5py

def store_data(data_names, data, path):
  hf = h5py.File(path, 'w')
  for i in range(len(data_names)):
    hf.create_dataset(data_names[i], data=data[i])
  hf.close()

def load_data(data_names, path):
  hf = h5py.File(path, 'r')
  data = []
  for i in range(len(data_names)):
    d = np.array(hf.get(data_names[i]))
    data.append(d)
  hf.close()
  return data

def load_data_w_idx(data_names, path, select_idx):
  hf = h5py.File(path, 'r')
  data = []
  for i in range(len(data_names)):
    d = np.array(hf.get(data_names[i])[select_idx])
    data.append(d)
  hf.close()
  return data



def fix_topology(topology):

    new_top = topology.copy()

    residues = {}
    for chain in new_top.chains:
        #print chain
        for residue in chain.residues:
            resname = str(residue)
            if resname in list(residues.keys()):
                residues[resname].append(residue)
            else:
                residues[resname] = [residue]

    for resname in list(residues.keys()):
        fragments = residues[resname]
        if len(fragments) > 1:
            main_fragment = fragments[0]
            new_atom_list = []
            new_atom_list += main_fragment._atoms
            for i in range(1,len(fragments)):
                fragment = fragments[i]
                for atom in fragment.atoms:
                    atom.residue = main_fragment
                new_atom_list += fragment._atoms
                fragment._atoms = []
                fragment.chain = main_fragment.chain
            main_fragment._atoms = new_atom_list

    return new_top

def fix_traj(traj):
    time0 = time.time()
    new_traj = copy.deepcopy(traj)
    topology = new_traj.topology 

    new_top = fix_topology(topology)
    topology = new_top
    new_traj.topology = new_top 

    new_atom_sequence = [a for a in topology.atoms]
    new_index_sequence = [a.index for a in topology.atoms]

    for i in range(0, np.shape(traj.xyz)[0]):
        new_traj.xyz[i] = new_traj.xyz[i][new_index_sequence]

    for i in range(0, len(new_index_sequence)):
        new_atom_sequence[i].index = i

    time1 = time.time()
    print(time1 - time0)
    return new_traj

def topology_to_json(topology):
    import json
    topology_object = topology
    try:
        topology_dict = {
            'chains': [],
            'bonds': []
        }

        for chain in topology_object.chains:
            chain_dict = {
                'residues': [],
                'index': int(chain.index)
            }
            for residue in chain.residues:
                residue_dict = {
                    'index': int(residue.index),
                    'name': str(residue.name),
                    'atoms': [],
                    "resSeq": int(residue.resSeq),
                    "segmentID": str(residue.segment_id)
                }

                for atom in residue.atoms:

                    try:
                        element_symbol_string = str(atom.element.symbol)
                    except AttributeError:
                        element_symbol_string = ""

                    residue_dict['atoms'].append({
                        'index': int(atom.index),
                        'name': str(atom.name),
                        'element': element_symbol_string
                    })
                chain_dict['residues'].append(residue_dict)
            topology_dict['chains'].append(chain_dict)

        for atom1, atom2 in topology_object.bonds:
            topology_dict['bonds'].append([
                int(atom1.index),
                int(atom2.index)
            ])

    except AttributeError as e:
        raise AttributeError('topology_object fails to implement the'
            'chains() -> residue() -> atoms() and bond() protocol. '
            'Specifically, we encountered the following %s' % e)


    data = json.dumps(topology_dict)
    if not isinstance(data, bytes):
        data = data.encode('ascii') 
    
    return data

    # if self.tables.__version__ >= '3.0.0':
    #     self._handle.create_array(where='/', name='topology', obj=[data])
    # else:
    #     self._handle.createArray(where='/', name='topology', object=[data])

