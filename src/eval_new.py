import os
import time
import hydra
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
import preprocessing.data as data
from train_new import seed_everything

from preprocessing.preprocessing_utils import dict_collate_fn, load_pt

# from graphwm.common import PROJECT_ROOT
# from graphwm.model import GNS, PnR
from models.simulator_diff import LearnedSimulatorDiff


def run_eval(cfg):
    seed_everything(cfg.random_seed)

    model_dir = Path(cfg.workdir)
    dataclass, modelclass = model_dir.parts[-1].split('_')[:2]
    save_dir = Path(cfg.savedir)
  
  
    folder_name = 'rollouts'
    # if (save_dir / folder_name / f'seed_{cfg.random_seed}.pt').exists():
    #   print('Rollout already exists.')
    #   return
    
    # last_model_dir = Path(cfg.model_dir) / 'last.ckpt'
    last_model_dir = Path("/home/ttunjic/MY_POLYMER_MODELS/chignolin_TINGnn/model_epoch_18") # TODO change
    if last_model_dir.exists():
        ckpt_path = str(last_model_dir)
    else:
        return
        
    print(f'load checkpoint: {ckpt_path}')

    model = torch.load(last_model_dir)
    # prepare data
    test_data = data.ProteinDataset(split=cfg.data.datamodule.datasets.test.split, 
                                    directory=cfg.data.dir,
                                    grouping=100)
    data_loader = DataLoader(test_data, 
                             batch_size=40, 
                             shuffle=False, 
                             collate_fn=dict_collate_fn, 
                             num_workers=8)
    model = model.to('cuda')
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    outputs = []
    
    now = time.time()
    
    last_component = 0
    last_cluster = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx == 1:
                break
            batch = {k: v.cuda() for k, v in batch.items()}
            simulate_steps = 100000
            output = model.simulate(batch, simulate_steps - 30, 
                                    save_positions=True, deter=False, 
                                    save_frequency=100)
            output.update({k: v.detach().cpu() for k, v in batch.items()})
            
            # if model.hparams.cg_level > 1:
            output['cluster'] += last_cluster  # fix the offests for <cluster>
            last_cluster = output['cluster'].max()+1 
            if 'component' in output:
                output['component'] += last_component  # fix the offsets for <component>
                last_component = output['component'].max()+1    
            outputs.append(output)
    elapsed = time.time() - now
    
    # for output in range(cfg.num_batches):
    outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[-1].keys()}
    outputs['time_elapsed'] = elapsed
    # outputs['model_params'] = model.hparams
    outputs['eval_cfg'] = cfg
    
    os.makedirs(save_dir / folder_name, exist_ok=True)
    torch.save(outputs, save_dir / folder_name / f'seed_{cfg.random_seed}.pt')
    
    print(f'Finished rollouts of {simulate_steps} steps.')
    
@hydra.main(config_path="conf", config_name="chignolin")
def main(cfg):
    run_eval(cfg)

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    main()
