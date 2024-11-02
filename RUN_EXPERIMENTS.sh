#!/bin/bash
# python src/train.py expname="CG_rgs_more_hparams_more_mp_steps"
# python src/train.py expname="CG_rgs_less_hparams_less_mp_steps" model.embedding_gn_hparams.latent_size=16 model.embedding_gn_hparams.hidden_size=16 model.embedding_gn_hparams.num_message_passing_steps=3
# python src/train.py expname="CG_rgs_more_hparams_less_mp_steps"  model.gn_hparams.num_message_passing_steps=3  model.embedding_gn_hparams.num_message_passing_steps=3 
# python src/train.py expname="CG_rgs_less_hparams_more_mp_steps" model.gn_hparams.latent_size=32 model.gn_hparams.hidden_size=32 model.embedding_gn_hparams.latent_size=16 model.embedding_gn_hparams.hidden_size=16 model.embedding_gn_hparams.output_size=32 model.property_net_hparams.hidden_size=64

python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/calc_CG_less_hparams_more_mp_steps"
python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/noise_not_rw_less_hparams_more_mp_steps"
python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/noise_rw_less_hparams_more_mp_steps"
python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/more_hparams_more_mp_steps" #deter=True
python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/less_hparams_less_mp_steps" #deter=True
python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/more_hparams_less_mp_steps" #deter=True
# python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/less_hparams_more_mp_steps" #deter=True



# How it suppose to be
# Train
# python src/train.py protein="Trp_cage" data="trp_cage" expname="Try_1" model.gn_hparams.latent_size=32 model.gn_hparams.hidden_size=32 model.embedding_gn_hparams.latent_size=16 model.embedding_gn_hparams.hidden_size=16 model.embedding_gn_hparams.output_size=32 model.property_net_hparams.hidden_size=64
# Eval
# python src/eval_new_try.py model_dir="/home/ttunjic/PROTEIN-FOLDING-MODELS/Trp_Cage/less_hparams_more_mp_steps" folded_protein="/home/ttunjic/Protein_Data/Folded Proteins/2jof.pdb" data.directory="/home/ttunjic/Protein_Data/Trp-cage/DESRES-Trajectory_2JOF-0-protein/protein_train"
