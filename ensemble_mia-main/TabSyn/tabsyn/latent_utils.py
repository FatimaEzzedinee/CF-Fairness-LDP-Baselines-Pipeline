import os
import json
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess
from tabsyn.vae.model import Decoder_model 

def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']


    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(dataset_dir, task_type = task_type, inverse = True)

    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)

    decoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/decoder.pt'
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    info['pre_decoder'] = pre_decoder
    info['token_dim'] = token_dim

    return train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


 
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    task_type = info['task_type']
    is_classification = task_type in ("classification", "binclass", "multiclass")

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    pre_decoder = info['pre_decoder']
    token_dim = info['token_dim']

    # Reshape for decoder
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    # Decode normalized numerical + categorical
    x_hat_num, x_hat_cat = pre_decoder(torch.tensor(syn_data))

    # Numerical inverse transform (if any numeric features or numeric targets exist)
    if len(num_col_idx) > 0 or (not is_classification and len(target_col_idx) > 0):
        syn_num = num_inverse(x_hat_num.cpu().numpy())
    else:
        syn_num = np.empty((syn_data.shape[0], 0))

    # Categorical inverse transform (if any cat features or categorical targets exist)
    if len(cat_col_idx) > 0 or (is_classification and len(target_col_idx) > 0):
        syn_cat_preds = [pred.argmax(dim=-1) for pred in x_hat_cat]
        syn_cat = torch.stack(syn_cat_preds).t().cpu().numpy()
        syn_cat = cat_inverse(syn_cat)
    else:
        syn_cat = np.empty((syn_data.shape[0], 0))

    # Separate target from features
    if is_classification:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]
    else:  # regression
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target
def recover_data(syn_num, syn_cat, syn_target, info):
    task_type = info['task_type']
    is_classification = task_type in ("classification", "binclass", "multiclass")

    num_col_idx_all = info['num_col_idx']
    cat_col_idx_all = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    idx_mapping = {int(k): v for k, v in info['idx_mapping'].items()}

    # Remove targets from feature lists so offsets match actual syn_num/syn_cat
    num_col_idx = [i for i in num_col_idx_all if i not in target_col_idx]
    cat_col_idx = [i for i in cat_col_idx_all if i not in target_col_idx]

    syn_df = pd.DataFrame()

    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in num_col_idx:
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in cat_col_idx:
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
        elif i in target_col_idx:
            syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]
        else:
            raise ValueError(f"Unexpected column index {i} in idx_mapping.")

    return syn_df

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat

