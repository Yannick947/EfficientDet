import os 

import pandas as pd
import numpy as np

PARAM_KEYS = ['phi', 'lr', 'random_transform', 'freeze_backbone', 'weighted_bifpn']
RESULT_KEYS = ['val_mAP']


def log_run(args, current_logs): 

    all_keys = PARAM_KEYS + RESULT_KEYS
    params = get_params(current_logs, args)

    existing_logs = get_logs(args.tensorboard_dir, all_keys)

    df_run = pd.Series(params)

    existing_logs = existing_logs.append(df_run, ignore_index=True)

    existing_logs.to_csv(os.path.join(args.tensorboard_dir, 'results.csv'), index=False)


def get_logs(logdir: str, all_keys: list): 
    if os.path.exists(logdir) and os.path.exists(os.path.join(logdir[:logdir.find('tensorboard_pcds')], 'tensorboard_pcds', 'results.csv')):
        return pd.read_csv(os.path.join(logdir[:logdir.find('tensorboard_pcds')], 'tensorboard_pcds', 'results.csv'))
    else: 
        return pd.DataFrame(columns=all_keys)


def get_params(logs, args):
    print('Availabe logs: ', logs)
    params = dict()
    for key in PARAM_KEYS: 
        if hasattr(args, key): 
            params[key] = getattr(args, key)
        else: 
            params[key] = np.nan

    for key in RESULT_KEYS: 
        if hasattr(logs, key):
            params[key] = getattr(logs, key)
        else: 
            params[key] = np.nan

    return params
