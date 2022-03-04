import argparse
import os

import ipdb
import torch

from utils.models import collate_for_delta_jobs
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from data.datasets import DeltaJobsDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        tgt_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_delta_t_TEST.pkl")
        dataset_test = DeltaJobsDataset(CFG["datadir"], tgt_file, hparams.load_dataset, "TEST")
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_delta_jobs)
        preds = []
        truth = []
        error = []
        for idx, jobs, labels in tqdm(test_loader, desc="Testing avg delta..."):
            preds.append(np.mean(jobs))
            truth.append(labels)
            error.append(torch.nn.functional.mse_loss(torch.mean(torch.FloatTensor(jobs)), torch.tensor(labels[0])).item())
        print(torch.mean(torch.FloatTensor(error)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--load_dataset", type=bool, default=False)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    hparams = parser.parse_args()
    main(hparams)
