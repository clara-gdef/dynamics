import yaml
import argparse
import joblib
import os
import ipdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import chain
import pandas as pd
from data.datasets import ExpIndJobsCustomVocabDataset
from utils.visualization import print_steps_to_csv
from utils.models import decode_tokenized_words


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        exp_dict = {0: "novice",
                    1: "junior",
                    2: "intermediary",
                    3: "senior",
                    4: "expert"}
        num_exp = len(exp_dict)
        arguments = {"data_dir": CFG["gpudatadir"],
                     "load": "True",
                     "light": "True",
                     "is_toy": False,
                     "max_len": 10,
                     "subsample": 0
                     }

        with open(os.path.join(CFG["gpudatadir"], "vocab_dict.pkl"), "rb") as f:
            vocab_dict = pkl.load(f)

        fname = "step_exp"

        for split in ["TRAIN", "VALID", "TEST"]:
            ds = ExpIndJobsCustomVocabDataset(**arguments, split=f"{split}")
            lookup = ds.user_lookup
            ppl_list = make_ppl_list(ds, lookup, split, vocab_dict, exp_dict)
            if args.subsample > 0:
                # np.random.shuffle(ppl_list)
                csv_file = print_steps_to_csv(ppl_list[:args.subsample], f"{fname}_{split}_{args.subsample}")
                df = pd.read_csv(csv_file)
                df.to_html(f'html/{fname}_{split}_{args.subsample}.html')
            else:
                csv_file = print_steps_to_csv(ppl_list, f"{fname}_{split}")
                df = pd.read_csv(csv_file)
                df.to_html(f'html/{fname}_{split}.html')


def make_ppl_list(ds, lookup, split, vocab_dict, exp_dict):
    ppl_list = []
    for user in tqdm(lookup.keys(), desc=f"making ppl list for split {split}"):
        start_job = lookup[user][0]
        end_job = lookup[user][1]
        path = lookup[user][2]
        for num, job in enumerate(ds.tuples[start_job: end_job + 1]):
            person = [user]
            person.append(lookup[user][-1])  # = number of steps in career
            person.append(exp_dict[job["exp_index"]])
            person.append(exp_dict[path[num]])
            person.append(decode_tokenized_words(job["indices"][0], vocab_dict))
            person.append(lookup[user][-2])
            ppl_list.append(person)
    return ppl_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=1000)
    args = parser.parse_args()
    main(args)
