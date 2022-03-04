import argparse
import joblib
import yaml
import os
import ipdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import Counter
from operator import itemgetter
from itertools import chain
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from data.datasets import ExpIndJobsCustomVocabDataset, StringIndSubDataset, StringDataset
from utils.baselines import map_profiles_to_label, get_class_weights, pre_proc_data


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        data_train, data_valid, data_test, class_weights = get_data(CFG, args)

        train_lu = data_train.user_lookup
        valid_lu = data_valid.user_lookup
        test_lu = data_test.user_lookup
        exp_seq = Counter()
        exp_seq = get_exp_sequence(train_lu, data_train.tuples, 0, exp_seq, "train")
        exp_seq = get_exp_sequence(valid_lu, data_valid.tuples, len(data_train), exp_seq, "valid")
        exp_seq = get_exp_sequence(test_lu, data_test.tuples, len(data_train)+len(data_valid), exp_seq, "test")
        total_users = len(train_lu)+len(valid_lu)+len(test_lu)
        ipdb.set_trace()
        exp_seq.most_common(10)


def get_exp_sequence(users, jobs, offset, exp_seq, split):
    for num, user in enumerate(tqdm(users, desc=f"parsing users for {split} split...")):
        current_user = users[user]
        start = current_user[0] + offset
        end = current_user[1] + offset
        current_seq = []
        for job in range(start, end):
            tmp = jobs[job]["exp_index"]
            current_seq.append(tmp)
        if len(current_seq) > 0:
            exp_seq[str(current_seq)] += 1
    print(f"Number of sequence acquired: {sum([i for i in exp_seq.values()])}")
    return exp_seq


def get_data(CFG, args):
    suffix = ""
    print("Building datasets...")
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": args.subsample,
                 "max_len": args.max_len,
                 "exp_levels": args.exp_levels,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    if args.sub_ind == 'True':
        arguments["already_subbed"] = 'True'
        arguments["tuple_list"] = None
        arguments["lookup"] = None
        arguments["suffix"] = args.suffix
        data_train = StringIndSubDataset(**arguments, split="TRAIN")
        data_valid = StringIndSubDataset(**arguments, split="VALID")
        data_test = StringIndSubDataset(**arguments, split="TEST")
        suffix = f"_subInd{args.suffix}"
    else:
        data_train = StringDataset(**arguments, split="TRAIN")
        data_valid = StringDataset(**arguments, split="VALID")
        data_test = StringDataset(**arguments, split="TEST")
    with open(os.path.join(CFG["gpudatadir"], f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_dynamics{suffix}.pkl"),
              'rb') as f_name:
        class_weights = pkl.load(f_name)
    return data_train, data_valid, data_test, class_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod_type", type=str, default="SVM")
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--suffix", type=str, default="_new_it8")
    parser.add_argument("--exp_type", type=str, default="uniform")
    args = parser.parse_args()
    main(args)
