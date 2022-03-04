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
from data.datasets import ExpIndJobsCustomVocabDataset, StringDataset
from utils.baselines import map_profiles_to_label, get_class_weights, pre_proc_data


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    num_delta = 5
    with ipdb.launch_ipdb_on_exception():
        if args.load_paths == "False":

            # load data
            data_train, train_lookup, data_valid, valid_lookup, data_test, test_lookup, class_weights = get_labelled_data(args)

            paths_train, lookup_train = get_paths_for_split(train_lookup, data_train, 'train')
            paths_valid, lookup_valid = get_paths_for_split(valid_lookup, data_valid, 'valid')
            paths_test, lookup_test = get_paths_for_split(test_lookup, data_test, 'test')

            # update dataset with path and jump number information
            for user in lookup_train.keys():
                lookup_train[user].append(count_steps(lookup_train[user][-1]))
            for user in lookup_valid.keys():
                lookup_valid[user].append(count_steps(lookup_valid[user][-1]))
            for user in lookup_test.keys():
                lookup_test[user].append(count_steps(lookup_test[user][-1]))
            arguments = {'data_dir': CFG["gpudatadir"],
                         "load": "False",
                         "subsample": -1,
                         "max_len": 10,
                         "exp_levels": 5,
                         "exp_type": "delta",
                         "is_toy": "False"}
            for split, lookup in zip(["TRAIN", "VALID", "TEST"], chain([lookup_train, lookup_valid, lookup_test])):
                ds = StringDataset(**arguments, split=f"{split}")
                ds.user_lookup = lookup
                ds.save_dataset('paths', split)
            ipdb.set_trace()

            save_paths(CFG, paths_train, args.exp_type, "train")
            save_paths(CFG, paths_valid, args.exp_type, "valid")
            save_paths(CFG, paths_test, args.exp_type, "test")
        else:
            paths_train = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_{args.exp_type}_train.pkl"), 'rb'))
            paths_valid = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_{args.exp_type}_valid.pkl"), 'rb'))
            paths_test = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_{args.exp_type}_test.pkl"), 'rb'))

        all_possible_paths = dict()
        for k, v in chain(paths_train.items(), paths_valid.items(), paths_test.items()):
            if k not in all_possible_paths.keys():
                all_possible_paths[k] = v
            else:
                all_possible_paths[k] += v

        unseen_paths, num_steps, num_step_dist = {}, {}, {}
        for split, paths in zip(["train", "valid", 'test'], [paths_train, paths_valid, paths_test]):
            unseen_paths[split] = {}
            for k, v in paths.items():
                if k not in unseen_paths.keys():
                    unseen_paths[split][k] = v
                else:
                    unseen_paths[split][k] += v
            num_steps[split] = {}
            for k, v in unseen_paths[split].items():
                k_array = string_to_float_array(k)
                num_st = count_steps(np.array(k_array))
                if num_st in num_steps[split].keys():
                    num_steps[split][num_st] += v
                else:
                    num_steps[split][num_st] = v
            total = sum(num_steps[split].values())
            num_step_dist[split] = {}
            for k, v in num_steps[split].items():
                num_step_dist[split][k] = 100*v/total


def save_paths(CFG, data, exp_type, split):
    tgt_file = os.path.join(CFG["gpudatadir"], f"paths_{exp_type}_{split}.pkl")
    with open(tgt_file, 'wb') as f:
        pkl.dump(data, f)
    print(f"Data saved at: {tgt_file}")


def string_to_float_array(input_string):
    array = []
    tmp = input_string.split(' ')
    for item in tmp:
        if not item.startswith('[') and not item.endswith(']'):
            array.append(float(item[0]))
        elif item.startswith('['):
            array.append(float(item.split('[')[-1][0]))
        else:
            array.append(float(item.split(']')[0]))
    return array


def get_paths_for_split(lookup, dataset, split):
    for user in tqdm(lookup.keys(), desc=f"Computing all paths for split {split}"):
        start_job = lookup[user][0]
        end_job = lookup[user][1]
        path = []
        for num, job_ind in enumerate(range(start_job, end_job + 1)):
            path.append(dataset[num][-1])
        lookup[user].append(path)
    existing_paths = Counter()
    for user in tqdm(lookup.keys(), desc=f"Getting exisiting paths for split {split}"):
        existing_paths[str(lookup[user][-1])] += 1
    return existing_paths, lookup


def count_steps(exp_levels):
    steps = 0
    for i in range(len(exp_levels) - 1):
        if exp_levels[i] < exp_levels[i+1]:
            steps += 1
    return steps


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.load_dataset == "True":
        print("Loading data...")
        data_train, train_user_lookup = load_data_from_save("TRAIN")
        data_valid, valid_user_lookup = load_data_from_save("VALID")
        data_test, test_user_lookup = load_data_from_save("TEST")

        with open(os.path.join(CFG["gpudatadir"], "class_weights_dict_dynamics.pkl"), 'rb') as f_name:
            class_weights = pkl.load(f_name)
    else:
        print("Building datasets...")
        arguments = {'data_dir': CFG["gpudatadir"],
                         "load": "False",
                         "subsample": -1,
                         "max_len": 10,
                         "exp_levels": 5,
                         "exp_type": "delta",
                         "is_toy": "False"}
        data_train, train_user_lookup = map_and_save_jobs_to_labels(arguments, "TRAIN")
        data_valid, valid_user_lookup = map_and_save_jobs_to_labels(arguments, "VALID")
        data_test, test_user_lookup = map_and_save_jobs_to_labels(arguments, "TEST")

        class_weights = get_class_weights(data_train)
        with open(os.path.join(CFG["gpudatadir"], "class_weights_dict_dynamics.pkl"), 'wb') as f_name:
            pkl.dump(class_weights, f_name)

    return data_train, train_user_lookup, \
           data_valid, valid_user_lookup, \
           data_test, test_user_lookup,\
           class_weights


def load_data_from_save(split):
    with open(os.path.join(CFG["gpudatadir"], f"txt_job_titles_labelled_{split}.pkl"), 'rb') as f_name:
        data = pkl.load(f_name)
    with open(os.path.join(CFG["gpudatadir"], f"user_to_job_lookup_{split}.pkl"), 'rb') as f_name:
        lookup = pkl.load(f_name)
    return data, lookup


def map_and_save_jobs_to_labels(arguments, split):
    labelled = StringDataset(**arguments, split=f"{split}")
    user_lookup = labelled.user_lookup

    ppl_file = os.path.join(CFG["gpudatadir"], f"profiles_jobs_ind_title_{split}.pkl")
    with open(ppl_file, 'rb') as f:
        raw_profiles_train = pkl.load(f)
    data = map_profiles_to_label(raw_profiles_train, labelled)
    with open(os.path.join(CFG["gpudatadir"], f"txt_job_titles_labelled_{split}.pkl"), 'wb') as f_name:
        pkl.dump(data, f_name)
    with open(os.path.join(CFG["gpudatadir"], f"user_to_job_lookup_{split}.pkl"), 'wb') as f_name:
        pkl.dump(user_lookup, f_name)
    return data, user_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--load_paths", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="delta")
    args = parser.parse_args()
    main(args)
