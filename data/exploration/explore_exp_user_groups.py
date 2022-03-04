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
from data.datasets import ExpIndJobsCustomVocabDataset
from utils.baselines import map_profiles_to_label, get_class_weights, pre_proc_data


def main(args):
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    exp_dict = {0: "novice",
                1: "junior",
                2: "intermediary",
                3: "senior",
                4: "expert"}
    num_exp = len(exp_dict)
    with ipdb.launch_ipdb_on_exception():
        if args.load_paths == "False":
            classifier_exp = joblib.load(f"/data/gainondefor/dynamics/exp_best_{args.mod_type}.joblib")
            vectorizer_exp = joblib.load(f"/data/gainondefor/dynamics/exp_best_{args.mod_type}_vectorizer.joblib")
            predict_fn_exp = classifier_exp.decision_function if args.mod_type == "SVM" else classifier_exp.predict_proba

            # load data
            data_train, train_lookup, data_valid, valid_lookup, data_test, test_lookup, class_weights = get_labelled_data(args)

            #ipdb.set_trace()

            # turn data into binary vectors
            feat_exp_train, labels_exp_train = vectorize_data(data_train, vectorizer_exp)
            feat_exp_valid, labels_exp_valid = vectorize_data(data_valid, vectorizer_exp)
            feat_exp_test, labels_exp_test = vectorize_data(data_test, vectorizer_exp)

            paths_train, lookup_train = get_paths_for_split(train_lookup, num_exp, softmax, predict_fn_exp, feat_exp_train, 'train')
            paths_valid, lookup_valid = get_paths_for_split(valid_lookup, num_exp, softmax, predict_fn_exp, feat_exp_valid, 'valid')
            paths_test, lookup_test = get_paths_for_split(test_lookup, num_exp, softmax, predict_fn_exp, feat_exp_test, 'test')

            # update dataset with path and jump number information
            for user in lookup_train.keys():
                lookup_train[user].append(count_steps(lookup_train[user][-1]))
            for user in lookup_valid.keys():
                lookup_valid[user].append(count_steps(lookup_valid[user][-1]))
            for user in lookup_test.keys():
                lookup_test[user].append(count_steps(lookup_test[user][-1]))
            arguments = {"data_dir": CFG["gpudatadir"],
                         "load": "True",
                         "light": "True",
                         "is_toy": False,
                         "max_len": 10,
                         "subsample": 0
                         }
            for split, lookup in zip(["TRAIN", "VALID", "TEST"], chain([lookup_train, lookup_valid, lookup_test])):
                ds = ExpIndJobsCustomVocabDataset(**arguments, split=f"{split}")
                ds.user_lookup = lookup
                ds.save_dataset('paths', split)
            ipdb.set_trace()

            save_paths(CFG, paths_train, args.exp_type, "train")
            save_paths(CFG, paths_valid, args.exp_type, "valid")
            save_paths(CFG, paths_test, args.exp_type, "test")
        else:
            paths_train = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_train.pkl"), 'rb'))
            paths_valid = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_valid.pkl"), 'rb'))
            paths_test = pkl.load(open(os.path.join(CFG["gpudatadir"], f"paths_test.pkl"), 'rb'))

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


def get_paths_for_split(lookup, num_exp, softmax, predict_fn_exp, features, split):
    for user in tqdm(lookup.keys(), desc=f"Computing all paths for split {split}"):
        start_job = lookup[user][0]
        end_job = lookup[user][1]
        num_jobs = end_job - start_job + 1
        confidence_scores = np.zeros((num_jobs, num_exp))
        for num, job_ind in enumerate(range(start_job, end_job + 1)):
            confidence_scores[num, :] = softmax(predict_fn_exp(features[job_ind])[0])
        shortest_path, path_confidence, path_indices = exp_lvl_prediction(num_jobs, num_exp, confidence_scores)
        lookup[user].append(shortest_path)
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


def exp_lvl_prediction(num_jobs, num_exp, confidence_scores):
    path_confidence = np.zeros((num_jobs, num_exp))
    parent_indices = np.zeros((num_jobs, num_exp)) - 1
    path_confidence[0, :] = confidence_scores[0, :]
    for j in range(1, num_jobs):
        for e in range(num_exp):
            if e > 0:
                path_confidence[j, e] = path_confidence[j, e-1]
                parent_indices[j, e] = parent_indices[j, e-1]
            else:
                path_confidence[j, e] = 0
                parent_indices[j, e] = -1
            if path_confidence[j-1, e] + confidence_scores[j, e] > path_confidence[j, e]:
                path_confidence[j, e] = path_confidence[j-1, e] + confidence_scores[j, e]
                parent_indices[j, e] = int(e)
    m = np.argmax(path_confidence[-1, :])# get highest score for last job
    path = [m]
    for job in range(num_jobs - 1, 0, -1):
        m = parent_indices[job, int(m)]
        path.append(int(m))
    shortest_path = path[::-1]
    return shortest_path, path_confidence, parent_indices


def vectorize_data(data, vectorizer_exp):
    pre_processessed_data, labels_exp, _ = pre_proc_data(data)
    features_exp = vectorizer_exp.transform(pre_processessed_data)
    return features_exp, labels_exp


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
        arguments = {"data_dir": CFG["gpudatadir"],
                     "load": "True",
                     "light": "True",
                     "is_toy": False,
                     "max_len": 10,
                     "subsample": 0
                     }
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
    labelled = ExpIndJobsCustomVocabDataset(**arguments, split=f"{split}")
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
    parser.add_argument("--mod_type", type=str, default="SVM")
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--load_paths", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="delta")
    args = parser.parse_args()
    main(args)
