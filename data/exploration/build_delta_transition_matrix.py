import argparse
import yaml
import os
import ipdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain
from data.datasets import StringDataset
from utils.baselines import map_profiles_to_label, get_class_weights, pre_proc_data
from sklearn.preprocessing import normalize


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    num_delta = 5
    with ipdb.launch_ipdb_on_exception():
        matrix_file = os.path.join(CFG["datadir"], "transitions_matrix_delta.pkl")
        if args.load_matrix == "False":

            # load data
            data_train, train_lookup, data_valid, valid_lookup, data_test, test_lookup, class_weights = get_labelled_data(args)
            transition_matrix = np.zeros((num_delta, num_delta))

            for user_id in tqdm(train_lookup.keys()):
                start_job = train_lookup[user_id][0]
                end_job = train_lookup[user_id][1]
                for i in range(start_job, end_job):
                    src_delta = data_train[i][-1]
                    tgt_delta = data_train[i+1][-1]
                    transition_matrix[src_delta, tgt_delta] += 1

            visu(normalize(transition_matrix))

            ipdb.set_trace()
            with open(matrix_file, 'wb') as f:
                pkl.dump(transition_matrix, f)

        else:
            with open(matrix_file, 'rb') as f:
                transition_matrix = pkl.load(f)


def visu(transition_matrix):
    x_start = 0.0
    x_end = 5.0
    y_start = 0.0
    y_end = 5.0
    size = 5

    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111)
    im = ax.imshow(transition_matrix)
    # Add the text
    jump_x = (x_end - x_start) / (4.0 * size)
    jump_y = (y_end - y_start) / (4.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = "%.2f" % transition_matrix[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

    fig.colorbar(im)
    plt.show()


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    print("Building datasets...")
    arguments = {'data_dir': CFG["datadir"],
                 "load": args.load_dataset,
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": 5,
                 "exp_type": "delta",
                 "is_toy": "False"}

    data_train = StringDataset(**arguments, split="TRAIN")
    data_valid = StringDataset(**arguments, split="VALID")
    data_test = StringDataset(**arguments, split="TEST")
    if args.load_dataset == "True":
        with open(os.path.join(CFG["datadir"], "class_weights_dict_dynamics_delta.pkl"), 'rb') as f_name:
            class_weights = pkl.load(f_name)
    else:
        class_weights = get_class_weights(data_train)
        with open(os.path.join(CFG["datadir"], "class_weights_dict_dynamics_delta.pkl"), 'wb') as f_name:
            pkl.dump(class_weights, f_name)
    return data_train, data_train.user_lookup, \
           data_valid, data_valid.user_lookup, \
           data_test, data_test.user_lookup,\
           class_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--load_matrix", type=str, default="False")
    args = parser.parse_args()
    main(args)
