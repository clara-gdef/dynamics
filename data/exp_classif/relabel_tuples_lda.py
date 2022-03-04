import argparse
import yaml
import os
import joblib
import ipdb
import numpy
import pickle as pkl
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA

from data.datasets import StringIndSubDataset
from utils.baselines import fit_vectorizer, get_labelled_data, pre_proc_data, get_class_weights
from itertools import chain
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        print('Commencing data gathering...')
        data_train, data_valid, data_test, class_weights, suffix = get_data(args, CFG)
        print('Data loaded')
        main(args, data_train, data_valid, data_test, class_weights, suffix)

def main(args, data_train, data_valid, data_test, class_dict, suffix):
    # TRAIN
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    model_exp = joblib.load(f"{tgt_file}_exp.joblib")
    vectorizer = joblib.load(f"{tgt_file}_vectorizer_svc_{args.kernel}.joblib")

    relabel_for_split(vectorizer, model_exp, data_train, "train")
    relabel_for_split(vectorizer, model_exp, data_valid, "valid")
    relabel_for_split(vectorizer, model_exp, data_test, "test")

    # TEST

    ipdb.set_trace()


def relabel_for_split(vectorizer, model, data, split):
    cleaned_profiles, labels_exp_train, labels_ind_train = pre_proc_data(data)
    train_features = vectorizer.transform(args, cleaned_profiles)
    transformed_jobs = model.transform(train_features[:args.subsample])
    for num, user in enumerate(tqdm(data.user_lookup.keys(), desc=f"relabelling profiles for {split} split")):
        ipdb.set_trace()



def get_exp_name(args):
    exp_name = f"LDA_{args.exp_levels}exp_{args.exp_type}_bs{args.b_size}_maxiter{args.epochs}"
    if args.subsample != -1:
        exp_name += f"_sub{args.subsample}"
    if args.max_len != 10:
        exp_name += f"_maxlen{args.max_len}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


def get_data(args, CFG):
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
        dataset_train = StringIndSubDataset(**arguments, split="TRAIN")
        class_weights = get_class_weights(dataset_train)
        dataset_valid = StringIndSubDataset(**arguments, split="VALID")
        dataset_test = StringIndSubDataset(**arguments, split="TEST")
        suffix = f"_subInd{args.suffix}"
    return dataset_train, dataset_valid, dataset_test, class_weights, suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type=str, default="False")
    parser.add_argument("--min_df", type=float, default=1e-4)
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--custom_centers", type=str, default="False")
    parser.add_argument("--load_model", type=str, default="False")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="uniform")  # "kmeans" or "uniform" or "kmeansrdm" or "delta"
    parser.add_argument("--exp_levels", type=int, default=3)  # 5 of 3
    parser.add_argument("--tfidf", default="True")
    args = parser.parse_args()
    init(args)
