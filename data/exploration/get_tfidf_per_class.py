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
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from scipy import sparse

from data.datasets import StringDataset
from data.exploration import vectorize_data
from utils.baselines import pre_proc_data


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        exp_title = f"{args.exp_type}_{args.exp_levels}exp_res_{args.mod_type}"
        if args.tfidf_vect == "True":
            exp_title += "_tfidf"
        if args.load_tfidf == "True":
            tfidf_file = os.path.join(CFG["gpudatadir"], f"tfidf_per_class_{exp_title}.pkl")
            with open(tfidf_file, 'rb') as f:
                tmp = pkl.load(f)
            ordered_tfidfs, vectorizer_exp = tmp[0], tmp[1]
        else:
            ordered_tfidfs, vectorizer_exp = get_tfidf_per_class(args)
        mc_ten = {}
        rev_voc = {v: k for k, v in vectorizer_exp.vocabulary_.items()}
        for class_num in range(args.exp_levels):
            mc_ten[class_num] = []
            for tup in ordered_tfidfs[class_num][:10]:
                word = rev_voc[tup[0]]
                mc_ten[class_num].append((word, tup[1]))

        # try with the sklearn tfidf
        corpus_vectorized = get_tfidf_for_corpus()
        tfidf_sklearn = dict()
        for class_num in range(args.exp_levels):
            tmp = []
            for num_job in tqdm(samples_per_class[class_num], desc=f"getting job for class {class_num}"):
                tmp.append(corpus_vectorized[num_job].toarray())
            tfidf_sklearn[class_num] = np.mean(np.stack(tmp), axis=0)[0]


def get_df(feat_exp_train, feat_exp_valid, feat_exp_test):
    corpus = sparse.vstack([feat_exp_train, feat_exp_valid, feat_exp_test]).toarray()
    tfid = TfidfTransformer()
    print(f"Fitting TfidfTransformer on {corpus.shape[0]} samples...")
    tfid.fit(corpus)
    print("TfidfTransformer fitted")
    return tfid.idf_, corpus


def get_tfidf_for_corpus():
    data_train, train_lookup, data_valid, valid_lookup, data_test, test_lookup, class_weights = get_labelled_data(
        args)
    vectorizer = TfidfVectorizer(max_features=2345)
    pre_proc_data_train, labels_exp_train, _ = pre_proc_data(data_train)
    pre_proc_data_valid, labels_exp_valid, _ = pre_proc_data(data_valid)
    pre_proc_data_test, labels_exp_test, _ = pre_proc_data(data_test)
    corpus = pre_proc_data_train
    corpus.extend(pre_proc_data_valid)
    corpus.extend(pre_proc_data_test)
    features_exp = vectorizer.fit_transform(corpus)
    return features_exp


def get_tfidf_per_class(args):
    data_train, train_lookup, data_valid, valid_lookup, data_test, test_lookup, class_weights = get_labelled_data(
        args)
    vect_file = f"{args.mod_type}_{args.exp_levels}exp_{args.exp_type}"
    if args.tfidf_vect == "True":
        vect_file += "_tfidf"
    vectorizer_exp = joblib.load(f"/data/gainondefor/dynamics/{vect_file}_vectorizer.joblib")
    feat_exp_train, labels_exp_train = vectorize_data(data_train, vectorizer_exp)
    feat_exp_valid, labels_exp_valid = vectorize_data(data_valid, vectorizer_exp)
    feat_exp_test, labels_exp_test = vectorize_data(data_test, vectorizer_exp)
    idf, corpus = get_df(feat_exp_train, feat_exp_valid, feat_exp_test)
    samples_per_class = sort_samples_per_class(args.exp_levels, labels_exp_train, labels_exp_valid, labels_exp_test)
    term_count_per_class = dict()
    for class_num in range(args.exp_levels):
        term_count_per_class[class_num] = Counter()
        for num_job in tqdm(samples_per_class[class_num], desc=f"getting term frqcy for class {class_num}"):
            for word, count in enumerate(corpus[num_job]):
                term_count_per_class[class_num][word] += count
    compare_count_dicts(term_count_per_class)
    # compute freq
    term_frequency_per_class = dict()
    for class_num in range(args.exp_levels):
        term_frequency_per_class[class_num] = dict()
        total_words = sum(term_count_per_class[class_num])
        for k, v in term_count_per_class[class_num].items():
            term_frequency_per_class[class_num][k] = v / total_words
    tf_file = os.path.join(CFG["gpudatadir"], "tf_per_class.pkl")
    with open(tf_file, 'wb') as f:
        pkl.dump(term_frequency_per_class, f)

    tfidf_per_class = dict()
    ordered_tfidfs = []
    for class_num in range(args.exp_levels):
        tfidf_per_class[class_num] = {}
        for k, v in tqdm(term_frequency_per_class[class_num].items(),
                         desc=f"computing tfidf for class {class_num}"):
            tfidf_per_class[class_num][k] = v * idf[k]
        ordered_tfidfs.append(sorted(tfidf_per_class[class_num].items(), key=itemgetter(1), reverse=True))
    tfidf_file = os.path.join(CFG["gpudatadir"], "tfidf_per_class.pkl")
    with open(tfidf_file, 'wb') as f:
        pkl.dump((ordered_tfidfs, vectorizer_exp), f)
    return ordered_tfidfs, vectorizer_exp


def compare_count_dicts(term_count_per_class):
    tcpc_dict = {}
    for num_class, tcpc in term_count_per_class.items():
        tcpc_list = []
        for k in sorted(tcpc.keys()):
            tcpc_list.append(tcpc[k])
        tcpc_dict[num_class] = np.array(tcpc_list)


def sort_samples_per_class(num_classes, labels_exp_train, labels_exp_valid, labels_exp_test):
    samples_per_class = {k: [] for k in range(num_classes)}
    for class_num in tqdm(range(num_classes)):
        counter = 0
        for sample in labels_exp_train:
            if sample == class_num:
                samples_per_class[class_num].append(counter)
            counter += 1
        for sample in labels_exp_valid:
            if sample == class_num:
                samples_per_class[class_num].append(counter)
            counter += 1
        for sample in labels_exp_test:
            if sample == class_num:
                samples_per_class[class_num].append(counter)
            counter += 1
    return samples_per_class


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    print("Loading datasets...")
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    data_train = StringDataset(**arguments, split="TRAIN")
    data_valid = StringDataset(**arguments, split="VALID")
    data_test = StringDataset(**arguments, split="TEST")

    with open(os.path.join(CFG["gpudatadir"],
                           f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_dynamics.pkl"),
              'rb') as f_name:
        class_weights = pkl.load(f_name)

    return data_train, data_train.user_lookup, \
           data_valid, data_valid.user_lookup, \
           data_test, data_test.user_lookup,\
           class_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod_type", type=str, default="SVM")
    parser.add_argument("--load_tfidf", type=str, default="False")
    parser.add_argument("--exp_levels", type=int, default=5)
    parser.add_argument("--tfidf_vect", default="True")
    parser.add_argument("--exp_type", type=str, default="kmeans")
    args = parser.parse_args()
    main(args)