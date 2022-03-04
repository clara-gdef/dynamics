import argparse
import yaml
import os
import joblib
import ipdb
import numpy as np
import pickle as pkl
from collections import Counter

from data.exp_classif.k_means_labelling import get_exp_dict


def init(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Commencing data gathering...')
        with open(os.path.join(CFG["gpudatadir"], f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_dynamics.pkl"), 'rb') as f_name:
            class_weights = pkl.load(f_name)
        print('Data loaded')
        main(args, class_weights)


def main(args, class_dict):
    inspect_experience()
    # inspect_industry()


def inspect_industry():
    with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f_name:
        industry_dict = pkl.load(f_name)
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    vectorizer = joblib.load(f"{tgt_file}_vectorizer.joblib")
    voc = vectorizer.vocabulary_
    rev_voc = {v: k for k, v in voc.items()}
    num_features = len(vectorizer.vocabulary_)
    model_ind = joblib.load(f"{tgt_file}_ind.joblib")
    num_classes = model_ind.coef_.shape[0]
    dec_matrix = model_ind.coef_.T
    # what words will automatically put you in a specific class
    class_determining_words_counter = {}
    for cl in range(num_classes):
        class_determining_words_counter[industry_dict[cl]] = Counter()
    for word in range(dec_matrix.shape[0]):
        cl = np.argmax(dec_matrix[word] + model_ind.intercept_)
        class_determining_words_counter[industry_dict[cl]][rev_voc[word]] = (dec_matrix[word] + model_ind.intercept_)[cl]
    # what words have the strongest coef for a given class
    class_representing_words_counter = {}
    for cl in range(num_classes):
        class_representing_words_counter[industry_dict[cl]] = Counter()
    for class_num in range(dec_matrix.shape[1]):
        tmp = np.argsort(dec_matrix[:, class_num])[::-1]
        words = tmp[:5]
        class_representing_words_counter[industry_dict[class_num]] = [rev_voc[i] for i in words]
    save_word_clouds(class_representing_words_counter, f"{exp_name}_ind")
    # print(f"Vocab: {num_features}")
    # mc_words = dict()
    # for cl in range(num_classes):
    #     mc_words[industry_dict[cl]] = [i[0] for i in class_determining_words_counter[industry_dict[cl]].most_common(5)]
    # print(mc_words)
    # ipdb.set_trace()


def save_word_clouds(word_dict, file_name):
    tgt_file = os.path.join("txt", f"{file_name}.txt")
    if os.path.isfile(tgt_file):
        os.system('rm ' + tgt_file)
    with open(tgt_file, "a") as f:
        for k in word_dict.keys():
            f.write(f"{k}: {word_dict[k]} \n")
    print(f"File saved at {tgt_file}")


def inspect_experience():
    exp_dict = get_exp_dict(args)
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    vectorizer = joblib.load(f"{tgt_file}_vectorizer.joblib")
    voc = vectorizer.vocabulary_
    rev_voc = {v: k for k, v in voc.items()}
    num_features = len(vectorizer.vocabulary_)
    model_exp = joblib.load(f"{tgt_file}_exp.joblib")
    class_counter = {}
    num_classes = model_exp.coef_.shape[0]
    dec_matrix = model_exp.coef_.T
    for cl in range(num_classes):
        class_counter[cl] = Counter()
    for word in range(dec_matrix.shape[0]):
        cl = np.argmax(dec_matrix[word] + model_exp.intercept_)
        class_counter[cl][rev_voc[word]] = (dec_matrix[word] + model_exp.intercept_)[cl]

    class_representing_words_counter = {}
    for cl in range(num_classes):
        class_representing_words_counter[exp_dict[cl]] = Counter()
    for class_num in range(dec_matrix.shape[1]):
        tmp = np.argsort(dec_matrix[:, class_num])[::-1]
        words = tmp[:10]
        class_representing_words_counter[exp_dict[class_num]] = [rev_voc[i] for i in words]

    save_word_clouds(class_representing_words_counter, f"{exp_name}_exp")

    # print(f"EXPERIENCE: {exp_name}")
    # print(f"Vocab: {num_features}")
    # for cl in range(num_classes):
    #     print(f"CLASS {cl}")
    #     mc_words = [i[0] for i in class_counter[cl].most_common(10)]
    #     print(mc_words)
    # exp_indices = [voc[v] for v in exp_dict.values() if v in voc.keys()]
    # tmp = {}
    # for index in exp_indices:
    #     tmp[rev_voc[index]] = np.argmax(dec_matrix[index] + model_exp.intercept_)
    # print(tmp)
    # ipdb.set_trace()


def get_exp_name(args):
    exp_name = f"{args.model}_{args.exp_levels}exp_{args.exp_type}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type=str, default="True")
    parser.add_argument("--min_df", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="SVM")# SVM or NB
    parser.add_argument("--max_df", type=float, default=.6)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--exp_type", type=str, default="uniform") #"kmeans" or "uniform" or "kmeansrdm"
    parser.add_argument("--exp_levels", type=int, default=5) # 5 of 3
    parser.add_argument("--tfidf", default="True")
    args = parser.parse_args()
    init(args)
