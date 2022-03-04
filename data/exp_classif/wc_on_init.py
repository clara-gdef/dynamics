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
    exp_name = get_exp_name(args)
    class_txt = {k: "" for k in range(args.exp_levels)}
    class_txt = get_jobs_str_per_class(class_txt, data_train, "train")
    class_txt = get_jobs_str_per_class(class_txt, data_valid, 'valid')
    class_txt = get_jobs_str_per_class(class_txt, data_test, "test")
    word_analysis(args, class_txt, exp_name)
    ipdb.set_trace()


def get_jobs_str_per_class(class_txt, dataset, split):
    # sort job id per class
    for job in tqdm(dataset.tuples, desc=f'getting words per class for split {split}'):
        ipdb.set_trace()
        lab = job["exp_index"]
        txt = job["words"]
        class_txt[lab] += f" {txt}"
    return class_txt


def word_analysis(args, class_txt, exp_name):
    word_counter = {k: Counter() for k in range(args.exp_levels)}
    for k in tqdm(word_counter.keys(), desc="counting words per class"):
        for word in class_txt[k].split(" "):
            word_counter[k][word] += 1
    mc_words_per_class = {}
    # min_len = min([len(word_counter[k] for k in range(args.exp_levels))])
    for k in word_counter.keys():
        mc_words_per_class[k] = set([i[0] for i in word_counter[k].most_common(500)])
    words_unique_to_class0 = mc_words_per_class[0] - mc_words_per_class[1] - mc_words_per_class[2]
    words_unique_to_class1 = mc_words_per_class[1] - mc_words_per_class[0] - mc_words_per_class[2]
    words_unique_to_class2 = mc_words_per_class[2] - mc_words_per_class[1] - mc_words_per_class[0]
    tmp = [words_unique_to_class0, words_unique_to_class1, words_unique_to_class2]
    for k, word_list in zip(mc_words_per_class.keys(), tmp):
        get_word_clouds_for_class(word_counter[k], word_list, k, exp_name)
    print("word clouds saved!")


def get_word_clouds_for_class(class_text, words_to_print, class_num, exp_name):
    string = ""
    for word, count in class_text.items():
        if word in words_to_print:
            for i in range(count):
                string += f" {word}"
    print("Samples sorted by label...")
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", collocations=False).generate(
        string)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"img/WORDCLOUD_{exp_name}_class{class_num}.png")
    wordcloud.to_file(f"img/WORDCLOUD_{exp_name}_class{class_num}.png")
    plt.close()



def get_exp_name(args):
    exp_name = f"init_{args.exp_levels}exp_{args.exp_type}"
    if args.subsample != -1:
        exp_name += f"_sub{args.subsample}"
    if args.max_len != 10:
        exp_name += f"_maxlen{args.max_len}"
    return exp_name


def get_data(args, CFG):
    suffix = ""
    print("Building datasets...")
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": args.load_data,
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
    parser.add_argument("--load_data", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="uniform")  # "kmeans" or "uniform" or "kmeansrdm" or "delta"
    parser.add_argument("--exp_levels", type=int, default=3)  # 5 of 3
    args = parser.parse_args()
    init(args)
