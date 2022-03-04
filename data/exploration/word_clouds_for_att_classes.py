import argparse
import yaml
import os
import joblib
import ipdb
import numpy as np
import pickle as pkl
import fasttext
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import matplotlib.pyplot as plt

from data.datasets import StringDataset
from utils.baselines import pre_proc_data


def init(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Commencing data gathering...')
        _, data_valid, data_test, _ = get_labelled_data(args)
        main(args, data_valid, data_test)


def main(args, data_valid, data_test):
    inspect_delta(data_valid, data_test)
    # inspect_experience(data_valid, data_test)
    # inspect_industry(data_valid, data_test)


def inspect_industry(data_valid, data_test):
    with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f_name:
        industry_dict = pkl.load(f_name)
    exp_name = get_exp_name(args)
    model = get_model(args, args.model)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    vectorizer = joblib.load(f"{tgt_file}_vectorizer.joblib")
    class_texts = {k: "" for k in range(len(industry_dict))}
    jobs, _, labels_ind = pre_proc_data(data_train)
    vected_data = vectorizer.transform(jobs)

    save_word_clouds(class_representing_words_counter, f"{exp_name}_ind")



def save_word_clouds(wordcloud, file_name):
    tgt_file = os.path.join("txt", f"{file_name}.txt")
    wordcloud.to_file("img/first_review.png")
    print(f"Word Cloud file saved at {tgt_file}")


def inspect_delta(data_valid, data_test):
    with open(os.path.join(CFG["gpudatadir"], f"delta_dict.pkl"), 'rb') as f_name:
        exp_dict = pkl.load(f_name)
    exp_name = get_exp_name(args)
    class_texts = {k: "" for k in range(len(exp_dict))}
    print("Predicting labels...")
    for person in tqdm(data_valid, desc="sorting predictions by class..."):
        class_texts[person[-1]] += f" {person[0]}"
    print("Samples sorted by label...")
    for i in tqdm(class_texts.keys(), desc="Computing word clouds..."):
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(class_texts[i])
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"img/WORDCLOUD_delta_{exp_name}_class{i}.png")
        wordcloud.to_file(f"img/WORDCLOUD_delta_{exp_name}_class{i}.png")
        plt.close()


def inspect_experience(data_valid, data_test):
    with open(os.path.join(CFG["gpudatadir"], f"exp_dict_fr_{args.exp_levels}.pkl"), 'rb') as f_name:
        exp_dict = pkl.load(f_name)
    exp_name = get_exp_name(args)
    model = get_model(args, args.model)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    vectorizer = joblib.load(f"{tgt_file}_vectorizer.joblib")
    class_texts = {k: "" for k in range(len(exp_dict))}
    jobs, _, _ = pre_proc_data(data_valid)
    vected_data = vectorizer.transform(jobs)
    print("Predicting labels...")
    predictions = np.argmax(model.decision_function(vected_data), axis=-1)
    for num, label in tqdm(enumerate(predictions[:30000]), desc="sorting predictions by class..."):
        class_texts[label] += f" {data_valid[num][0]}"
    print("Samples sorted by label...")
    for i in tqdm(class_texts.keys(), desc="Computing word clouds..."):
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(class_texts[i])
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"img/WORDCLOUD_{exp_name}_class{i}.png")
        wordcloud.to_file(f"img/WORDCLOUD_{exp_name}_class{i}.png")
        plt.close()

def get_exp_name(args):
    exp_name = f"{args.model}_{args.exp_levels}exp_{args.exp_type}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    print("Building datasets...")
    arguments = {'data_dir': CFG["gpudatadir"],
                     "load": args.load_dataset,
                     "subsample": -1,
                     "max_len": 10,
                     "exp_levels": 5,
                     "exp_type": args.exp_type,
                     "is_toy": "False"}
    data_train = StringDataset(**arguments, split="TRAIN")
    data_valid = StringDataset(**arguments, split="VALID")
    data_test = StringDataset(**arguments, split="TEST")

    with open(os.path.join(CFG["gpudatadir"], "class_weights_dict_dynamics.pkl"), 'rb') as f_name:
        class_weights = pkl.load(f_name)

    return data_train, data_valid, data_test, class_weights


def get_model(args, model_type):
    tgt_file = os.path.join(CFG["modeldir"], get_exp_name(args))
    if model_type == "SVM":
        model = joblib.load(f"{tgt_file}_{args.att_type}.joblib")
    elif model_type == "FT":
        if {args.att_type} == "ind":
            model = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_ind_10.bin")
        else:
            model = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_{args.exp_type}_{args.exp_levels}_10.bin")
    else:
        raise Exception("Wrong type of model specified, can be SVM of FT")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--min_df", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="FT")# SVM or FT
    parser.add_argument("--max_df", type=float, default=.6)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--att_type", type=str, default="delta") #"ind" or "exp" or "delta"
    parser.add_argument("--exp_type", type=str, default="uniform") #"kmeans" or "uniform" or "kmeansrdm"
    parser.add_argument("--exp_levels", type=int, default=5) # 5 of 3
    parser.add_argument("--tfidf", default="True")
    args = parser.parse_args()
    init(args)
