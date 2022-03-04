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
from utils.baselines import fit_vectorizer, get_labelled_data, pre_proc_data
from itertools import chain
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter


def init(args):
    with ipdb.launch_ipdb_on_exception():
        print('Commencing data gathering...')
        data_train, data_test, class_weights = get_labelled_data(args)
        print('Data loaded')
        main(args, data_train, data_test, class_weights)


def main(args, data_train, data_test, class_dict):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    # TRAIN
    if args.subsample > 0:
        data_train, data_test = subsample_according_to_class_weight(args, data_train, data_test, class_dict,
                                                                    args.subsample)
    cleaned_profiles, labels_exp_train, labels_ind_train = pre_proc_data(data_train)
    train_features, vectorizer = fit_vectorizer(args, cleaned_profiles)
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    if args.load_model == "True":
        model_exp = joblib.load(f"{tgt_file}_exp.joblib")
        vectorizer = joblib.load(f"{tgt_file}_vectorizer_svc_{args.kernel}.joblib")
        transformed_jobs_train = model_exp.transform(train_features[:args.subsample])
    else:
        model_exp = LDA(n_components=args.exp_levels, doc_topic_prior=None,
                        topic_word_prior=None, learning_method='batch',
                        learning_decay=0.7, learning_offset=10.0, max_iter=args.epochs,
                        batch_size=args.b_size, evaluate_every=10,
                        total_samples=1000000.0, perp_tol=0.1,
                        mean_change_tol=0.001, max_doc_update_iter=100,
                        n_jobs=None, verbose=1, random_state=7)
        transformed_jobs_train = model_exp.fit_transform(train_features[:args.subsample])
        if args.save_model == "True":
            joblib.dump(model_exp, f"{tgt_file}_exp.joblib")
            joblib.dump(vectorizer, f"{tgt_file}_vectorizer.joblib")
    # TEST
    cleaned_abstracts_test, labels_exp_test, labels_ind_test = pre_proc_data(data_test)
    test_features = vectorizer.transform(cleaned_abstracts_test)
    transformed_jobs_test = model_exp.transform(test_features[:args.subsample])
    # sort the jobs per majority topic
    topic_index = {k: [] for k in range(args.exp_levels)}
    for num, job in enumerate(tqdm(chain(transformed_jobs_train, transformed_jobs_test), desc="relabelling according to lda...")):
        pred_topic = np.argmax(job, axis=-1)
        topic_index[pred_topic].append(num)
    rev_topic_index = {}
    for k, v in tqdm(topic_index.items(), desc="Building reversed index for topics..."):
        for job_index in v:
            rev_topic_index[job_index] = k
    total_jobs = sum([len(topic_index[i]) for i in topic_index.keys()])
    topic_dist = {k: 100*len(v)/total_jobs for k, v in topic_index.items()}
    print(topic_dist)
    rev_voc = {v: k for k, v in vectorizer.vocabulary_.items()}
    topic_txt = {k: "" for k in range(args.exp_levels)}
    for index, feat in enumerate(tqdm(train_features, desc="sorting jobs by class on train...")):
        topic_txt[rev_topic_index[index]] += f" {get_defining_words(feat, rev_voc)}"
    for index, feat in enumerate(tqdm(test_features, desc="sorting jobs by class on test...")):
        topic_txt[rev_topic_index[index]] += f" {get_defining_words(feat, rev_voc)}"

    word_counter = {k: Counter() for k in range(args.exp_levels)}
    for k in tqdm(word_counter.keys(), desc="counting words per class"):
        for word in topic_txt[k].split(" "):
            word_counter[k][word] += 1
    mc_words_per_class = {}
    for k in word_counter.keys():
        mc_words_per_class[k] = set([i[0] for i in word_counter[k].most_common(50)])

    unique_words_per_topic = get_unique_words_per_class(mc_words_per_class)
    for k in mc_words_per_class.keys():
        get_word_clouds_for_class(word_counter[k], unique_words_per_topic[k], k, exp_name + f"_topic{k}")
    ipdb.set_trace()
    # for index, person in tqdm(enumerate(data_train), desc="sorting jobs by class on train..."):
    #     topic_txt[rev_topic_index[index]] += f" {person[0]}"
    # for index, person in tqdm(enumerate(data_test), desc="sorting jobs by class on test..."):
    #     topic_txt[rev_topic_index[index]] += f" {person[0]}"
    #get_word_clouds(topic_txt, exp_name)


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


def get_unique_words_per_class(mc_words_per_class):
    unique_words_per_class = {}
    for k in mc_words_per_class.keys():
        unique_words_per_class[k] = mc_words_per_class[k]
        for k2 in mc_words_per_class.keys():
            if k2 != k:
                unique_words_per_class[k] -= mc_words_per_class[k2]
    return unique_words_per_class


def get_defining_words(features, rev_voc):
    defining_index = np.argsort(features)[::-1][0]
    return rev_voc[defining_index]


def get_word_clouds(class_texts, exp_name):
    print("Samples sorted by label...")
    for i in tqdm(class_texts.keys(), desc="Computing word clouds..."):
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(class_texts[i])
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"img/WORDCLOUD_defining_words_{exp_name}_class{i}.png")
        wordcloud.to_file(f"img/WORDCLOUD_defining_words_{exp_name}_class{i}.png")
        plt.close()


def subsample_according_to_class_weight(args, data_train, data_test, class_dict, subsample):
    tgt_file = os.path.join(CFG["gpudatadir"], f"subsampled_data_{subsample}_{args.exp_type}_{args.exp_levels}exp.pkl")
    if os.path.isfile(tgt_file):
        with open(tgt_file, "rb") as f:
            dat = pkl.load(f)
        sub_train = dat["train"]
        sub_test = dat["test"]
    else:
        ind_weights = class_dict["ind"]
        num_sample_per_class = {k: int(v * subsample) for k, v in ind_weights.items()}
        sub_train, sub_test = [], []
        sampled_index = []
        for k in tqdm(num_sample_per_class.keys(), desc="Subsampling train data according to industry class ratio"):
            class_counter = 0
            while class_counter < num_sample_per_class[k]:
                rdm_index = numpy.random.randint(len(data_train))
                if (data_train[rdm_index][1] == k) and (rdm_index not in sampled_index):
                    sub_train.append(data_train[rdm_index])
                    class_counter += 1
                    sampled_index.append(rdm_index)
        sampled_index = []
        for k in tqdm(num_sample_per_class.keys(), desc="Subsampling test data according to industry class ratio"):
            class_counter = 0
            while class_counter < num_sample_per_class[k]:
                rdm_index = numpy.random.randint(len(data_test))
                if (data_test[rdm_index][1] == k) and (rdm_index not in sampled_index):
                    sub_test.append(data_test[rdm_index])
                    class_counter += 1
                    sampled_index.append(rdm_index)
        tmp = {"train": sub_train, "test": sub_test}
        with open(tgt_file, "wb") as f:
            pkl.dump(tmp, f)
    return sub_train, sub_test


def get_exp_name(args):
    exp_name = f"LDA_{args.exp_levels}exp_{args.exp_type}_bs{args.b_size}_maxiter{args.epochs}"
    if args.subsample != -1:
        exp_name += f"_sub{args.subsample}"
    if args.max_len != 10:
        exp_name += f"_maxlen{args.max_len}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


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
