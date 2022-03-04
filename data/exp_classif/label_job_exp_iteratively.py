import ipdb
import argparse
import yaml
import joblib
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from data.datasets import StringIndSubDataset, StringDataset
from utils import get_metrics
from utils.baselines import train_svm, pre_proc_data
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import chain


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(args)
    else:
        return main(args)


def main(args):
    data_train_valid, data_test, (len_train, len_valid), train_lookup, valid_lookup, test_lookup = load_datasets(args)
    if args.initial_check == "True":
        check_monotonic_dynamic(data_train_valid, train_lookup, "train")
        check_monotonic_dynamic(data_train_valid, valid_lookup, "valid")
        check_monotonic_dynamic(data_test, test_lookup, "test")

    cleaned_profiles, labels_exp_train, _ = pre_proc_data(data_train_valid)
    cleaned_profiles_test, labels_exp_test, _ = pre_proc_data(data_test)
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=.8, min_df=1e-4, max_features=50000)
    print("Fitting vectorizer...")
    train_features = vectorizer.fit_transform(cleaned_profiles)
    print("Vectorizer Fitted.")

    test_features = vectorizer.transform(cleaned_profiles_test)
    iteration = args.start_iter
    f1 = 0
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    # all_users = {**train_lookup, **valid_lookup, **test_lookup}
    offset = len(data_train_valid)
    offset_test_lookup = {}
    for k, v in test_lookup.items():
        offset_test_lookup[k] = [v[0] + offset, v[1] + offset]
    assert v[1] + offset <= train_features.shape[0] + test_features.shape[0]
    all_users = {**train_lookup, **offset_test_lookup}

    check_monotonic_dynamic(data_train_valid + data_test, all_users, "all")
    print(f"Concatenating all {train_features.shape[0] + test_features.shape[0]} features...")
    all_features = np.concatenate((train_features.toarray(), test_features.toarray()), axis=0)
    all_labels = labels_exp_train
    all_labels.extend(labels_exp_test)
    print("Features and labels concatenated.")
    while f1 < args.f1_threshold and iteration < args.max_iter:
        print(f"Iteration number: {iteration}")
        # svc training
        subset_data, subset_labels, user_trains = get_subset_data_and_labels(train_features.toarray(), labels_exp_train,
                                                                             data_train_valid.user_lookup,
                                                                             args.train_user_len)
        print(f"Training classifier on {len(subset_data)} jobs...")
        class_weigths = get_class_dist(subset_labels)
        classifier = train_svm(subset_data, subset_labels, class_weigths, args.kernel)
        joblib.dump(classifier, f"{tgt_file}_exp_svc_{args.kernel}_it{iteration}.joblib")
        print(f"Classifier saved at: {tgt_file}_exp_svc_{args.kernel}_it{iteration}.joblib")
        preds, labels = [], []
        # SVC eval
        for cnt, user in enumerate(tqdm(all_users.keys(), desc="parsing users...")):
            if user not in user_trains and cnt % args.user_step == 0:
                current_user = all_users[user]
                exp_seq_pred = [all_labels[current_user[0]]]
                exp_seq_init = [all_labels[current_user[0]]]
                for job in range(current_user[0] + 1, current_user[1]):
                    prev_exp = all_labels[job - 1]
                    if job == current_user[1]-1:
                        next_exp = 2 # max possible
                    else:
                        next_exp = all_labels[job + 1]
                    tmp = classifier.decision_function(all_features[job].reshape(1, -1))[0]
                    pred = np.argsort(tmp)[::-1][0]
                    # relabel current tuple according to evolution constraint
                    exp_seq_init.append(all_labels[job])
                    if prev_exp <= pred <= next_exp:
                        all_labels[job] = pred
                    # keep predictions for evaluation of convergence
                    preds.append(pred)
                    labels.append(all_labels[job])
                    exp_seq_pred.append(all_labels[job])
                assert all(exp_seq_pred[i] <= exp_seq_pred[i+1] for i in range(len(exp_seq_pred)-1))
                assert all(exp_seq_init[i] <= exp_seq_init[i+1] for i in range(len(exp_seq_init)-1))
        metrics = get_metrics(preds, labels, args.exp_levels, f"it_{iteration}")
        f1 = metrics[f"f1_it_{iteration}"]
        print(f"Iteration: {iteration}, F1 score: {f1}%")
        pred_class_dist = get_class_dist(preds)
        label_class_dist = get_class_dist(labels)
        print(f"Class distributions in PREDS: {pred_class_dist}")
        print(f"Class distributions in LABELS: {label_class_dist}")
        iteration += 1
        word_analysis(args, all_features, all_labels, vectorizer, exp_name, iteration)
        save_new_tuples(data_train_valid, data_test, all_labels, len_train, len_valid, train_lookup, valid_lookup,
                        test_lookup, iteration)

    ipdb.set_trace()
    # the model converged, we test it on the whole dataset
    all_results = test_model_on_all_test_data(args, classifier, vectorizer)


def check_monotonic_dynamic(data, lookup, split):
    if split == "all":
        all_labels = [i[-1] for i in data]
    else:
        all_labels = [i["exp_index"] for i in data.tuples]
    for cnt, user in enumerate(tqdm(lookup.keys(), desc=f"initial monotonicity check on split {split} users...")):
        current_user = lookup[user]
        exp_seq_init = [all_labels[current_user[0]]]
        for job in range(current_user[0] + 1, current_user[1]):
            exp_seq_init.append(all_labels[job])
        assert all(exp_seq_init[i] <= exp_seq_init[i + 1] for i in range(len(exp_seq_init) - 1))


def word_analysis(args, all_features, all_labels, vectorizer, exp_name, iteration):
    class_txt = get_jobs_str_per_class(args, all_features, all_labels, vectorizer)
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
        get_word_clouds_for_class(word_counter[k], word_list, k, exp_name + f"_it{iteration}")
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


def get_jobs_str_per_class(args, all_features, all_labels, vectorizer):
    # sort job id per class
    class_index = {k: [] for k in range(args.exp_levels)}
    for num, _ in enumerate(tqdm(all_labels)):
        class_index[all_labels[num]].append(num)
    rev_class_index = {}
    for k, v in tqdm(class_index.items(), desc="Building reversed index for topics..."):
        for job_index in v:
            rev_class_index[job_index] = k
    rev_voc = {v: k for k, v in vectorizer.vocabulary_.items()}
    class_txt = {k: "" for k in range(args.exp_levels)}
    for num, feat in enumerate(tqdm(all_features, desc="sorting jobs by class on all features...")):
        defining_index = np.argsort(feat)[::-1][0]
        class_txt[rev_class_index[num]] += f" {rev_voc[defining_index]}"
    return class_txt


def save_new_tuples(data_train_valid, data_test, all_labels, len_train, len_valid, train_lookup,
                    valid_lookup, test_lookup, iteration):

    tuples_train = data_train_valid.tuples[:len_train]
    labels_train = all_labels[:len_train]
    for num, label in enumerate(tqdm(labels_train, desc="relabel train tuples...")):
        tuples_train[num]["exp_index"] = label

    tuples_valid = data_train_valid.tuples[len_train:]
    labels_valid = all_labels[len_train:len_train+len_valid]
    for num, label in enumerate(tqdm(labels_valid, desc="relabel valid tuples...")):
        tuples_valid[num]["exp_index"] = label

    offset = len_train
    reset_valid_lookup = {}
    if min(min(valid_lookup.values())) >= offset:
        for k, v in valid_lookup.items():
            assert v[0]-offset >= 0
            reset_valid_lookup[k] = [v[0]-offset, v[1]-offset]
    else:
        reset_valid_lookup = valid_lookup

    labels_test = all_labels[len_train+len_valid:-1]
    tuples_test = []
    for num, label in enumerate(tqdm(labels_test, desc="relabel test tuples...")):
        new = {"ind_index": data_test.tuples[num]["ind_index"],
               "exp_index": label,
               "words": data_test.tuples[num]["words"]}
        tuples_test.append(new)

    offset = len_train + len_valid
    reset_test_lookup = {}
    if min(min(test_lookup.values())) >= offset:
        for k, v in test_lookup.items():
            assert v[0]-offset >= 0
            reset_test_lookup[k] = [v[0]-offset, v[1]-offset]
    else:
        reset_test_lookup = test_lookup
    save_new_tuples_per_split(tuples_train, train_lookup, "TRAIN", iteration)
    save_new_tuples_per_split(tuples_valid, reset_valid_lookup, "VALID", iteration)
    save_new_tuples_per_split(tuples_test, reset_test_lookup, "TEST", iteration)


def save_new_tuples_per_split(tuple_list, lookup, split, iteration):
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "False",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "tuple_list": tuple_list,
                 'lookup': lookup,
                 "already_subbed": "True",
                 "exp_type": args.exp_type,
                 "suffix": f"_it{iteration}",
                 "split": split,
                 "is_toy": "False"}
    StringIndSubDataset(**arguments)


def test_model_on_all_test_data(args, model, vectorizer):
    args.subsample_jobs = 10000
    data_train, data_test, (len_train, len_valid) = load_datasets(args)
    cleaned_profiles_test, labels_exp_test, _ = pre_proc_data(data_test)
    test_features = vectorizer.transform(cleaned_profiles_test)
    preds = model.predict(test_features.toarray())
    assert len(preds) == len(labels_exp_test)
    metrics = get_metrics(preds, labels_exp_test, args.exp_levels, f"whole")
    print(metrics)
    ipdb.set_trace()
    return metrics


def get_class_dist(class_list):
    cnt = Counter()
    total = len(class_list)
    for i in class_list:
        cnt[i] += 100 / total
    return cnt


def load_datasets(args):
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    if args.ind_sub == "True":
        if args.start_iter == 0:
            suffix = ""
        else:
            suffix = f"_it{args.start_iter}"
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": args.load_dataset,
                     "subsample": args.subsample_jobs,
                     "max_len": 10,
                     "exp_levels": args.exp_levels,
                     "tuple_list": None,
                     "already_subbed": "True",
                     'lookup': None,
                     "suffix": suffix,
                     "exp_type": args.exp_type,
                     "is_toy": "False"}
        for split in splits:
            datasets.append(StringIndSubDataset(**arguments, split=split))
    else:
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": args.load_dataset,
                     "subsample": args.subsample_jobs,
                     "max_len": 10,
                     "exp_levels": args.exp_levels,
                     "exp_type": args.exp_type,
                     "is_toy": "False"}
        for split in splits:
            datasets.append(StringDataset(**arguments, split=split))
    len_train = len(datasets[0])
    len_valid = len(datasets[1])
    init_train_lookup = datasets[0].user_lookup
    init_valid_lookup = datasets[1].user_lookup
    init_test_lookup = datasets[-1].user_lookup

    offset = len(datasets[0])
    offset_valid_lookup = {}
    for k, v in init_valid_lookup.items():
        offset_valid_lookup[k] = [v[0] + offset, v[1] + offset]
    datasets[1].user_lookup = offset_valid_lookup

    data_train_valid = datasets[0]
    data_train_valid.tuples.extend(datasets[1].tuples)

    train_lookup_sub = subsample_user_lookup(args, datasets[0])
    valid_lookup_sub = subsample_user_lookup(args, datasets[1])
    test_lookup_sub = subsample_user_lookup(args, datasets[-1])

    data_train_valid.user_lookup = {**train_lookup_sub, **valid_lookup_sub}
    data_train_valid.check_monotonicity()
    datasets[-1].user_lookup = test_lookup_sub

    # len_valid = len(datasets[1])
    # assert len(data_train_valid) == len_train + len_valid
    return data_train_valid, datasets[-1], (len_train, len_valid), \
           init_train_lookup, offset_valid_lookup, init_test_lookup


def get_subset_data_and_labels(features, labels, user_lookup, train_user_len):
    user_id_as_list = [i for i, _ in user_lookup.items()]
    user_train = []
    arr = np.arange(train_user_len)
    np.random.shuffle(arr)
    sub_data, sub_labels = [], []
    for i in arr:
        user_train.append(user_id_as_list[i])
        current_user = user_lookup[user_id_as_list[i]]
        for job_num in range(current_user[0], current_user[1]):
            sub_data.append(features[job_num])
            sub_labels.append(labels[job_num])
    assert len(sub_data) == len(sub_labels)
    return sub_data, sub_labels, user_train


def get_exp_name(args):
    exp_name = f"label_iter_{args.exp_levels}exp_{args.exp_type}_train{args.train_user_len}"
    if args.subsample_users != -1:
        exp_name += f"_eval{args.subsample_users}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


def subsample_user_lookup(args, datasets):
    if args.subsample_users != -1:
        usr_num = args.subsample_users
        tmp_user_lu = {}
        i = 0
        for k, v in datasets.user_lookup.items():
            if i < usr_num:
                tmp_user_lu[k] = v
                i += 1
            else:
                break
        print(f"Subsampling {usr_num} users, returning lookup of length: {len(tmp_user_lu)}")
        return tmp_user_lu
    else:
        print(f"No subsampling of users, returning full lookup of length: {len(datasets.user_lookup)}")
        return datasets.user_lookup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample_users", type=int, default=-1)
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--subsample_jobs", type=int, default=-1)
    parser.add_argument("--train_user_len", type=int, default=1000)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--user_step", type=int, default=10)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--f1_threshold", type=int, default=80)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--ind_sub", type=str, default="True")
    parser.add_argument("--initial_check", type=str, default="False")
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--tfidf", type=str, default="True")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
