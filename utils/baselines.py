import os
import yaml
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
import ipdb
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from data.datasets import StringDataset, StringIndSubDataset


def get_labelled_data(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    file_root = f"txt_job_titles_labelled_{args.exp_type}_{args.exp_levels}exp"
    if args.load_data == "True":
        print("Loading data...")
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TRAIN.pkl"), 'rb') as f_name:
            data_train = pkl.load(f_name)
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TEST.pkl"), 'rb') as f_name:
            data_test = pkl.load(f_name)
        with open(os.path.join(CFG["gpudatadir"], f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_dynamics.pkl"), 'rb') as f_name:
            class_weights = pkl.load(f_name)
    else:
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
            data_train = StringIndSubDataset(**arguments, split="TRAIN")
            class_weights = get_class_weights(data_train)
            dataset_valid = StringIndSubDataset(**arguments, split="VALID")
            dataset_test = StringIndSubDataset(**arguments, split="TEST")
            data_test = dataset_valid
            data_test.tuples.extend(dataset_test.tuples)
            suffix = f"_subInd{args.suffix}"
        else:
            if args.custom_centers == "True":
                data_train, data_test, class_weights = load_custom_center_data(arguments, args)
            else:
                data_train = StringDataset(**arguments, split="TRAIN")
                class_weights = get_class_weights(data_train)
                dataset_valid = StringDataset(**arguments, split="VALID")
                dataset_test = StringDataset(**arguments, split="TEST")
                data_test = dataset_valid
                data_test.tuples.extend(dataset_test.tuples)

        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TRAIN.pkl"), 'wb') as f_name:
            pkl.dump(data_train, f_name)
        with open(os.path.join(CFG["gpudatadir"], f"{file_root}_TEST.pkl"), 'wb') as f_name:
            pkl.dump(data_test, f_name)
        with open(os.path.join(CFG["gpudatadir"], f"class_weights_dict_{args.exp_type}_{args.exp_levels}exp_dynamics{suffix}.pkl"), 'wb') as f_name:
            pkl.dump(class_weights, f_name)
    return data_train, data_test, class_weights


def load_custom_center_data(arguments, args):
    data_train = StringDataset(**arguments, split="TRAIN")
    suffix = f"kmeans_{args.exp_levels}custom_centers"
    data_train.load_dataset(args.subsample, "TRAIN", suffix)
    class_weights = get_class_weights(data_train)
    dataset_valid = StringDataset(**arguments, split="VALID")
    dataset_valid.load_dataset(args.subsample, "VALID", suffix)
    dataset_test = StringDataset(**arguments, split="TEST")
    dataset_test.load_dataset(args.subsample, "TEST", suffix)
    data_test = dataset_valid
    data_test.tuples.extend(dataset_test.tuples)
    return data_train, data_test, class_weights


def map_profiles_to_label(raw_profiles, labelled_data):
    mapped_profiles = {}
    for person in tqdm(raw_profiles, desc='parsing raw profiles...'):
        person_id = person[0]
        if person_id in labelled_data.user_lookup.keys():
            relevant_jobs = labelled_data.user_lookup[person_id]
            mapped_profiles[person_id] = {}
            for num, item in enumerate(labelled_data.tuples[relevant_jobs[0]:relevant_jobs[-1]+1]):
                mapped_profiles[person_id][num] = {}
                # ipdb.set_trace()
                mapped_profiles[person_id][num]["job_title"] = person[-1][num]["job"]
                mapped_profiles[person_id][num]["exp_index"] = item["delta_index"]
                mapped_profiles[person_id][num]["ind_index"] = item["ind_index"]

    return mapped_profiles


def get_class_weights(dataset_train):
    exp, ind = Counter(), Counter()
    for job in tqdm(dataset_train, desc="getting class weights..."):
        exp[job[2]] += 1
        ind[job[1]] += 1
    exp_weights, ind_weights = dict(), dict()
    for k in exp.keys():
        exp_weights[k] = exp[k] / len(dataset_train)
    for k in ind.keys():
        ind_weights[k] = ind[k] / len(dataset_train)
    return {"exp": exp_weights, "ind": ind_weights}


def fit_vectorizer(args, input_data):
    mf = args.max_features if args.max_features != 0 else None
    if args.tfidf == "True":
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=args.max_df, min_df=args.min_df, max_features=mf)
    else:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=args.max_df, min_df=args.min_df, max_features=mf)
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform([np.str_(x) for x in input_data])
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features, vectorizer


def pre_proc_data(data):
    stop_words = set(stopwords.words("french"))
    labels_exp, labels_ind, jobs = [], [], []
    for job in tqdm(data, desc="Parsing profiles..."):
        labels_exp.append(job[2])
        labels_ind.append(job[1])
        cleaned_ab = [w for w in job[0].split(" ") if (w not in stop_words) and (w != "")]
        jobs.append(" ".join(cleaned_ab))
    return jobs, labels_exp, labels_ind


def train_svm(data, labels, class_weights, kernel):
    # model = LinearSVC(class_weight=class_weights)
    model = SVC(kernel=kernel, class_weight=class_weights, verbose=True)
    print("Fitting SVM...")
    model.fit(data, labels)
    print("SVM fitted!")
    return model


def train_nb(data, labels, class_weights):
    priors = [i[1] for i in sorted(class_weights.items())]
    model = MultinomialNB(class_prior=priors)
    print("Fitting Naive Bayes...")
    model.fit(data, labels)
    print("Naive Bayes fitted!")
    return model


def test_for_att(args, class_dict, att_type, labels, model, features, split):
    num_c = len(class_dict[att_type])
    handle = f"{att_type} {split} {args.model}"
    preds, preds_at_k, k = get_predictions(args, model, features, labels, att_type)
    res_at_1 = eval_model(labels, preds, num_c, handle)
    res_at_k = eval_model(labels, preds_at_k, num_c, f"{handle}_@{k}")
    return {**res_at_1, **res_at_k}


def get_predictions(args, model, features, labels, att_type):
    if args.model == "SVM":
        predictions = model.decision_function(features)
    elif args.model == "NB":
        predictions = model.predict_log_proba(features)
    if att_type == "exp":
        k = 2
    else:
        k = 10
    predictions_at_1 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_1.append(get_pred_at_k(sample, lab, 1))

    predictions_at_10 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_10.append(get_pred_at_k(sample, lab, k))
    return predictions_at_1, predictions_at_10, k


def eval_model(labels, preds, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_pred_at_k(pred, label, k):
    ranked = np.argsort(pred, axis=-1)
    largest_indices = ranked[::-1][:len(pred)]
    new_pred = largest_indices[0]
    if label in largest_indices[:k]:
        new_pred = label
    return new_pred
