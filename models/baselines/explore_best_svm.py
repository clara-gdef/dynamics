import argparse

import yaml
import os
import pickle as pkl
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
import ipdb
import numpy as np
from collections import Counter

from models.baselines.grid_search_bow import analyze_results
from utils.baselines import get_labelled_data, pre_proc_data,  train_svm, train_nb
from models.baselines.exp_bow_svm import test_for_att, eval_model


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        print('Commencing data gathering...')
        data_train, data_test, class_weights = get_labelled_data(args)
        print('Data loaded')
        with open(os.path.join(CFG["gpudatadir"], f"gs_bow_res_{args.model}.pkl"), 'rb') as f:
            gs_results = pkl.load(f)
        exp_best_model_args, _ = analyze_results(gs_results, f"exp TEST {args.model}")
        ind_best_model_args, _ = analyze_results(gs_results, f"ind TEST {args.model}")
        main(args, data_train, data_test, class_weights, exp_best_model_args, ind_best_model_args)


def main(args, data_train, data_test, class_weights, exp_args, ind_args):
    cleaned_profiles, labels_exp_train, labels_ind_train = pre_proc_data(data_train)
    if args.load_model == "True":
        model_exp = load(os.path.join(CFG["modeldir"], f"exp_best_{args.model}.joblib"))
        model_ind = load(os.path.join(CFG["modeldir"], f"ind_best_{args.model}.joblib"))
        vectorizer_exp = load(os.path.join(CFG["modeldir"], f"exp_best_{args.model}_vectorizer.joblib"))
        vectorizer_ind = load(os.path.join(CFG["modeldir"], f"ind_best_{args.model}_vectorizer.joblib"))
    else:
        train_features_exp, vectorizer_exp = fit_vectorizer_with_specific_args(cleaned_profiles, exp_args[0],
                                                                               exp_args[1], exp_args[2])
        train_features_ind, vectorizer_ind = fit_vectorizer_with_specific_args(cleaned_profiles, ind_args[0],
                                                                               ind_args[1], ind_args[2])
        if args.model == "SVM":
            model_exp = train_svm(train_features_exp, labels_exp_train, class_weights["exp"])
            model_ind = train_svm(train_features_ind, labels_ind_train, class_weights["ind"])
        elif args.model == "NB":
            model_exp = train_nb(train_features_exp, labels_exp_train, class_weights["exp"])
            model_ind = train_nb(train_features_ind, labels_ind_train, class_weights["ind"])
        else:
            raise Exception("Wrong model type specified, can be either SVM or NB")
        dump(model_exp, os.path.join(CFG["modeldir"], f"exp_best_{args.model}.joblib"))
        dump(model_ind, os.path.join(CFG["modeldir"], f"ind_best_{args.model}.joblib"))
        dump(vectorizer_exp, os.path.join(CFG["modeldir"], f"exp_best_{args.model}_vectorizer.joblib"))
        dump(vectorizer_ind, os.path.join(CFG["modeldir"], f"ind_best_{args.model}_vectorizer.joblib"))

    # TEST
    cleaned_abstracts_test, labels_exp_test, labels_ind_test = pre_proc_data(data_test)
    test_features_exp = vectorizer_exp.transform(cleaned_abstracts_test)
    test_features_ind = vectorizer_ind.transform(cleaned_abstracts_test)

    split = "TEST"
    num_c = len(class_weights["exp"])
    handle = f"exp {split} {args.model}"
    if args.model == "SVM":
        predictions = model_exp.decision_function(test_features_exp)
        k = 2
    if args.model == "NB":
        predictions = model_exp.predict_proba(test_features_exp)
        k = 2
    predictions_at_1 = []
    for sample, lab in zip(predictions, labels_exp_test):
        ranked = np.argsort(sample, axis=-1)
        largest_indices = ranked[::-1][:len(sample)]
        new_pred = largest_indices[0]
        if lab in largest_indices[:1]:
            new_pred = lab
        predictions_at_1.append(new_pred)
    pred_train_dist = get_pred_dist_per_class(predictions_at_1)
    predictions_at_k = []
    for sample, lab in zip(predictions, labels_exp_test):
        ranked = np.argsort(sample, axis=-1)
        largest_indices = ranked[::-1][:len(sample)]
        new_pred = largest_indices[0]
        if lab in largest_indices[:k]:
            new_pred = lab
        predictions_at_k.append(new_pred)
    res_at_1 = eval_model(labels_exp_test, predictions_at_1, num_c, handle)
    res_at_k = eval_model(labels_exp_test, predictions_at_k, num_c, f"{handle}_@{k}")

    res_test_exp = test_for_att(args, class_weights, "exp", labels_exp_test, model_exp, test_features_exp, "TEST")
    res_test_ind = test_for_att(args, class_weights, "ind", labels_ind_test, model_ind, test_features_ind, "TEST")

    res_train_exp = test_for_att(args, class_weights, "exp", labels_exp_train, model_exp, train_features_exp, "TRAIN")
    res_train_ind = test_for_att(args, class_weights, "ind", labels_ind_train, model_ind, train_features_ind, "TRAIN")
    ipdb.set_trace()


def get_pred_dist_per_class(predictions):
    pred_counter = Counter()
    for i in predictions:
        pred_counter[i] += 1
    for i in pred_counter.keys():
        pred_counter[i] = 100*pred_counter[i]/len(predictions)
    return pred_counter


def fit_vectorizer_with_specific_args(input_data, min_df, max_df, max_features):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_df=max_df, min_df=min_df, max_features=max_features)
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform(input_data)
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--load_model", type=str, default="True")
    parser.add_argument("--model", type=str, default="SVM")
    parser.add_argument("--tfidf", default="True")
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--max_len", type=int, default=10)
    args = parser.parse_args()
    init(args)
