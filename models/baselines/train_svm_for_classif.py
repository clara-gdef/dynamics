import argparse
import yaml
import os
import joblib
import ipdb
import numpy
import pickle as pkl
from tqdm import tqdm
from sklearn.preprocessing import scale
from utils.baselines import fit_vectorizer, train_svm, train_nb, get_labelled_data, pre_proc_data, test_for_att


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
        data_train, data_test = subsample_according_to_class_weight(args, data_train, data_test, class_dict, args.subsample)
    cleaned_profiles, labels_exp_train, labels_ind_train = pre_proc_data(data_train)
    train_features, vectorizer = fit_vectorizer(args, cleaned_profiles)
    exp_name = get_exp_name(args)
    tgt_file = os.path.join(CFG["modeldir"], exp_name)
    if args.load_model == "True":
        model_exp = joblib.load(f"{tgt_file}_exp.joblib")
        vectorizer = joblib.load(f"{tgt_file}_vectorizer_svc_{args.kernel}.joblib")
        # model_ind = joblib.load(f"{tgt_file}_ind.joblib")
        # vectorizer = joblib.load(f"{tgt_file}_exp.joblib")
    else:
        if args.model == "SVM":
            model_exp = train_svm(scale(train_features[:args.subsample]), labels_exp_train[:args.subsample], class_dict["exp"], args.kernel)
            # model_ind = train_svm(train_features[:args.subsample], labels_ind_train[:args.subsample], class_dict["ind"])
        elif args.model == "NB":
            model_exp = train_nb(train_features[:args.subsample], labels_exp_train[:args.subsample], class_dict["exp"])
            # model_ind = train_nb(train_features[:args.subsample], labels_ind_train[:args.subsample], class_dict["ind"])
        else:
            raise Exception("Wrong model type specified, can be either SVM or NB")
        if args.save_model == "True":
            joblib.dump(model_exp, f"{tgt_file}_exp_svc_{args.kernel}.joblib")
            # joblib.dump(model_ind, f"{tgt_file}_ind.joblib")
            joblib.dump(vectorizer, f"{tgt_file}_vectorizer_svc_{args.kernel}.joblib")
    # TEST
    # assert model_exp.coef_.shape[0] == args.exp_levels
    cleaned_abstracts_test, labels_exp_test, labels_ind_test = pre_proc_data(data_test)
    test_features = vectorizer.transform(cleaned_abstracts_test)

    res_test_exp = test_for_att(args, class_dict, "exp", labels_exp_test[:args.subsample], model_exp, test_features[:args.subsample].todense(), "TEST")
    # res_test_ind = test_for_att(args, class_dict, "ind", labels_ind_test[:args.subsample], model_ind, test_features[:args.subsample], "TEST")

    res_train_exp = test_for_att(args, class_dict, "exp", labels_exp_train[:args.subsample], model_exp, train_features[:args.subsample], "TRAIN")
    # res_train_ind = test_for_att(args, class_dict, "ind", labels_ind_train[:args.subsample], model_ind, train_features[:args.subsample], "TRAIN")

    # all_res = {**res_test_exp, **res_test_ind, **res_train_exp, **res_train_ind}
    all_res = {**res_test_exp, **res_train_exp}
    print(f"EXP NAME: {exp_name}")
    print(f"NUM FEATURES: {model_exp.coef_.shape[1]}")
    print(select_relevant_keys(f"exp TEST {args.model}", all_res))
    #ipdb.set_trace()
    return all_res


def select_relevant_keys(handle, test_results):
    relevant_results = {}
    for k in test_results.keys():
        if k.startswith(f"acc_{handle}") or k.startswith(f"f1_{handle}"):
            relevant_results[k] = test_results[k]
    return relevant_results


def subsample_according_to_class_weight(args, data_train, data_test, class_dict, subsample):
    tgt_file = os.path.join(CFG["gpudatadir"], f"subsampled_data_{subsample}_{args.exp_type}_{args.exp_levels}exp.pkl")
    if os.path.isfile(tgt_file):
        with open(tgt_file, "rb") as f:
            dat = pkl.load(f)
        sub_train = dat["train"]
        sub_test = dat["test"]
    else:
        ind_weights = class_dict["ind"]
        num_sample_per_class = {k: int(v*subsample) for k, v in ind_weights.items()}
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
    exp_name = f"{args.model}_{args.exp_levels}exp_{args.exp_type}"
    if args.max_len != 10:
        exp_name += f"_maxlen{args.max_len}"
    if args.tfidf == "True":
        exp_name += "_tfidf"
    return exp_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type=str, default="True")
    parser.add_argument("--min_df", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="SVM") # SVM or NB
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--max_features", type=int, default=3000)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--load_model", type=str, default="False")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--exp_type", type=str, default="uniform") #"kmeans" or "uniform" or "kmeansrdm" or "delta"
    parser.add_argument("--exp_levels", type=int, default=3) # 5 of 3
    parser.add_argument("--tfidf", default="True")
    parser.add_argument("--suffix", type=str, default="_new_it8")
    parser.add_argument("--custom_centers", type=str, default="False")
    parser.add_argument("--kernel", type=str, default="linear")
    args = parser.parse_args()
    init(args)
