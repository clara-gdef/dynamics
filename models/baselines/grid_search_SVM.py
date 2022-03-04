import ipdb
import yaml
import argparse
import os
import pickle as pkl
from utils import DotDict
from models.baselines import train_svm_for_classif


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        print('Commencing data gathering...')
        dico_args = DotDict(get_common_params())
        data_train, data_test, class_weights = train_svm_for_classif.get_labelled_data(dico_args)
        print('Data loaded')
        main(data_train, data_test, class_weights, dico_args)


def main(data_train, data_test, class_weights, dico):
    exp_title = f"gs_poly_SVC_{args.exp_type}_{args.exp_levels}exp_res_{args.model}"
    if args.tfidf == "True":
        exp_title += "_tfidf"
    if args.subsample > 0:
        exp_title += f"_{args.subsample}"
    if args.TRAIN == "True":
        results = {}
        for min_df in [1e-2, 1e-3, 1e-4]:
            results[min_df] = {}
            for max_df in [.6, .8, 1.0]:
                results[min_df][max_df] = {}
                for max_features in [2000, 3000, 5000]:
                    print(f"min_df: {min_df}, max_df: {max_df}, max_features: {max_features}")
                    results[min_df][max_df] = {}
                    dico['min_df'] = min_df
                    dico['max_df'] = max_df
                    dico['max_features'] = int(max_features)
                    arg = DotDict(dico)
                    results[min_df][max_df][int(max_features)] = train_svm_for_classif.main(arg, data_train, data_test, class_weights)
                    with open(os.path.join(CFG["gpudatadir"], f"{exp_title}.pkl"), 'wb') as f:
                        pkl.dump(results, f)
    if args.TRAIN == "False" and args.TEST == "True":
        with open(os.path.join(CFG["gpudatadir"], f"{exp_title}.pkl"), 'rb') as f:
            results = pkl.load(f)
    if args.TEST == "True":
        print("RESULTS FOR EXP")
        results_sanity_check(results, f"f1_exp TEST {args.model}")
        best_acc_keys, best_f1_keys = analyze_results(results, f"exp TEST {args.model}")
        print(select_relevant_keys(f"exp TEST {args.model}", results[best_f1_keys[0]][best_f1_keys[1]][best_f1_keys[2]]))
        print(select_relevant_keys(f"exp TRAIN {args.model}", results[best_f1_keys[0]][best_f1_keys[1]][best_f1_keys[2]]))
        ipdb.set_trace()
        print("RESULTS FOR IND")
        best_acc_keys, best_f1_keys = analyze_results(results, f"ind TEST {args.model}")
        print(select_relevant_keys(f"ind TEST {args.model}", results[best_f1_keys[0]][best_f1_keys[1]][best_f1_keys[2]]))
        print(select_relevant_keys(f"ind TRAIN {args.model}", results[best_f1_keys[0]][best_f1_keys[1]][best_f1_keys[2]]))
        ipdb.set_trace()


def select_relevant_keys(handle, test_results):
    relevant_results = {}
    for k in test_results.keys():
        if k.startswith(f"acc_{handle}") or k.startswith(f"f1_{handle}"):
            relevant_results[k] = test_results[k]
    return relevant_results


def results_sanity_check(results, handle):
    tmp = []
    for min_df in results.keys():
        for max_df in results[min_df].keys():
            for max_features in results[min_df][max_df].keys():
                tmp.append(results[min_df][max_df][max_features][handle])
    print(set(tmp))
    return tmp


def get_common_params():
    dico = {"model": args.model,
            "load_data": args.load_data,
            "load_dataset": "True",
            "subsample": args.subsample,
            "save_model": "False",
            "load_model": "False",
            "max_len": args.max_len,
            "kernel": "linear",
            "exp_type": args.exp_type,
            "exp_levels": args.exp_levels,
            "suffix": args.suffix,
            "tfidf": args.tfidf}
    return dico


def analyze_results(test_results, handle):
    best_acc, best_f1 = 0, 0
    best_acc_keys = None
    best_f1_keys = None
    for min_df in test_results.keys():
        for max_df in test_results[min_df].keys():
            for max_features in test_results[min_df][max_df].keys():
                if test_results[min_df][max_df][max_features]["acc_" + handle] > best_acc:
                    best_acc_keys = (min_df, max_df, max_features)
                    best_acc = test_results[min_df][max_df][max_features]["acc_" + handle]
                if test_results[min_df][max_df][max_features]["f1_" + handle] > best_f1:
                    best_f1_keys = (min_df, max_df, max_features)
                    best_f1 = test_results[min_df][max_df][max_features]["f1_" + handle]
    print("Evaluated for min_df= [" + str(test_results.keys()) + "], max_df=[" + str(test_results[min_df].keys()) + "]")
    return best_acc_keys, best_f1_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SVM")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=10000)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="_new_it8")
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--load_data", type=str, default="True")
    parser.add_argument("--tfidf", default="True")
    args = parser.parse_args()
    init(args)
