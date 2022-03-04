import argparse
import fasttext
import ipdb
from tqdm import tqdm
import yaml
import pickle as pkl
import os
import numpy as np
from collections import Counter
from utils.baselines import eval_model, get_pred_at_k
from utils.models import handle_fb_preds, handle_fb_pred, get_metrics, get_metrics_at_k


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        params, tmp, train_file, test_file = get_param_and_suffix(args.att_type)
        suffix = tmp + args.suffix

        data_dist = get_data_dist(train_file)
        print(f"initial data dist is: {data_dist}")
        tgt_file = os.path.join(CFG["modeldir"], f"ft_att_classifier_{suffix}.bin")
        if args.TRAIN == "True":
            model = fasttext.train_supervised(input=train_file, lr=params[0], epoch=params[1], wordNgrams=params[2])
            model.save_model(tgt_file)
            print(f"Model saved at {tgt_file}")
        if args.TEST == "True":
            if args.att_type == "ind" or args.att_type == "ind_user":
                with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f_name:
                    industry_dict = pkl.load(f_name)
                rev_ind_dict = dict()
                for k, v in industry_dict.items():
                    if len(v.split(" ")) > 1:
                        new_v = "_".join(v.split(" "))
                    else:
                        new_v = v
                    rev_ind_dict[new_v] = k
                label_dict = rev_ind_dict
            else:
                label_dict = None
            if args.TRAIN == "False":
                model = fasttext.load_model(tgt_file)
            metrics_at_1, metrics_at_2 = eval_procedure(test_file, model, label_dict)
            print(metrics_at_1)
            print(metrics_at_2)
            print("TRAIN METRICS")
            metrics_at_1, metrics_at_2 = eval_procedure(train_file, model, label_dict)
            print(metrics_at_1)
            print(metrics_at_2)


def get_data_dist(file_name):
    cnt = Counter()
    num_lines = 0
    with open(file_name, 'r') as test_f:
        for line in test_f:
            num_lines += 1
    with open(file_name, 'r') as test_f:
        pbar = tqdm(test_f, total=num_lines)
        for line in pbar:
            tmp = line.split("__label__")[1].split(" ")[0]
            cnt[tmp] += 1
    total = sum([i for i in cnt.values()])
    dist_dict = {k: 100*v/total for k, v in cnt.items()}
    return dist_dict


def eval_procedure(test_file, model, label_dict):
    print("Testing...")
    num_lines = 0
    with open(test_file, 'r') as test_f:
        for line in test_f:
            num_lines += 1
    labels = []
    predictions = []
    with open(test_file, 'r') as test_f:
        pbar = tqdm(test_f, total=num_lines)
        for line in pbar:
            tmp = line.split("__label__")[1]
            if args.att_type == "exp" or args.att_type == "delta":
                k = 2
                labels.append(int(tmp[0]))
                pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
                predictions.append([int(p) for p in pred])
            else:
                k = 10
                labels.append(label_dict[tmp.split(" ")[0]])
                pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
                predictions.append([label_dict[p] for p in pred])
    preds = np.stack(predictions)
    pred_dist = Counter()
    for i in preds[:, 0]:
        pred_dist[i] += 100/len(preds)
    print(f"prediction distribution : {pred_dist}")
    metrics_at_1 = get_metrics(preds[:, 0], labels, len(model.labels), args.att_type)
    metrics_at_k = get_metrics_at_k(preds, labels, len(model.labels), f"{args.att_type} @{k}")
    return metrics_at_1, metrics_at_k


def get_param_and_suffix(att_type):
    if args.max_len == 128:
        suffix = ""
        if att_type == "ind":
            params = (0.1, 25, 1)
            att_string = "ind"
        elif att_type == "exp":
            params = (0.1, 5, 3)
            att_string = "exp_5"
        elif att_type == "delta":
            params = (0.1, 5, 3)
            att_string = "delta_5"
        else:
            raise Exception(f"Wrong att_type specified, can be \'ind\' or \'exp\', got {att_type}")
    elif args.max_len == 10:
        suffix = '_10'
        if att_type == "ind":
            att_string = "ind"
            params = (.1, 5, 3)
        elif att_type == "exp":
            if args.sub_ind == "True":
                params = (0.1, 50, 5)
                if args.suffix == "_new_it1":
                    params = (0.1, 5, 1)
                if args.suffix == "_new_it8":
                    params = (0.1, 50, 5)
                if args.suffix == "_it50":
                    params = (0.1, 5, 5)
            else:
                params = (0.1, 5, 1)
            att_string = f"exp{args.exp_levels}_{args.exp_type}"
        elif att_type == "delta":
            params = (.1, 5, 3)
            att_string = f"delta_{args.exp_levels}"
        elif att_type == "ind_user":
            params = (0.3, 5, 1)
            att_string = "ind_user"
        else:
            raise Exception(f"Wrong att_type specified, can be \'ind\' or \'exp\' or \'delta\', got {att_type}")
    else:
        raise Exception(f"Wrong max len specified, can be 10 or 128, got {args.max_len}")
    if args.sub_ind == "True":
        suffix += "_subInd"
    train_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.train")
    test_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.test")
    if att_type == "ind_user":
        params = (0.3, 5, 1)
        train_file = os.path.join(CFG["gpudatadir"], "ft_classif_USER_supervised_ind.train")
        test_file = os.path.join(CFG["gpudatadir"], "ft_classif_USER_supervised_ind.test")
    return params, f"{att_string}{suffix}", train_file, test_file



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--att_type", type=str, default="exp")
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--sub_ind", type=str, default="True")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--suffix", type=str, default="_new_it8")
    parser.add_argument("--TRAIN", type=str, default="True")
    args = parser.parse_args()
    main(args)
