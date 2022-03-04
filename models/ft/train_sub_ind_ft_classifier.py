import argparse
import fasttext
import ipdb
from tqdm import tqdm
import yaml
import pickle as pkl
import os
import numpy as np
from utils.baselines import eval_model, get_pred_at_k
from utils.models import handle_fb_preds, handle_fb_pred, get_metrics, get_metrics_at_k


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        params, suffix, train_file, test_file = get_param_and_suffix()
        tgt_file = os.path.join(CFG["modeldir"], f"ft_att_classifier_{suffix}.bin")
        if args.TRAIN == "True":
            model = fasttext.train_supervised(input=train_file, lr=params[0], epoch=params[1], wordNgrams=params[2])
            model.save_model(tgt_file)
            print(f"Model saved at {tgt_file}")
        if args.TEST == "True":
            with open(os.path.join(CFG["gpudatadir"], "20_industry_dict.pkl"), 'rb') as f_name:
                industry_dict = pkl.load(f_name)
            rev_ind_dict = dict()
            for k, v in industry_dict.items():
                if len(v.split(" ")) > 1:
                    new_v = "_".join(v.split(" "))
                else:
                    new_v = v
                rev_ind_dict[new_v] = k
            label_dict = rev_ind_dict
            if args.TRAIN == "False":
                model = fasttext.load_model(tgt_file)
            metrics_at_1, metrics_at_2 = eval_procedure(test_file, model, label_dict)
            print(metrics_at_1)
            print(metrics_at_2)
            print("TRAIN METRICS")
            metrics_at_1, metrics_at_2 = eval_procedure(train_file, model, label_dict)
            print(metrics_at_1)
            print(metrics_at_2)


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
            k = 5
            labels.append(label_dict[tmp.split(" ")[0]])
            pred = handle_fb_preds(model.predict(tmp[2:-2], k=k))
            predictions.append([label_dict[p] for p in pred])
    preds = np.stack(predictions)
    metrics_at_1 = get_metrics(preds[:, 0], labels, len(model.labels), "ind20")
    metrics_at_k = get_metrics_at_k(preds, labels, len(model.labels), f"ind20 @{k}")
    return metrics_at_1, metrics_at_k


def get_param_and_suffix():
    suffix = '_10'
    att_string = "ind20"
    params = (0.5, 50, 5)

    train_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.train")
    test_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.test")
    return params, f"{att_string}{suffix}", train_file, test_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--TRAIN", type=str, default="True")
    args = parser.parse_args()
    main(args)
