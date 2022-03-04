import argparse
import fasttext
import ipdb
from tqdm import tqdm
import yaml
import pickle as pkl
import os
import numpy as np
from utils.baselines import eval_model, get_pred_at_k, get_labelled_data
from utils.models import handle_fb_preds, handle_fb_pred, get_metrics, get_metrics_at_k


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        data_train, data_test, class_weights = get_labelled_data(args)
        if args.att_type == "ind":
            k = 10
        else:
            k = 2
        metrics_at_1, metrics_at_k = eval_procedure(data_test, args.bl_type, class_weights, k)
        print(metrics_at_1)
        print(metrics_at_k)


def eval_procedure(datatest, bl_type, class_weights, k):
    print("Testing...")
    att_type = "exp" if k == 2 else "ind"
    if att_type == "exp":
        labels = [i[-1] for i in datatest]
    else:
        labels = [i[1] for i in datatest]
    predictions = np.zeros((len(labels), k))
    if bl_type == "mc":
        ordered_class = sorted(class_weights[att_type].items(), key=lambda x: x[1], reverse=True)
        prediction = [i[0] for i in ordered_class]
        predictions += prediction[:k]
    else: # random case
        for num in tqdm(range(len(labels)), desc="sampling random predictions..."):
            predictions[num, :] = np.random.randint(len(class_weights[att_type]), size=k)
    metrics_at_1 = get_metrics(predictions[:, 0], labels, len(class_weights[att_type]), f"{att_type} {bl_type}")
    metrics_at_k = get_metrics_at_k(predictions, labels, len(class_weights[att_type]), f"{att_type} {bl_type} @{k}")
    return metrics_at_1, metrics_at_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--att_type", type=str, default="ind")
    parser.add_argument("--bl_type", type=str, default="mc") # can be "mc" or "rdm"
    parser.add_argument("--exp_type", type=str, default="delta") #"kmeans" or "uniform" or "kmeansrdm" or "delta"
    parser.add_argument("--exp_levels", type=int, default=5) # 5 of 3
    parser.add_argument("--load_data", type=str, default="True")

    args = parser.parse_args()
    main(args)
