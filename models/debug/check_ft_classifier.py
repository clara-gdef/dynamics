import argparse
import yaml
from data.datasets import ExpIndJobsDataset
import ipdb
import pickle as pkl
import torch
from tqdm import tqdm
import fasttext
from utils.models import decode_tokenized_words, get_metrics_at_k, handle_fb_preds
import os
from collections import Counter


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        ds_arguments = {'data_dir': CFG["gpudatadir"],
                         "light": args.light,
                         "max_len": args.max_len,
                         "is_toy": False,
                         "subsample": 0,
                         "load": "True"}

        with open(os.path.join( CFG["gpudatadir"], "vocab_dict.pkl"), "rb") as f:
            vocab_dict = pkl.load(f)
        classifier_ind = fasttext.load_model("/data/gainondefor/dynamics/ft_att_classifier_ind.bin")
        with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f_name:
            industry_dict = pkl.load(f_name)
        rev_ind_dict = dict()
        for k, v in industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        file_train_pre_proc = "/local/gainondefor/work/data/ft_classif_supervised_ind.preprocessed.txt"

        file_train =  "/local/gainondefor/work/data/ft_classif_supervised_ind.txt"
        model_pp = fasttext.train_supervised(input=file_train_pre_proc, autotuneValidationFile=file_train_pre_proc,
                                          autotuneDuration=600, verbose=True)
        model_raw = fasttext.train_supervised(input=file_train, autotuneValidationFile=file_train,
                                          autotuneDuration=600, verbose=True)
        model_pp.test(file_train_pre_proc)
        model_raw.test(file_train)
        ipdb.set_trace()
        for split in ["TRAIN", "TEST"]:
            dataset = ExpIndJobsDataset(**ds_arguments, split=split)
            classif_res = get_classif_metrics(dataset, split, vocab_dict, classifier_ind, rev_ind_dict)
            print(classif_res)
            ipdb.set_trace()




def get_classif_metrics(dataset, split, vocab_dict, classifier_ind, rev_ind_dict):
    predicted_inds = []
    expected_ind = []
    for job in tqdm(dataset, desc="Classifying " + split + " dataset..."):
        sequence = job[0]
        expected_ind.append(job[3])
        sentence = decode_tokenized_words(sequence[0], vocab_dict)
        predictions_ind = handle_fb_preds(classifier_ind.predict(sentence, k=5))
        predicted_inds.append([rev_ind_dict[i] for i in predictions_ind])
    ind_metrics = get_metrics_at_k(predicted_inds, expected_ind, len(rev_ind_dict), 'ind_@5_' + split.lower(()))
    return ind_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    main(args)
