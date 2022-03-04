import argparse
import ipdb
import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader

from data.datasets import StringDataset, StringIndSubDataset
from models.classes import RewriterDeltaBertIndSub, RewriterDeltaBert
from models.train_delta_bert_rewriter_ind_sub import make_xp_title, init_lightning, get_latest_model
from utils import print_tilde_to_csv, collate_for_delta_bert
from tqdm import tqdm
import random
import pandas as pd
import yaml


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        # get the relevant model architecture
        xp_title = make_xp_title(hparams)
        model_name = "/".join(xp_title.split('_'))
        _, _, _, model_path = init_lightning(CFG, xp_title, model_name)
        arguments = {'emb_size': 768,
                     'hp': hparams,
                     'desc': xp_title,
                     "vocab_size": 32005,
                     "model_path": model_path,
                     "datadir": CFG["gpudatadir"],
                     "alpha": hparams.alpha,
                     "beta": hparams.beta}
        print("Initiating model...")
        if hparams.ind_sub == "True":
            model = RewriterDeltaBertIndSub(**arguments)
        else:
            model = RewriterDeltaBert(**arguments)
        model = model.cuda()
        print("Model Loaded.")
        # load the relevant file
        if hparams.eval_mode == "latest":
            model_file = get_latest_model(CFG["modeldir"], model_name)
        elif hparams.eval_mode == "spe":
            model_file = os.path.join(CFG["modeldir"], model_name + "/" + hparams.model_to_test)
        else:
            raise Exception(
                f"Wrong argument provided for EVAL_MODE, can be either \"latest\" or \"spe\", {hparams.eval_mode} was given.")
        print("Loading from previous run...")
        try:
            model.load_state_dict(torch.load(model_file))
        except RuntimeError:
            model.load_state_dict(torch.load(model_file)["state_dict"])
        print(f"Evaluating model : {model_file}.")
        model.on_test_epoch_start()
        # get the test dataset
        datasets = load_datasets(CFG, hparams, ["TEST"], hparams.load_dataset)
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=hparams.test_b_size, collate_fn=collate_for_delta_bert,
                                 num_workers=hparams.num_workers, drop_last=True)
        for batch in tqdm(test_loader, desc="getting predictions for test set..."):
            model.test_step((batch[0], batch[1].cuda(), batch[2].cuda()), 0)

        initial_jobs = [i[0] for i in model.test_decoded_labels]
        initial_delta = [i[1] for i in model.test_decoded_labels]
        initial_ind = [i[2] for i in model.test_decoded_labels]
        expected_delta = [i[3] for i in model.test_decoded_labels]
        expected_ind = [i[4] for i in model.test_decoded_labels]

        stacked_outs = torch.stack(model.test_decoded_outputs)

        x_tilde_to_y_tilde_metrics = get_x_tilde_to_y_tilde_metrics(model, stacked_outs, expected_delta, expected_ind)
        x_to_y_tilde_metrics = get_x_to_y_tilde_metrics(model, initial_jobs, expected_delta, expected_ind)
        x_tilde_to_y_metrics = get_x_tilde_to_y_metrics(model, stacked_outs, initial_delta, initial_ind)
        x_to_y_metrics = get_x_to_y_metrics(model, initial_jobs, initial_delta, initial_ind)
        ipdb.set_trace()
        return {**x_tilde_to_y_tilde_metrics, **x_to_y_tilde_metrics, **x_tilde_to_y_metrics, **x_to_y_metrics}


def get_x_tilde_to_y_tilde_metrics(model, stacked_outs, expected_delta, expected_ind):
    ind_metrics, delta_metrics, ind_preds, delta_preds = model.get_att_control_score(stacked_outs,
                                                                                     expected_delta,
                                                                                     expected_ind)
    return ind_metrics, delta_metrics


def get_x_to_y_metrics(model, initial_jobs, initial_delta, initial_ind):
    tmp = []
    for i in initial_jobs:
        inputs = model.tokenizer(i, truncation=True, padding="max_length", max_length=model.max_len,
             return_tensors="pt")
        tmp.append(inputs["input_ids"].cuda())
    stacked = torch.stack(tmp)
    ind_metrics, delta_metrics, ind_preds, delta_preds = model.get_att_control_score(stacked,
                                                                                     initial_delta,
                                                                                     initial_ind)
    return ind_metrics, delta_metrics


def get_x_to_y_tilde_metrics(model, initial_jobs, expected_delta, expected_ind):
    tmp = []
    for i in initial_jobs:
        inputs = model.tokenizer(i, truncation=True, padding="max_length", max_length=model.max_len,
             return_tensors="pt")
        tmp.append(inputs["input_ids"].cuda())
    stacked = torch.stack(tmp)
    ind_metrics, delta_metrics, ind_preds, delta_preds = model.get_att_control_score(stacked,
                                                                                     expected_delta,
                                                                                     expected_ind)
    return ind_metrics, delta_metrics


def get_x_tilde_to_y_metrics(model, stacked_outs, initial_delta, initial_ind):
    ind_metrics, delta_metrics, ind_preds, delta_preds = model.get_att_control_score(stacked_outs,
                                                                                     initial_delta,
                                                                                     initial_ind)
    return ind_metrics, delta_metrics

def load_datasets(CFG, hparams, splits, load):
    datasets = []
    if hparams.ind_sub == "True":
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": load,
                     "subsample": hparams.subsample,
                     "max_len": 10,
                     "exp_levels": 5,
                     "exp_type": "delta",
                     "tuple_list": None,
                     "lookup": None,
                     "is_toy": hparams.toy_dataset}
        for split in splits:
            datasets.append(StringIndSubDataset(**arguments, split=split))
    else:
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": load,
                     "subsample": hparams.subsample,
                     "max_len": 10,
                     "exp_levels": 5,
                     "exp_type": "delta",
                     "is_toy": hparams.toy_dataset}
        for split in splits:
            datasets.append(StringDataset(**arguments, split=split))
    return datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--prev_model_name", type=str, default=None)
    parser.add_argument("--ind_sub", type=str, default="True")
    parser.add_argument("--eval_mode", type=str, default="latest")  # can be "spe" or "latest"
    # parser.add_argument("--model_to_test", type=str, default="epoch=3-step=14191.tmp_end.ckpt")
    parser.add_argument("--model_to_test", type=str, default="epoch=04.ckpt")
    parser.add_argument("--input_recopy", default="False")
    parser.add_argument("--print_preds", type=str, default="False")
    parser.add_argument("--print_cm", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--debug_noiseless", type=str, default="False")
    parser.add_argument("--beam_search", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--no_tilde", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_to_csv", default="False")
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--min_denoising_epochs", type=int, default=5)
    parser.add_argument("--test_b_size", type=int, default=1)
    parser.add_argument("--feed_z", type=str, default="True")
    parser.add_argument("--divide_lr", type=int, default=10)
    parser.add_argument("--denoising_only", type=str, default="False")
    # model attributes
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--dec_hs", type=int, default=512)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--shuffle_dist", type=int, default=3)
    parser.add_argument("--drop_proba", type=float, default=.1)
    parser.add_argument("--model_type", type=str, default="deltaBertIndSub") #deltaBert2LayFullSeq deltaBertSchedLr deltaBert2LayFullSeqPooled
    parser.add_argument("--att_type", type=str, default="both")
    parser.add_argument("--pooling_window", type=int, default=5)
    parser.add_argument("--classifier_type", type=str, default="ft")
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_val", type=float, default=100.)
    parser.add_argument("--tf", type=float, default=-1.)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=3)
    hparams = parser.parse_args()
    main(hparams)
