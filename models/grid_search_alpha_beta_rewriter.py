import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train_rewriter
from utils import DotDict


def grid_search(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            main(hparams)
    else:
        main(hparams)


def main(hparams):
    test_results = {}
    dico = init_args(hparams)
    for str_alpha in hparams.alpha:
        alpha = float(str_alpha)
        beta = 1 - alpha
        test_results[alpha] = {}
        for str_lr in hparams.lr:
            lr = float(str_lr)
            test_results[alpha][lr] = {}
            for hs in hparams.hidden_size:
                test_results[alpha][lr][int(hs)] = {}
                print("Grid Search for (lr=" + str(lr) + ", alpha=" + str(alpha) + ", hidden_size=" + str(hs) + " )")
                dico['lr'] = float(lr)
                dico["alpha"] = float(alpha)
                dico["beta"] = float(beta)
                dico["hidden_size"] = int(hs)
                arg = DotDict(dico)
                if dico["TRAIN"] == "True":
                    train_rewriter.init(arg)
                if dico["TEST"] == "True":
                    dico["TRAIN"] = "False"
                    test_results[alpha][lr][int(hs)] = train_rewriter.init(arg)
                dico["TRAIN"] = "True"
        ## TODO REMOVE THIS - UNINDENT
        res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_" + hparams.model_type)
        with open(res_path, "wb") as f:
            pkl.dump(test_results, f)


def init_args(hparams):
    dico = {"gpus": hparams.gpus,
            "load_dataset": "True",
            "auto_lr_find": "False",
            "load_from_checkpoint": "False",
            "fine_tune_word_emb": "False",
            "print_preds": "False",
            "optim": "adam",
            "checkpoint": "09",
            "DEBUG": hparams.DEBUG,
            "TEST": hparams.TEST,
            "TRAIN": hparams.TRAIN,
            "subsample": hparams.subsample,
            "num_workers": hparams.num_workers,
            "light": "True",
            "toy_dataset": "False",
            # model attributes
            "para": "False",
            "b_size": hparams.b_size,
            "max_len": hparams.max_len,
            "pooling_window": 5,
            "model_type": hparams.model_type,
            # global hyper params
            "clip_val": 1.,
            "wd": 0.,
            "save_step": 50,
            "dpo": 0.,
            "epochs": hparams.epochs}
    return dico


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--num_workers", type=int, default=4)
    # model attributes
    parser.add_argument("--b_size", type=int, default=30)
    parser.add_argument("--hidden_size", nargs='+', default=[512])
    parser.add_argument("--model_type", type=str, default="title_beam_gs")
    parser.add_argument("--max_len", type=int, default=10)
    # global hyper params
    parser.add_argument("--alpha", nargs='+', default=[.25, .5, .75])
    parser.add_argument("--lr", nargs='+', default=[1e-4])
    parser.add_argument("--epochs", type=int, default=30)
    hparams = parser.parse_args()
    grid_search(hparams)
