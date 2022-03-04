import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train_delta_bert_rewriter
from utils import DotDict
from tqdm import tqdm


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
    for pw in [10, 12, 15, 17, 20]:
        print(f"Grid Search for (pooling_window={pw}")
        dico["pooling_window"] = pw
        test_results[pw] = {}
        arg = DotDict(dico)
        if dico["TRAIN"] == "True":
            train_delta_bert_rewriter.init(arg)
        if dico["TEST"] == "True":
            dico["TRAIN"] = "False"
            arg = DotDict(dico)
            test_results[pw] = train_delta_bert_rewriter.init(arg)
        dico["TRAIN"] = "True"
        res_path = os.path.join(CFG["gpudatadir"], "EVAL_PW2_gs_" + hparams.model_type)
        with open(res_path, "wb") as f:
            pkl.dump(test_results, f)
    if dico["TEST"] == "True":
        if dico["TRAIN"] == "False":
            res_path = os.path.join(CFG["gpudatadir"], "EVAL_PW2_gs_" + hparams.model_type)
            with open(res_path, "rb") as f:
                test_results = pkl.load(f)
        best_params_bleu, best_bleu = get_best_params_for_bleu(test_results)
        best_params_bleu_1, best_bleu_1 = get_best_params_for_bleu_1(test_results)
        best_params_f1, best_f1 = get_best_params_for_f1_ind(test_results, "ind")

    ipdb.set_trace()


def get_best_params_for_bleu(test_results):
    best_bleu = 0.0
    best_params_bleu = None
    for pw in tqdm(test_results.keys(), desc="Parsing gs results to find best bleu score..."):
        if test_results[pw]["bleu"] > best_bleu:
            best_bleu = test_results[pw]["bleu"]
            best_params_bleu = pw
    return best_params_bleu, best_bleu


def get_best_params_for_bleu_1(test_results):
    best_bleu_1 = 0.0
    best_params_bleu_1 = None
    for pw in tqdm(test_results.keys(), desc="Parsing gs results to find best bleu score..."):
        if test_results[pw]["bleu1"] > best_bleu_1:
            best_bleu_1 = test_results[pw]["bleu1"]
            best_params_bleu_1 = pw
    return best_params_bleu_1, best_bleu_1



def get_best_params_for_f1_ind(test_results, handle):
    best_f1 = 0.0
    best_params_f1 = None
    for pw in tqdm(test_results.keys(), desc=f"Parsing gs results to find best f1_{handle} score..."):
        if test_results[pw][f"f1_{handle}"] > best_f1:
            best_f1 = test_results[pw][f"f1_{handle}"]
            best_params_f1 = pw
    return best_params_f1, best_f1


def init_args(hparams):
    dico = {"gpus": hparams.gpus,
            "load_dataset":  hparams.load_dataset,
            "feed_z": "True",
            "auto_lr_find": "False",
            "load_from_checkpoint": "False",
            "fine_tune_word_emb": "False",
            "print_cm": "False",
            "denoising_only": "False",
            "print_preds": "False",
            "print_to_csv": "True",
            "no_tilde": "False",
            "beam_search": "False",
            "classifier_type": "FT",
            "optim": "adam",
            "checkpoint": None,
            "DEBUG": hparams.DEBUG,
            "TEST": hparams.TEST,
            "TRAIN": hparams.TRAIN,
            "subsample": hparams.subsample,
            "num_workers": hparams.num_workers,
            "light": "True",
            "toy_dataset": "True",
            # model attributes
            "num_layer": 2,
            "para": "False",
            "lr": hparams.lr,
            "drop_proba": hparams.drop_proba,
            "shuffle_dist": hparams.shuffle_dist,
            "b_size": hparams.b_size,
            "test_b_size": 1,
            "hidden_size": hparams.hidden_size,
            "max_len": hparams.max_len,
            "tf": 1,
            "att_type": "both",
            "input_recopy": "False",
            "eval_mode": "latest",
            "debug_noiseless": "False",
            "model_type": hparams.model_type,
            "alpha": hparams.alpha,
            "dec_hs": hparams.dec_hs,
            "beta": hparams.beta,
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
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--num_workers", type=int, default=4)
    # model attributes
    parser.add_argument("--b_size", type=int, default=200)
    parser.add_argument("--dec_hs", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="TESTdeltaBert2LayFullSeqPooledYExcluded")
    parser.add_argument("--shuffle_dist", type=int, default=3)
    parser.add_argument("--drop_proba", type=float, default=.1)
    parser.add_argument("--max_len", type=int, default=10)
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    hparams = parser.parse_args()
    grid_search(hparams)
