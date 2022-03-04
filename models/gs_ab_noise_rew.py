import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train_rewriter
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

    # for str_alpha in hparams.alpha:
    #     alpha = float(str_alpha)
    #     beta = 1 - alpha
    #     test_results[alpha] = {}
    #     for str_dpba in hparams.drop_proba:
    #         str_dpba = float(str_dpba)
    #         test_results[alpha][str_dpba] = {}
    #         for str_shdist in hparams.shuffle_dist:
    #             test_results[alpha][str_dpba][int(str_shdist)] = {}
    #             print(f"Grid Search for (drop_proba={str_dpba}, alpha={alpha}, shuffle_dist={str_shdist})")
    #             dico['drop_proba'] = float(str_dpba)
    #             dico["alpha"] = float(alpha)
    #             dico["beta"] = float(beta)
    #             dico["shuffle_dist"] = int(str_shdist)
    #             arg = DotDict(dico)
    #             if dico["TRAIN"] == "True":
    #                 train_rewriter.init(arg)
    #             if dico["TEST"] == "True":
    #                 dico["TRAIN"] = "False"
    #                 test_results[alpha][str_dpba][int(str_shdist)] = train_rewriter.init(arg)
    #             dico["TRAIN"] = "True"
    #     res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_" + hparams.model_type)
    #     with open(res_path, "wb") as f:
    #         pkl.dump(test_results, f)
    if dico["TEST"] == "True":
        if dico["TRAIN"] == "False":
            res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_" + hparams.model_type)
            with open(res_path, "rb") as f:
                test_results = pkl.load(f)
        best_params_bleu, best_bleu1 = get_best_params_for_bleu(test_results)
    ipdb.set_trace()


def get_best_params_for_bleu(test_results):
    best_bleu1 = 0.0
    best_params_bleu = None
    for alpha in tqdm(test_results.keys(), desc="Parsing gs results to find best bleu score..."):
        for dpba in test_results[alpha].keys():
            for shdist in test_results[alpha][dpba].keys():
                if test_results[alpha][dpba][shdist][0]["bleu1"] > best_bleu1:
                    best_bleu1 = test_results[alpha][dpba][shdist][0]["bleu1"]
                    best_params_bleu = (alpha, dpba, shdist)
    return best_params_bleu, best_bleu1


def init_args(hparams):
    dico = {"gpus": hparams.gpus,
            "load_dataset": "True",
            "auto_lr_find": "False",
            "load_from_checkpoint": "False",
            "fine_tune_word_emb": "False",
            "print_preds": "False",
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
            "para": "False",
            "lr": hparams.lr,
            "b_size": hparams.b_size,
            "hidden_size": hparams.hidden_size,
            "max_len": hparams.max_len,
            "pooling_window": 5,
            "tf": 1,
            "att_type": "both",
            "input_recopy": False,
            "debug_noiseless": False,
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
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="title_beam_gs")
    parser.add_argument("--shuffle_dist", nargs='+', default=[1, 3, 5])
    parser.add_argument("--drop_proba", nargs='+', default=[.1, .3, .5])
    parser.add_argument("--max_len", type=int, default=10)
    # global hyper params
    parser.add_argument("--alpha", nargs='+', default=[.25, .5, .75])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    hparams = parser.parse_args()
    grid_search(hparams)
