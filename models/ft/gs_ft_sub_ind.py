import argparse
import fasttext
import ipdb
from tqdm import tqdm
import yaml
import pickle as pkl
import os

def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        att_string, tmp = get_suffix(args)
        suffix = tmp + args.suffix
        print(suffix)
        train_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.train")
        test_file = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{att_string}{suffix}.test")
        print(train_file)
        if args.TRAIN == "True":
            results = dict()
            pbar = tqdm(range(len([.1, .3, .5, .7, .9, 1.0])*len([5, 25, 50])*len([1, 3, 5])))
            for lr in [.1, .3, .5, .7, .9, 1.0]:
                results[lr] = dict()
                for epochs in [5, 25, 50]:
                    results[lr][epochs] = dict()
                    for word_ngram in [1, 3, 5]:
                        print(f"lr: {lr}, ep: {epochs}, ngram: {word_ngram}")
                        try:
                            results[lr][epochs][word_ngram] = dict()
                            model = fasttext.train_supervised(input=train_file, lr=lr, epoch=epochs,  wordNgrams=word_ngram, loss='hs')
                            res = model.test(train_file)
                            prec, rec = res[1], res[2]
                            results[lr][epochs][word_ngram]["train_f1"] = 2 * (prec * rec) / (prec + rec)
                            results[lr][epochs][word_ngram]["train_prec"] = prec
                            results[lr][epochs][word_ngram]["train_rec"] = rec
                            res = model.test(test_file)
                            prec, rec = res[1], res[2]
                            results[lr][epochs][word_ngram]["test_f1"] = 2 * (prec * rec) / (prec + rec)
                            results[lr][epochs][word_ngram]["test_prec"] = prec
                            results[lr][epochs][word_ngram]["test_rec"] = rec
                        except RuntimeError:
                            continue
                        pbar.update(1)
                with open(os.path.join(CFG["gpudatadir"], f"gs_ft_classif{att_string}{suffix}.pkl"), "wb") as f:
                    pkl.dump(results, f)
        if args.TEST == "True":
            with open(os.path.join(CFG["gpudatadir"], f"gs_ft_classif{att_string}{suffix}.pkl"), "rb") as f:
                results = pkl.load(f)
            best_f1_test = 0
            best_f1_train = 0
            for lr in tqdm(results.keys(), desc="Finding best model..."):
                for ep in results[lr].keys():
                    for ngram in results[lr][ep].keys():
                        if len(results[lr][ep][ngram]) > 0:
                            if results[lr][ep][ngram]["train_f1"] > best_f1_train:
                                best_f1_train = results[lr][ep][ngram]["train_f1"]
                                best_f1_train_keys = (lr, ep, ngram)
                            if results[lr][ep][ngram]["test_f1"] > best_f1_test:
                                best_f1_test = results[lr][ep][ngram]["test_f1"]
                                best_f1_test_keys = (lr, ep, ngram)
            print("Best F1 for TEST is " + str(best_f1_test*100))
            print("With params " + str(best_f1_test_keys))
            print("Best F1 for TRAIN is " + str(best_f1_train*100))
            print("With params " + str(best_f1_train_keys))
            ipdb.set_trace()
            # Best F1 for TEST is 51.734872498462416                                                                                                                                                | 0/6 [00:00<?, ?it/s]
            # With params (0.5, 50, 5)
            # Best F1 for TRAIN is 68.61821072858018
            # With params (0.5, 50, 5)


def get_suffix(args):
    suffix = '_10'
    if args.att_type == "ind":
        att_string = "ind20"
    elif args.att_type == "exp":
        att_string = f"exp{args.exp_levels}_{args.exp_type}"
    suffix += "_subInd"
    return att_string, suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--att_type", type=str, default="exp")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--suffix", type=str, default="_new_it1")
    args = parser.parse_args()
    main(args)
