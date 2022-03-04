import argparse
import ipdb
import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader

from models.classes import RewriterDeltaBert
from models.train_delta_bert_rewriter import make_xp_title, init_lightning, get_latest_model, load_datasets
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
        initial_jobs, initial_exp, initial_ind = [], [], []
        decoded_tokens, expected_delta, expected_ind = [], [], []
        # get N predictions for a given sample
        for i in tqdm(range(hparams.num_predictions), desc=f"Computing {hparams.num_predictions*hparams.num_tilde} predictions..."):
            batch = test_loader.sampler.data_source[random.randint(0, len(test_loader)-1)]
            sentences, ind_indices, delta_indices = batch[0], batch[1], batch[2]
            for j in range(hparams.num_tilde):
                dlt_tensor = (torch.zeros(1) + delta_indices).long().cuda()
                ind_tensor = (torch.zeros(1) + ind_indices).long().cuda()
                dec_outs, delta_tilde_indices, ind_tilde_indices = model.get_predictions([sentences],
                                                                                         ind_tensor,
                                                                                         dlt_tensor)
                # ipdb.set_trace()
                initial_jobs.append(sentences)
                decoded_tokens.append(dec_outs)
                if hparams.att_type == "ind" or hparams.att_type == "both":
                    initial_ind.append(ind_indices)
                    expected_ind.append(ind_tilde_indices)
                if hparams.att_type == "exp" or hparams.att_type == "both":
                    initial_exp.append(delta_indices)
                    expected_delta.append(delta_tilde_indices)
        # check tokens, they look weird
        reshaped_predicted_tokens = torch.stack(decoded_tokens).view(-1, decoded_tokens[0].shape[-1])
        rev_ind_dict = dict()
        for k, v in model.industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        ind_preds, delta_preds = model.get_att_prediction(reshaped_predicted_tokens, "FT", rev_ind_dict)
        # to file
        file_prefix = "TEST_split_prediction_exploration"
        decoded_sentences = [model.tokenizer.decode(i.long()[0], skip_special_tokens=True) for i in decoded_tokens]
        #ipdb.set_trace()
        csv_file = print_tilde_to_csv(hparams.att_type, initial_jobs, initial_exp, initial_ind, decoded_sentences,
                                      expected_delta, expected_ind, delta_preds, ind_preds,
                                      f"{file_prefix}_{xp_title}", model.industry_dict, model.delta_dict)
        df = pd.read_csv(csv_file)
        df.to_html(f'html/{file_prefix}_{xp_title}.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--prev_model_name", type=str, default=None)
    parser.add_argument("--eval_mode", type=str, default="latest")  # can be "spe" or "latest"
    parser.add_argument("--model_to_test", type=str, default="epoch=3-step=14191.tmp_end.ckpt")
    parser.add_argument("--input_recopy", default="False")
    parser.add_argument("--print_preds", type=str, default="False")
    parser.add_argument("--num_predictions", type=int, default=100)
    parser.add_argument("--num_tilde", type=int, default=3)
    parser.add_argument("--print_cm", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--debug_noiseless", type=str, default="False")
    parser.add_argument("--beam_search", type=str, default="False")
    parser.add_argument("--no_tilde", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_to_csv", default="False")
    parser.add_argument("--toy_dataset", type=str, default="True")
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
    parser.add_argument("--model_type", type=str, default="deltaBert2LayFullSeqPooled")
    parser.add_argument("--pooling_window", type=int, default=5)
    parser.add_argument("--att_type", type=str, default="both")
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
