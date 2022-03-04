import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import ExpIndJobsDataset
from models.classes.Rewriter import Rewriter
from utils.models import collate_for_rewriter, collate_for_rewriter_light, get_latest_model
from transformers import CamembertModel, CamembertTokenizer, CamembertForCausalLM, CamembertConfig


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(hparams)
    else:
        return main(hparams)


def main(hparams):
    xp_title = make_xp_title(hparams)
    model_name = "/".join(xp_title.split('_'))
    if hparams.light == "True":
        collate_fn = collate_for_rewriter_light
    else:
        collate_fn = collate_for_rewriter

    model_path = init_lightning(hparams, model_name)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=[MyPrintingCallback()],
                         gradient_clip_val=hparams.clip_val
                         )

    print("Dataloaders initiated.")
    arguments = {'emb_size': 768,
                 'hp': hparams,
                 'desc': xp_title,
                 "vocab_size": 32005,
                 "model_path": model_path,
                 "datadir": CFG["gpudatadir"],
                 "alpha": hparams.alpha,
                 "beta": hparams.beta,
                 "save_step": hparams.save_step,
                 "mode": "eval",
                 "fine_tune_word_emb": hparams.fine_tune_word_emb == "False"}
    print("Initiating model...")
    model = Rewriter(**arguments)
    print("Model Loaded.")

    model_file = get_latest_model(CFG["modeldir"], model_name)
    print("Loading from previous run...")
    tmp = torch.load(model_file, map_location=torch.device(0))
    model.load_state_dict(tmp["state_dict"])
    print("Evaluating model : " + model_file + ".")
    datasets = load_datasets(hparams, ["TEST"], "True")
    dataset_test = datasets[0]
    dataset_test.tuples = datasets[0].tuples[:100]

    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn,
                             num_workers=hparams.num_workers, drop_last=False)
    model.hp.b_size = 1
    return trainer.test(test_dataloaders=test_loader, model=model)


def load_datasets(hparams, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "load": load,
        "light": hparams.light,
        "is_toy": hparams.toy_dataset == "True",
        "max_len": hparams.max_len,
        "subsample": hparams.subsample
    }
    for split in splits:
        datasets.append(ExpIndJobsDataset(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, model_name):
    model_path = os.path.join(CFG['modeldir'], model_name)
    return model_path


def make_xp_title(hparams):
    xp_title = str(hparams.model_type) + "_bs" + str(hparams.b_size) + "_hs" + str(hparams.hidden_size) + "_lr" + str(
        hparams.lr) + "_" + hparams.optim
    # if hparams.subsample > 0:
    #     xp_title += "_sub" + str(hparams.subsample)
    print("xp_title = " + xp_title)
    return xp_title


class MyPrintingCallback(Callback):
    def on_test_epoch_start(self, trainer, model):
        # https://huggingface.co/transformers/model_doc/roberta.html#transformers.RobertaForCausalLM.forward
        config = CamembertConfig.from_pretrained("camembert-base")
        config.is_decoder = True
        model.ppl_model = CamembertForCausalLM.from_pretrained("camembert-base", config=config).cuda()
        print("added causal LM model to main model for evaluation")
        model.test_decoded_outputs = []
        model.test_decoded_labels = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--fine_tune_word_emb", default="False")
    parser.add_argument("--print_preds", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--TRAIN", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--light", default="True")
    parser.add_argument("--toy_dataset", type=str, default="True")
    # model attributes
    parser.add_argument("--para", type=str, default="False")
    parser.add_argument("--b_size", type=int, default=42)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pooling_window", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="rewriter")
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_val", type=float, default=1.)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=10)
    hparams = parser.parse_args()
    init(hparams)
