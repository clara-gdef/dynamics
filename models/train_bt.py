import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import ExpIndJobsDataset
from models.classes.BackTranslator import BackTranslator
from utils.models import collate_for_rewriter, collate_for_rewriter_light, get_latest_model


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

    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, xp_title, model_name)
    call_back_list = [checkpoint_callback, early_stop_callback]
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=call_back_list,
                         logger=logger,
                         gradient_clip_val=hparams.clip_val,
                         accelerator='dp'
                         )
    if hparams.TRAIN == "True":
        datasets = load_datasets(hparams, ["VALID", "VALID"], hparams.load_dataset)
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=hparams.num_workers, drop_last=True, pin_memory=True)
    print("Dataloaders initiated.")
    arguments = {'emb_size': 768,
                 'hp': hparams,
                 'desc': xp_title,
                 "vocab_size": 32005,
                 "datadir": CFG["gpudatadir"],
                 "fine_tune_word_emb": hparams.fine_tune_word_emb == "True"}

    print("Initiating model...")
    model = BackTranslator(**arguments)
    print("Model Loaded.")
    if hparams.TRAIN == "True":
        if hparams.load_from_checkpoint == "True":
            print("Loading from previous checkpoint...")
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
            model.load_state_dict(torch.load(model_file)["state_dict"])
            print("Resuming training from checkpoint : " + model_file + ".")

        if hparams.auto_lr_find == "True":
            print("looking for best lr...")
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

            # Results can be found in
            print(lr_finder.results)

            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()

            # update hparams of the model
            model.hp.lr = new_lr
            ipdb.set_trace()
        print("Starting training for " + xp_title + "...")

        trainer.fit(model, train_loader, valid_loader)
    if hparams.TEST == "True":
        model_file = get_latest_model(CFG["modeldir"], model_name)
        if hparams.TRAIN == "False":
            print("Loading from previous run...")
            model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Evaluating model : " + model_file + ".")
        datasets = load_datasets(hparams, ["VALID"], "True")
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn,
                                  num_workers=hparams.num_workers,  shuffle=True, drop_last=True)
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


def init_lightning(hparams, xp_title, model_name):
    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_path, '{epoch:02d}'),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    print("callback initiated.")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback


def make_xp_title(hparams):
    xp_title = str(hparams.model_type) + "_bs" + str(hparams.b_size) + "_hs" + str(hparams.hidden_size) + "_lr" + str(hparams.lr) + "_" + hparams.optim
    if hparams.subsample > 0:
        xp_title += "_sub" + str(hparams.subsample)
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--fine_tune_word_emb", default="False")
    parser.add_argument("--print_preds", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--TEST", type=str, default="True")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--light", default="True")
    parser.add_argument("--toy_dataset", type=str, default="True")
    # model attributes
    parser.add_argument("--para", type=str, default="False")
    parser.add_argument("--b_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pooling_window", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="bt")
    # global hyper params
    parser.add_argument("--clip_val", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=500)
    hparams = parser.parse_args()
    init(hparams)
