import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import StringDataset
from models.classes.CamembertRewriter import CamembertRewriter
from utils.models import collate_for_delta_bert, get_latest_model


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
    logger, checkpoint_callback, early_stop_callback, model_path = init_lightning(hparams, xp_title, model_name)
    call_back_list = [checkpoint_callback, early_stop_callback]
    if hparams.DEBUG == "True":
        trainer = pl.Trainer(gpus=1,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             precision=16
                             )
        num_workers = 0
    else:
        trainer = pl.Trainer(gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             accelerator='ddp_spawn',
                             precision=16
                             )
        num_workers = hparams.num_workers
    if hparams.TRAIN == "True":
        # TODO replace by right splits
        datasets = load_datasets(hparams, ["VALID", "VALID"], hparams.load_dataset)
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_delta_bert,
                                  num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_delta_bert,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
        print("Dataloaders initiated.")
    print("Dataloaders initiated.")
    arguments = {'emb_size': 768,
                 'hp': hparams,
                 'desc': xp_title,
                 "vocab_size": 32005,
                 "model_path": model_path,
                 "datadir": CFG["gpudatadir"],
                 "alpha": hparams.alpha,
                 "beta": hparams.beta}
    print("Initiating model...")
    model = CamembertRewriter(**arguments)
    print("Model Loaded.")
    if hparams.TRAIN == "True":
        if hparams.load_from_checkpoint == "True":
            print("Loading from previous checkpoint...")
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
            model.load_state_dict(torch.load(model_file)["state_dict"])
            print("Resuming training from checkpoint : " + model_file + ".")
            if hparams.divide_lr > 0:
                new_lr = hparams.lr / hparams.divide_lr
                model.hp.lr = new_lr
                print(f"lr updated from {new_lr*hparams.divide_lr} to {new_lr}")
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
            model.load_state_dict(torch.load(model_file))
        print("Evaluating model : " + model_file + ".")
        # TODO replace by right splits
        datasets = load_datasets(hparams, ["VALID"], hparams.load_dataset)
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=hparams.test_b_size, collate_fn=collate_for_delta_bert,
                                 num_workers=hparams.num_workers, drop_last=True)
        return trainer.test(test_dataloaders=test_loader, model=model)


def load_datasets(hparams, splits, load):
    datasets = []
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


def init_lightning(hparams, xp_title, model_name):
    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
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

    return logger, checkpoint_callback, early_stop_callback, model_path


def make_xp_title(hparams):
    xp_title = f"{hparams.model_type}_bs{hparams.b_size}_hs{hparams.dec_hs}_lr{hparams.lr}_{hparams.optim}"
    # TODO uncomment after gs eval
    if hparams.alpha != .5:
        xp_title += f"_alpha{hparams.alpha}"
    if hparams.drop_proba != .1:
        xp_title += f"_dpba{hparams.drop_proba}"
    if hparams.shuffle_dist != 3:
        xp_title += f"_shufdist{hparams.shuffle_dist}"
    # if hparams.subsample != -1:
    #     xp_title += f"sub{hparams.subsample}"
    # if hparams.no_tilde == "True":
    #     xp_title += f"_notilde"
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--prev_model_name", type=str, default=None)
    parser.add_argument("--input_recopy", default="False")
    parser.add_argument("--print_preds", type=str, default="True")
    parser.add_argument("--print_cm", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--debug_noiseless", type=str, default="False")
    parser.add_argument("--beam_search", type=str, default="True")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--no_tilde", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_to_csv", default="False")
    parser.add_argument("--toy_dataset", type=str, default="False")
    parser.add_argument("--feed_z", type=str, default="False")
    parser.add_argument("--divide_lr", type=int, default=0)
    parser.add_argument("--min_denoising_epochs", type=int, default=10)
    parser.add_argument("--denoising_only", type=str, default="False")
    # model attributes
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--test_b_size", type=int, default=64)
    parser.add_argument("--dec_hs", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--shuffle_dist", type=int, default=3)
    parser.add_argument("--drop_proba", type=float, default=.1)
    parser.add_argument("--model_type", type=str, default="camRew")
    parser.add_argument("--att_type", type=str, default="both")
    parser.add_argument("--classifier_type", type=str, default="ft")
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_val", type=float, default=100.)
    parser.add_argument("--tf", type=float, default=-1.)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=500)
    hparams = parser.parse_args()
    init(hparams)
