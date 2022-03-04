import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import ExpIndJobsCustomVocabDataset
from models.classes.Rewriter import Rewriter
#from models.classes.RewriterBert import RewriterBert
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

    logger, checkpoint_callback, early_stop_callback, model_path = init_lightning(hparams, xp_title, model_name)
    call_back_list = [checkpoint_callback, early_stop_callback]
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=call_back_list,
                         logger=logger,
                         gradient_clip_val=hparams.clip_val,
                         accelerator='ddp',
                         precision=16
                         )
    if hparams.TRAIN == "True":
        datasets = load_datasets(hparams, ["TRAIN", "VALID"], hparams.load_dataset)
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_fn,
                                  num_workers=hparams.num_workers, drop_last=True, pin_memory=True)
        print("Dataloaders initiated.")
    print("Dataloaders initiated.")
    arguments = {'emb_size': 768,
                 'hp': hparams,
                 'desc': xp_title,
                 "vocab_size": 33271,# custom vocab
                 # "vocab_size": 32005,
                 "model_path": model_path,
                 "datadir": CFG["gpudatadir"],
                 "alpha": hparams.alpha,
                 "beta": hparams.beta,
                 "save_step": hparams.save_step,
                 "mode": "train",
                 "fine_tune_word_emb": hparams.fine_tune_word_emb == "False"}
    if hparams.TEST == "True" and hparams.TRAIN == "False":
        arguments["mode"] = "eval"
    print("Initiating model...")
    model = Rewriter(**arguments)
    print("Model Loaded.")
    if hparams.TRAIN == "True":
        if hparams.load_from_checkpoint == "True":
            print("Loading from previous checkpoint...")
            if hparams.prev_model_name is None:
                model_to_load = model_name
            else:
                if hparams.prev_model_name == "sos_title_full_finetuned_lessnoise":
                    model_to_load = "sos/title/full/finetuned/lessnoise/bs30/hs512/lr0.0001/adam/"
                else:
                    raise Exception("wrong prev model name provided")
            model_path = os.path.join(CFG['modeldir'], model_to_load)
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

        trainer.fit(model.cuda(), train_loader, valid_loader)
    if hparams.TEST == "True":
        model_file = get_latest_model(CFG["modeldir"], model_name)
        if hparams.TRAIN == "False":
            print("Loading from previous run...")
            model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Evaluating model : " + model_file + ".")
        # TODO REMOVE TEST AND REPLACE BY RIGHT SPLITS
        datasets = load_datasets(hparams, ["TEST"], hparams.load_dataset)
        dataset_test = datasets[0]
        # tmp = dataset_test.tuples[:100]
        # dataset_test.tuples = tmp
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn,
                                 num_workers=hparams.num_workers, drop_last=False)
        #model.hp.b_size = 1
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
        datasets.append(ExpIndJobsCustomVocabDataset(**common_hparams, split=split))
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

    return logger, checkpoint_callback, early_stop_callback, model_path


def make_xp_title(hparams):
    xp_title = f"{hparams.model_type}_bs{hparams.b_size}_hs{hparams.hidden_size}_lr{hparams.lr}_{hparams.optim}"
    # TODO uncomment after gs eval
    if hparams.alpha != .5:
        xp_title += f"_alpha{hparams.alpha}"
    # if hparams.subsample > 0:
    #     xp_title += f"_sub{hparams.subsample}"
    if hparams.drop_proba != .1:
        xp_title += f"_dpba{hparams.drop_proba}"
    if hparams.shuffle_dist != 3:
        xp_title += f"_shufdist{hparams.shuffle_dist}"
    if hparams.no_tilde == "True":
        xp_title += f"_notilde"
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--prev_model_name", type=str, default=None)
    parser.add_argument("--input_recopy", default="False")
    parser.add_argument("--fine_tune_word_emb", default="True")
    parser.add_argument("--print_preds", type=str, default="False")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--checkpoint", type=str, default="09")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--debug_noiseless", type=str, default="False")
    parser.add_argument("--beam_search", type=str, default="True")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--no_tilde", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--light", default="True")
    parser.add_argument("--print_to_csv", default="True")
    parser.add_argument("--toy_dataset", type=str, default="False")
    # model attributes
    parser.add_argument("--b_size", type=int, default=30)
    parser.add_argument("--save_step", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--pooling_window", type=int, default=5)
    parser.add_argument("--shuffle_dist", type=int, default=3)
    parser.add_argument("--drop_proba", type=float, default=.1)
    parser.add_argument("--model_type", type=str, default="title_exp_only_fs")
    parser.add_argument("--att_type", type=str, default="ind")
    parser.add_argument("--classifier_type", type=str, default="SVM")
    # global hyper params
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_val", type=float, default=1.)
    parser.add_argument("--tf", type=float, default=-1.)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=500)
    hparams = parser.parse_args()
    init(hparams)
