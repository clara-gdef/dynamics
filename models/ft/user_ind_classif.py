import argparse
import yaml
from data.datasets import ExpIndJobsCustomVocabDataset, StringDataset
import ipdb
from tqdm import tqdm
import fastText
import os
from utils.models import indices_to_words
from collections import Counter


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        train_file, test_file = build_or_load_files(args)

def build_or_load_files(args):
    tgt_train_file_ind = os.path.join(CFG["gpudatadir"], "ft_classif_USER_supervised_ind.train")
    tgt_test_file_ind = os.path.join(CFG["gpudatadir"], "ft_classif_USER_supervised_ind.test")
    if args.force_rebuild == "True":
        os.system('rm ' + tgt_train_file_ind)
        print("Removed " + tgt_train_file_ind)
        os.system('rm ' + tgt_test_file_ind)
        print("Removed " + tgt_test_file_ind)
    if not os.path.isfile(tgt_train_file_ind) or not os.path.isfile(tgt_test_file_ind):
        print("Files not found, build txt file for traning and evaluation")
        build_txt_file(tgt_train_file_ind, tgt_test_file_ind)
    return tgt_train_file_ind, tgt_test_file_ind


def build_txt_file(tgt_train_file_ind, tgt_test_file_ind):
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": 5,
                 "exp_type": "delta",
                 "is_toy": "False"}
    train_dataset = StringDataset(**arguments, split="TRAIN")
    valid_dataset = StringDataset(**arguments, split="VALID")
    test_dataset = StringDataset(**arguments, split="TEST")

    build_train_file(args, tgt_train_file_ind, train_dataset, valid_dataset, "ind")

    tgt_file_ind_model = os.path.join(CFG["gpudatadir"], "ft_classif_USER_supervised_ind.test")
    if os.path.isfile(tgt_file_ind_model):
        os.system('rm ' + tgt_file_ind_model)
        print("removing previous file")
    write_in_file_with_label(args, tgt_test_file_ind, test_dataset, "ind", "test")
    print("File " + tgt_file_ind_model + " built.")
    return tgt_train_file_ind, tgt_test_file_ind


def build_train_file(args, tgt_file, train_dataset, valid_dataset, att_type):
    if os.path.isfile(tgt_file):
        os.system('rm ' + tgt_file)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file, train_dataset, att_type, "train")
    write_in_file_with_label(args, tgt_file, valid_dataset, att_type, "valid")
    print("File " + tgt_file + " built.")


def write_in_file_with_label(args, tgt_file, dataset, att_type, split):
    lookup = dataset.user_lookup
    with open(tgt_file, 'a+') as f:
        tmp = []
        tmp_ind = []
        for user in tqdm(lookup.keys(), desc=f"Parsing {split} dataset for {att_type}..."):
            start_job = lookup[user][0]
            end_job = lookup[user][1]
            prof_str = ""
            for job_num in range(start_job, end_job):
                job_str = dataset[job_num][0]
                att = get_attribute(att_type, dataset, dataset[job_num])
                prof_str += f" {job_str}."
                tmp.append(att)
                tmp_ind.append(dataset[job_num][-2])
            final_str = f"__label__{att} {prof_str} \n"
            f.write(final_str)
    print(len(set(tmp)))


def get_attribute(att_type, dataset, item):
    if att_type == "ind":
        industry_tmp = dataset.ind_dict[item[-2]]
        if len(industry_tmp.split(" ")) > 1:
            att = "_".join(industry_tmp.split(" "))
        else:
            att = industry_tmp
    elif att_type == "exp":
        att = dataset.exp_dict[item[-1]]
    elif att_type == "delta":
        for k in dataset.delta_dict.keys():
            if dataset.delta_dict[k]["min"] <= item[-1] < dataset.delta_dict[k]["max"]:
                att = k
    else:
        raise Exception("Wrong att type specified! Can be \"ind\" or \"exp\" got: " + att_type)
    return att


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--force_rebuild", default="True")
    parser.add_argument("--max_len", type=int, default=10)
    args = parser.parse_args()
    main(args)
