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

        if args.exp_type == "delta":
            arguments = {'data_dir': CFG["gpudatadir"],
                         "load": "False",
                         "subsample": -1,
                         "max_len": 10,
                         "exp_levels": 5,
                         "exp_type": "delta",
                         "is_toy": "False"}
            train_dataset = StringDataset(**arguments, split="TRAIN")
            valid_dataset = StringDataset(**arguments, split="VALID")
            test_dataset = StringDataset(**arguments, split="TEST")

        else:
            arguments = {'data_dir': CFG["gpudatadir"],
                         "light": args.light,
                         "max_len": args.max_len,
                         "is_toy": False,
                         "subsample": 0,
                         "load": "True"}
            train_dataset = ExpIndJobsCustomVocabDataset(**arguments, split="TRAIN")
            valid_dataset = ExpIndJobsCustomVocabDataset(**arguments, split="VALID")
            test_dataset = ExpIndJobsCustomVocabDataset(**arguments, split="TEST")
        if args.max_len == 10:
            suffix = '_10'
        else:
            suffix = ""
        # tgt_file_ind_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_ind{suffix}.train")
        # build_train_file(args, tgt_file_ind_model, train_dataset, valid_dataset, "ind")
        #
        # ipdb.set_trace()
        #
        # tgt_file_ind_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_ind{suffix}.test")
        # if os.path.isfile(tgt_file_ind_model):
        #     os.system('rm ' + tgt_file_ind_model)
        #     print("removing previous file")
        # write_in_file_with_label(args, tgt_file_ind_model, test_dataset, "ind", "test")
        # print("File " + tgt_file_ind_model + " built.")

        tgt_file_exp_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{args.exp_type}_5{suffix}.train")
        build_train_file(args, tgt_file_exp_model, train_dataset, valid_dataset, args.exp_type)
        ipdb.set_trace()

        tgt_file_exp_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_{args.exp_type}_5{suffix}.test")
        if os.path.isfile(tgt_file_exp_model):
            os.system('rm ' + tgt_file_exp_model)
            print("removing previous file")
        write_in_file_with_label(args, tgt_file_exp_model, test_dataset, args.exp_type, "test")
        print("File " + tgt_file_exp_model + " built.")


def build_train_file(args, tgt_file, train_dataset, valid_dataset, att_type):
    if os.path.isfile(tgt_file):
        os.system('rm ' + tgt_file)
        print("removing previous file")
    write_in_file_with_label(args, tgt_file, train_dataset, att_type, "train")
    write_in_file_with_label(args, tgt_file, valid_dataset, att_type, "valid")
    print("File " + tgt_file + " built.")


def write_in_file_with_label(args, tgt_file, dataset, att_type, split):
    with open(tgt_file, 'a+') as f:
        tmp = []
        for item in tqdm(dataset, desc="Parsing train " + split + " dataset for " + att_type + "..."):
            if args.exp_type == "delta":
                job_str = item[0]
            else:
                job_str = indices_to_words(item[0][0], dataset.vocab_dict)
            att = get_attribute(att_type, dataset, item)
            final_str = f"__label__{att} {job_str} \n"
            f.write(final_str)
            tmp.append(att)
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
        att = item[-1]
    else:
        raise Exception("Wrong att type specified! Can be \"ind\" or \"exp\" got: " + att_type)
    return att


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--exp_type", type=str, default="delta") #"kmeans" or "uniform" or "kmeansrdm" or "delta"
    args = parser.parse_args()
    main(args)
