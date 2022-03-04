import argparse
import yaml
from data.datasets import StringIndSubDataset
import ipdb
from tqdm import tqdm
import os


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        arguments = {'data_dir': CFG["gpudatadir"],
                     "load": "True",
                     "subsample": -1,
                     "max_len": 10,
                     "exp_levels": args.exp_levels,
                     "tuple_list": None,
                     "already_subbed": "True",
                     'lookup': None,
                     "suffix": args.suffix,
                     "exp_type": args.exp_type,
                     "is_toy": "False"}
        train_dataset = StringIndSubDataset(**arguments, split="TRAIN")
        valid_dataset = StringIndSubDataset(**arguments, split="VALID")
        test_dataset = StringIndSubDataset(**arguments, split="TEST")
        suffix = f"_10_subInd{args.suffix}"
        tgt_file_ind_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_ind20{suffix}.train")
        tgt_file_exp_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_exp{args.exp_levels}_{args.exp_type}{suffix}.train")
        print(tgt_file_exp_model)
        # build_train_file(args, tgt_file_ind_model, train_dataset, valid_dataset, "ind20")
        build_train_file(args, tgt_file_exp_model, train_dataset, valid_dataset, "exp")

        tgt_file_ind_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_ind20{suffix}.test")
        tgt_file_exp_model = os.path.join(CFG["gpudatadir"], f"ft_classif_supervised_exp{args.exp_levels}_{args.exp_type}{suffix}.test")
        if os.path.isfile(tgt_file_ind_model):
            os.system('rm ' + tgt_file_ind_model)
            print("removing previous file")
        # write_in_file_with_label(args, tgt_file_ind_model, test_dataset, "ind20", "test")
        write_in_file_with_label(args, tgt_file_exp_model, test_dataset, "exp", "test")
        #print("File " + tgt_file_ind_model + " built.")
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
            job_str = item[0]
            att = get_attribute(att_type, dataset, item)
            final_str = f"__label__{att} {job_str} \n"
            f.write(final_str)
            tmp.append(att)
        print(len(set(tmp)))


def get_attribute(att_type, dataset, item):
    if att_type == "ind20":
        industry_tmp = dataset.ind_dict[item[-2]]
        if len(industry_tmp.split(" ")) > 1:
            att = "_".join(industry_tmp.split(" "))
        else:
            att = industry_tmp
    elif att_type == "exp":
        att = item[-1]
    else:
        raise Exception(att_type)
    return att


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--exp_levels", type=int, default=3)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="_new_it1")
    args = parser.parse_args()
    main(args)
