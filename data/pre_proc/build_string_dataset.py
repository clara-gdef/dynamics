import ipdb
import argparse
import yaml
from data.datasets import StringDataset


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(args)
    else:
        return main(args)


def main(args):
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "False",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    for split in splits:
        datasets.append(StringDataset(**arguments, split=split))
    # # building uniform experience with 5 lvls
    # arguments = {'data_dir': CFG["gpudatadir"],
    #              "load": "False",
    #              "subsample": -1,
    #              "max_len": 10,
    #              "exp_levels": 5,
    #              "exp_type": "uniform",
    #              "is_toy": "False"}
    # for split in splits:
    #     datasets.append(StringDataset(**arguments, split=split))
    # arguments = {'data_dir': CFG["gpudatadir"],
    #              "load": "False",
    #              "subsample": -1,
    #              "max_len": 10,
    #              "exp_levels": 3,
    #              "exp_type": "uniform",
    #              "is_toy": "False"}
    # for split in splits:
    #     datasets.append(StringDataset(**arguments, split=split))

    #raw_train, raw_valid, raw_test = datasets[0], datasets[1], datasets[2]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="delta")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
