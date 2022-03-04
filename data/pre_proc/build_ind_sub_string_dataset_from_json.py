import ipdb
import argparse
import yaml
from data.datasets import StringIndSubDataset


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
    data_train, data_test = load_datasets(args)


def load_datasets(args):
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "False",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "tuple_list": None,
                 'lookup': None,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    for split in splits:
        datasets.append(StringIndSubDataset(**arguments, split=split))
    ipdb.set_trace()
    data_train_valid = datasets[0].copy()
    data_train_valid.tuples.extend(datasets[1].tuples)
    data_train_valid.lookup.extend(datasets[1].lookup)
    return data_train_valid, datasets[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
