import ipdb
import argparse
import yaml
from data.datasets import StringDataset
from tqdm import tqdm


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
    splits = ["TRAIN"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": 5,
                 "exp_type": "kmeans",
                 "is_toy": "False"}
    for split in splits:
        datasets_kmeans = StringDataset(**arguments, split=split)
    # building uniform experience with 5 lvls
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": 5,
                 "exp_type": "uniform",
                 "is_toy": "False"}
    for split in splits:
        datasets_uniform = StringDataset(**arguments, split=split)

    for uni, km in tqdm(zip(datasets_uniform, datasets_kmeans)):
        assert uni[0] == km[0]
    ipdb.set_trace()
    # raw_train, raw_valid, raw_test = datasets[0], datasets[1], datasets[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    args = parser.parse_args()
    init(args)
