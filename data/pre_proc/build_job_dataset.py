import argparse
import ipdb
import yaml
from data.datasets import ExpIndJobsDataset

def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for split in ["VALID", "TEST", "TRAIN"]:
            arguments = {'data_dir': CFG["gpudatadir"],
                         "light": args.light,
                         "max_len": args.max_len,
                         "subsample": 0,
                         "is_toy": False,
                         "load": "False"}
            ExpIndJobsDataset(**arguments, split=split)
        print("All datasets built!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    main(args)

