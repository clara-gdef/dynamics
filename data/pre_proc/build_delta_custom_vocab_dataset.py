import argparse
import ipdb
import yaml
from data.datasets import DeltaIndJobsCustomVocabDataset

def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for split in ["TEST", "VALID", "TRAIN"]:
            arguments = {'data_dir': CFG["gpudatadir"],
                         "max_len": args.max_len,
                         "subsample": args.subsample,
                         "is_toy": args.toy_dataset == "True",
                         "load": "False"}
            DeltaIndJobsCustomVocabDataset(**arguments, split=split)
        print("All datasets built!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--toy_dataset", default="True")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=-1)
    args = parser.parse_args()
    main(args)

