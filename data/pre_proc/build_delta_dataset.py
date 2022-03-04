import argparse
import yaml
import ipdb
from data.datasets.DeltaJobsDataset import DeltaJobsDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        ds = []
        ppl_file = f"profiles_jobs_skills_edu_delta_t"
        if args.gran == "years":
            ppl_file += "_years"
        for split in ["TRAIN", "VALID", "TEST"]:
            arguments_delta = {"data_dir": CFG["gpudatadir"],
                               "ppl_file":f"{ppl_file}_{split}.pkl",
                               "load": False,
                               "granularity": args.gran,
                               "split": split}
            ds.append(DeltaJobsDataset(**arguments_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gran", default="years")
    args = parser.parse_args()
    main(args)
