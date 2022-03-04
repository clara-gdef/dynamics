import yaml
import ipdb
from tqdm import tqdm
from scipy.stats import describe
import argparse
from data.datasets import ExpIndJobsCustomVocabDataset
from data.datasets.DeltaJobsDataset import DeltaJobsDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    # splits = ["TRAIN", "VALID", "TEST"]
    splits = ["TRAIN"]
    with ipdb.launch_ipdb_on_exception():
        ds_delta = []
        ds_jobs = []
        ppl_file = f"profiles_jobs_skills_edu_delta_t"
        if args.gran == "years":
            ppl_file += "_years"
        for split in splits:
            arguments_delta = {"data_dir": CFG["gpudatadir"],
                               "ppl_file": f"{ppl_file}_{split}.pkl",
                               "load": "True",
                               "granularity": args.gran,
                               "split": split}
            ds_delta.append(DeltaJobsDataset(**arguments_delta))
            # arguments = {'data_dir': CFG["gpudatadir"],
            #              "light": "True",
            #              "max_len": 10,
            #              "subsample": -1,
            #              "is_toy": False,
            #              "load": "True"}
            # ds_jobs.append(ExpIndJobsCustomVocabDataset(**arguments, split=split))
            # assert ds_delta[num].user_lookup == ds_jobs[num].user_lookup
        describe_deltas(ds_delta, splits)

        ipdb.set_trace()


def describe_deltas(ds_delta, splits):
    all_deltas = []
    all_deltas_standardized = []
    for num, split in enumerate(splits):
        for tup in tqdm(ds_delta[num], desc=f"parsing dataset for split {split}"):
            all_deltas.append(tup[0])
            all_deltas_standardized.append(tup[1])
    print(describe(all_deltas))
    print(describe(all_deltas_standardized))
    ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gran", default="years")
    args = parser.parse_args()
    main(args)
