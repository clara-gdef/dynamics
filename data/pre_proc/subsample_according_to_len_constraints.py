import os
import pickle as pkl
import ipdb
import argparse
import yaml
from tqdm import tqdm


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    max_len = args.max_len
    min_len = args.min_len
    ppl_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_skills_edu_fr_only")
    tgt_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_fr_" + str(min_len) + "_" + str(max_len))
    with ipdb.launch_ipdb_on_exception():
        for split in ["_TRAIN", "_VALID", "_TEST"]:
            with open(ppl_file + split + ".pkl", 'rb') as fp:
                data = pkl.load(fp)
            dataset = build_dataset(data, max_len, min_len, split)
            with open(tgt_file + split + ".pkl", 'wb') as f:
                pkl.dump(dataset, f)
            print("File saved at location " + tgt_file + split + ".pkl")


def build_dataset(data, max_len, min_len, split):
    retained_profiles = []
    num_jobs_retained = 0
    num_jobs_total = 0
    for person in tqdm(data):
        flag = False
        for job in person[-1]:
            if len(job["job"]) < min_len:
                flag = True
        if not flag:
            # we removed user with too short jobs. Now we trim the longest ones.
            new_person = trim_jobs_to_max_len(person, max_len)
            num_jobs_retained += len(person[-1])
            retained_profiles.append(new_person)
        num_jobs_total += len(person[-1])
    print("ratio of USERS retained: " + str(100*len(retained_profiles)/len(data)) + str(" for split " + str(split)))
    print("ratio of JOBS retained: " + str(100*num_jobs_retained/num_jobs_total) + str(" for split " + str(split)))
    return retained_profiles


def trim_jobs_to_max_len(person, max_len):
    new_p = person
    for num, job in enumerate(person[-1]):
        final_len = min(len(job["job"]), max_len)
        new_p[-1][num]["job"] = job["job"][:final_len]
    return new_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_len", type=int, default=3)
    args = parser.parse_args()
    main(args)

