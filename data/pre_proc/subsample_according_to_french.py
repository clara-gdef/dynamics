import argparse
import os
import pickle as pkl
import ipdb
import yaml
from tqdm import tqdm
import fastText


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        ppl_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_skills_edu")
        tgt_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_skills_edu_fr_only")
        ft_model = fastText.load_model(os.path.join(CFG["modeldir"], "lid.176.bin"))
        with ipdb.launch_ipdb_on_exception():
            for split in ["_TRAIN", "_VALID", "_TEST"]:
                with open(ppl_file + split + ".pkl", 'rb') as fp:
                    data = pkl.load(fp)
                dataset = subsample_dataset(data, ft_model, split)
                print("Ratio of original dataset kept: " + str(100 * len(dataset)/ len(data)) + " %")
                with open(tgt_file + split + ".pkl", 'wb') as f:
                    pkl.dump(dataset, f)

def subsample_dataset(data, ft_model, split):
    fr_dataset = []
    for person in tqdm(data, desc="Parsing dataset for split " + str(split) + "..."):
        language_pred = identify_profile_language(person, ft_model)
        if language_pred[0][0][0] == "__label__fr" and  language_pred[1][0][0] > .9:
            fr_dataset.append(person)
    return fr_dataset

def identify_profile_language(person, ft_model):
    jobs_str = ""
    for j in person[-1]:
        jobs_str  += " ".join(j["job"])
    ft_model.predict([jobs_str])
    return ft_model.predict([jobs_str])



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)