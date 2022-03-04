import os
import torch
import ipdb
import pickle as pkl
from transformers import CamembertTokenizer, RobertaTokenizer
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class StringDataset(Dataset):
    def __init__(self, data_dir, load, subsample, max_len, is_toy, exp_levels, exp_type, split):
        self.max_len = max_len
        self.is_toy = is_toy
        self.subsample = subsample
        self.split = split
        self.datadir = data_dir
        self.exp_levels = exp_levels
        self.exp_type = exp_type
        self.name = f"StringDataset_{self.exp_levels}exp_{self.exp_type}_no_unk_{split}"
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(subsample, split, "")
        else:
            print("Loading data files...")
            if exp_type == "delta":
                with open(os.path.join(self.datadir, f"delta_{self.exp_levels}_dict.pkl"), 'rb') as f_name:
                    self.delta_dict = pkl.load(f_name)
                ppl_file = f"profiles_jobs_ind_title_delta_years_{split}.pkl"
                with open(os.path.join(data_dir, ppl_file), 'rb') as f_name:
                    ppl_reps = pkl.load(f_name)
            else:
                with open(os.path.join(self.datadir, f"exp_dict_fr_{self.exp_levels}.pkl"), 'rb') as f_name:
                    self.exp_dict = pkl.load(f_name)
                ppl_file = f"profiles_jobs_ind_title_{split}.pkl"
                with open(os.path.join(data_dir, ppl_file), 'rb') as f_name:
                    ppl_reps = pkl.load(f_name)
            ind_file = "ind_class_dict.pkl"
            with open(os.path.join(data_dir, ind_file), 'rb') as f_name:
                industry_dict = pkl.load(f_name)
            # with open(os.path.join(self.datadir, 'vocab_dict.pkl'), 'rb') as f:
            #     self.vocab_dict = pkl.load(f)
            # self.rev_voc_dict = {v: k for k, v in self.vocab_dict.items()}
            print("Data files loaded.")
            # self.rep_dim = 300
            self.ind_dict = industry_dict
            self.rev_ind_dict = {v: k for k, v in industry_dict.items()}
            self.tuples, self.user_lookup = self.build_ppl_tuples(ppl_reps, split)
            self.save_dataset("", subsample)

        if subsample != -1 and self.is_toy == "False":
            np.random.shuffle(self.tuples)
            tmp = self.tuples[:subsample]
            self.tuples = tmp

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if "exp_index" in self.tuples[idx].keys():
            return self.tuples[idx]["words"], \
                self.tuples[idx]["ind_index"], \
                self.tuples[idx]["exp_index"]
        elif "delta_index" in self.tuples[idx].keys():
            return self.tuples[idx]["words"], \
                self.tuples[idx]["ind_index"], \
                self.tuples[idx]["delta_index"]
        else:
            return self.tuples[idx]["words"], \
               self.tuples[idx]["ind_index"]

    def save_dataset(self, suffix, subsample):
        if self.is_toy == "True":
            print("Subsampling dataset...")
            new_tuples = []
            np.random.shuffle(self.tuples)
            retained_tuples = self.tuples[:subsample]
            for i in range(100):
                if len(retained_tuples) == 1:
                    new_tuples.append(retained_tuples[0])
                else:
                    new_tuples.extend(retained_tuples)
            self.tuples = new_tuples
            print(f"len tuples in save_dataset: {len(self.tuples)}")
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = self.get_tgt_file(suffix, subsample)

        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, subsample, split, suffix):
        tgt_file = self.get_tgt_file(suffix, subsample)
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)

        for key in tqdm(ds_dict, desc="Loading attributes from save..."):
            vars(self)[key] = ds_dict[key]
        print("Dataset load from : " + tgt_file)

    def build_ppl_tuples(self, ppl_reps, split):
        user_lookup = {}
        tuples = []
        counter = 0
        for person in tqdm(ppl_reps, desc="Building tuples for split " + split + "..."):
            id_person = person[0]
            if person[1] != "" and person[1] in self.rev_ind_dict.keys():
                sorted_jobs = sorted(person[-1], key=lambda k: k['from'])
                if self.exp_type == "uniform":
                    exp = self.get_uniform_experience(len(sorted_jobs))
                for num, job in enumerate(sorted_jobs):
                    new_job = dict()
                    new_job["ind_index"] = self.rev_ind_dict[person[1]]
                    if self.exp_type == "uniform":
                        new_job["exp_index"] = exp[num]
                    if self.exp_type == "delta":
                        new_job["delta_index"] = self.attribute_delta_class(job["delta_years"])
                    new_job["words"] = self.tokenize_job(job["job"])
                    tuples.append(new_job)
                user_lookup[id_person] = [counter, counter + len(sorted_jobs) - 1]
                counter = counter + len(sorted_jobs)
        return tuples, user_lookup

    def get_tgt_file(self, suffix, subsample):
        if self.is_toy != "True":
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}.pkl")
        else:
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}_{subsample}.pkl")
        return tgt_file

    def tokenize_job(self, job):
        word_list = []
        for num, word in enumerate(job):
            if num < self.max_len - 3:
                # if word in self.rev_voc_dict.keys():
                word_list.append(word)
                # else:
                #     word_list.append("unk")
        return " ".join(word_list)

    def save_with_new_labels(self, new_labels, suffix):
        assert len(new_labels) == len(self.tuples)
        new_tuples = []
        for num, tup in tqdm(enumerate(self.tuples), desc="adding new labels to dataset..."):
            new_tup = tup
            new_tup["exp_index"] = new_labels[num]
            new_tuples.append(new_tup)
        self.tuples = new_tuples
        self.save_dataset(suffix, self.subsample)

    def get_uniform_experience(self, career_len):
        return [round(i) for i in np.linspace(0, self.exp_levels - 1, career_len)]

    def attribute_delta_class(self, job_delta):
        for k in self.delta_dict.keys():
            if self.delta_dict[k]["min"] <= job_delta < self.delta_dict[k]["max"]:
                return k

## camembert(s base special tokens
# 0 <s>NOTUSED
# 1 <pad>
# 2 </s>NOTUSED
# 3 <unk>
# 4
# 5 <s>
# 6 </s>
# 32004 <mask>