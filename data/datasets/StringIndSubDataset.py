import os
import pickle as pkl

import ipdb
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class StringIndSubDataset(Dataset):
    def __init__(self, data_dir, load, subsample, max_len, is_toy, exp_levels, exp_type, tuple_list, lookup, split, already_subbed="True", suffix=""):
        self.max_len = max_len
        self.is_toy = is_toy
        self.subsample = subsample
        self.split = split
        self.datadir = data_dir
        self.exp_levels = exp_levels
        self.exp_type = exp_type
        self.name = f"StringIndSubDataset_{self.exp_levels}exp_{self.exp_type}_no_unk_{split}{suffix}"
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(subsample, split, "")
        else:
            print("Loading data files...")
            if exp_type == "delta":
                with open(os.path.join(self.datadir, f"delta_{self.exp_levels}_dict.pkl"), 'rb') as f_name:
                    self.delta_dict = pkl.load(f_name)
            else:
                with open(os.path.join(self.datadir, f"exp_dict_fr_{self.exp_levels}.pkl"), 'rb') as f_name:
                    self.exp_dict = pkl.load(f_name)
            ind_file = "20_industry_dict.pkl"
            with open(os.path.join(data_dir, ind_file), 'rb') as f_name:
                industry_dict = pkl.load(f_name)
            print("Data files loaded.")
            with open(os.path.join(self.datadir, "ind_map_to_subsampled.pkl"), 'rb') as f:
                self.ind_map_to_subsampled = pkl.load(f)
            self.rev_ind_map_to_subsampled = {v: k for k, v in self.ind_map_to_subsampled.items()}
            self.ind_dict = industry_dict
            self.rev_ind_dict = {v: k for k, v in industry_dict.items()}
            self.tuples, self.user_lookup = self.build_ppl_tuples(tuple_list, lookup, already_subbed, split)
            if self.exp_type == "uniform":
                self.check_monotonicity()
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
        print(f"Dataset saved at {tgt_file}")

    def load_dataset(self, subsample, split, suffix):
        tgt_file = self.get_tgt_file(suffix, subsample)
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)

        for key in tqdm(ds_dict, desc="Loading attributes from save..."):
            vars(self)[key] = ds_dict[key]
        print("Dataset load from : " + tgt_file)

    def build_ppl_tuples(self, tuple_list, lookup, already_subbed, split):
        if already_subbed == "False":
            for blob in tuple_list:
                assert blob["ind_index"] in self.rev_ind_map_to_subsampled.keys()
        else:
            for blob in tuple_list:
                assert (blob["ind_index"] < 20)
        user_lookup = {}
        tuples = []
        counter = 0
        for person_id, (start, end) in tqdm(lookup.items(), desc="Building tuples for split " + split + "..."):
            try:
                id_person = person_id
                num_jobs = end - start
                exp_unif = self.get_uniform_experience(num_jobs)
                if already_subbed == "False":
                    end += 1
                    num_jobs += 1
                for num, i in enumerate(range(start, end)):
                    new_job = tuple_list[i].copy()
                    if already_subbed == "False":
                        tmp = new_job["ind_index"]
                        new_job["ind_index"] = self.rev_ind_map_to_subsampled[tmp]
                        new_job["exp_index"] = exp_unif[num]
                    tuples.append(new_job.copy())
                user_lookup[id_person] = [counter, counter + num_jobs]
                counter = counter + num_jobs
            except IndexError:
                if end == 107493:
                    continue
                else:
                    ipdb.set_trace()
        return tuples, user_lookup

    def get_tgt_file(self, suffix, subsample):
        if self.is_toy != "True":
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}.pkl")
        else:
            tgt_file = os.path.join(self.datadir, f"{self.name}{suffix}_{subsample}.pkl")
        return tgt_file

    def get_uniform_experience(self, career_len):
        return [round(i) for i in np.linspace(0, self.exp_levels - 1, career_len)]

    def check_monotonicity(self):
        all_labels = [i["exp_index"] for i in self.tuples]
        for cnt, user in enumerate(tqdm(self.user_lookup.keys(), desc=f"Checking monotonicity of experience for users...")):
            current_user = self.user_lookup[user]
            exp_seq_init = []
            for job in range(current_user[0], current_user[1]):
                exp_seq_init.append(all_labels[job])
            assert all(exp_seq_init[i] <= exp_seq_init[i + 1] for i in range(len(exp_seq_init) - 1))
        print("all experience are monotonic")


## camembert(s base special tokens
# 0 <s>NOTUSED
# 1 <pad>
# 2 </s>NOTUSED
# 3 <unk>
# 4
# 5 <s>
# 6 </s>
# 32004 <mask>