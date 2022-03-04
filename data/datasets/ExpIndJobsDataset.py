import os
import torch
import ipdb
import pickle as pkl
from transformers import CamembertTokenizer, RobertaTokenizer
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class ExpIndJobsDataset(Dataset):
    def __init__(self, data_dir, light, load, subsample, max_len, is_toy, split):
        self.light = light == "True"
        self.max_len = max_len
        self.is_toy = is_toy
        self.datadir = data_dir
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(data_dir, subsample, split)
        else:
            print("Loading transformers...")
            self.tokenizer_fr = CamembertTokenizer.from_pretrained("camembert-base")
            self.tokenizer_en = RobertaTokenizer.from_pretrained("roberta-base")
            print("Transformers loaded.")
            print("Loading data files...")
            ppl_file = "profiles_jobs_fr_3_" + str(self.max_len) + "_" + split + ".pkl"
            with open(os.path.join(data_dir, ppl_file), 'rb') as f_name:
                ppl_reps = pkl.load(f_name)
            ind_file = "ind_class_dict.pkl"
            with open(os.path.join(data_dir, ind_file), 'rb') as f_name:
                industry_dict = pkl.load(f_name)
            print("Data files loaded.")
            # self.rep_dim = 300
            self.exp_dict = {0: "novice",
                             1: "junior",
                             2: "intermediary",
                             3: "senior",
                             4: "expert"}
            self.ind_dict = industry_dict
            self.rev_ind_dict = {v :k for k, v in industry_dict.items()}
            self.tuples, self.user_lookup = self.build_ppl_tuples(ppl_reps, split)
            self.save_dataset(data_dir, subsample, split)

        if subsample > 0 and not self.is_toy:
            np.random.shuffle(self.tuples)
            tmp = self.tuples[:subsample]
            self.tuples = tmp

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if self.light:
            return self.tuples[idx]["indices"], \
                   self.tuples[idx]["mask"], \
                   self.tuples[idx]["exp_index"], \
                   self.tuples[idx]["ind_index"]
        else:
            return self.tuples[idx]["indices"], \
                   self.tuples[idx]["mask"], \
                   self.tuples[idx]["exp_tokenized"], \
                   self.tuples[idx]["exp_index"], \
                   self.tuples[idx]["ind_tokenized"], \
                   self.tuples[idx]["ind_index"]

    def save_dataset(self, datadir, subsample, split):
        if self.is_toy:
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
            #ipdb.set_trace()
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = self.get_tgt_file(split, subsample)

        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, datadir, subsample, split):
        tgt_file = self.get_tgt_file(split, subsample)
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
                exp = get_uniform_experience(len(self.exp_dict), len(sorted_jobs))
                for num, job in enumerate(sorted_jobs):
                    new_job = dict()
                    new_job["ind_index"] = self.rev_ind_dict[person[1]]
                    new_job["indices"], new_job["mask"] = self.tokenize_job(job["job"])
                    new_job["exp_index"] = exp[num]
                    if not self.light:
                        new_job["exp_tokenized"] = self.tokenize_attribute(self.exp_dict[exp[num]])
                        new_job["ind_tokenized"] = self.tokenize_attribute(person[1])
                    tuples.append(new_job)
                user_lookup[id_person] = [counter, counter + len(sorted_jobs) - 1]
                counter = counter + len(sorted_jobs)
        return tuples, user_lookup

    def tokenize_job(self, job):
        tmp_job = job
        job_as_str = " ".join(tmp_job)
        outs = self.tokenizer_fr(job_as_str, max_length=self.max_len, truncation=True, return_tensors="pt")
        job_tensor = torch.ones(1, self.max_len).type(torch.LongTensor)
        mask = torch.zeros(1, self.max_len).type(torch.LongTensor)
        job_tensor[0, :outs["input_ids"].shape[-1]] = outs["input_ids"]
        mask[0, :outs["attention_mask"].shape[-1]] =  outs['attention_mask']
        return job_tensor, mask

    def tokenize_attribute(self, att):
        outs = self.tokenizer_en(att, max_length=self.max_len, truncation=True, return_tensors="pt")
        return outs["input_ids"]

    def get_tgt_file(self, split, subsample):
        if not self.light:
            tgt_file = os.path.join(self.datadir, "ExpIndJobsDataset_" + str(self.max_len) + "_" + split + ".pkl")
        elif not self.is_toy and self.light:
            tgt_file = os.path.join(self.datadir, "ExpIndJobsDataset_" + str(self.max_len) + "_" + split + "_light.pkl")
        else:
            tgt_file = os.path.join(self.datadir, "ExpIndJobsToyDataset_" + str(self.max_len) + "_" + split + "_" + str(subsample) + ".pkl")
        return tgt_file

def get_uniform_experience(exp_lvls, career_len):
    return [round(i) for i in np.linspace(0, exp_lvls-1, career_len)]

## camembert(s base special tokens
# 0 <s>NOTUSED
# 1 <pad>
# 2 </s>NOTUSED
# 3 <unk>
# 4
# 5 <s>
# 6 </s>
# 32004 <mask>