import os
import torch
import pickle as pkl
from transformers import CamembertTokenizer, RobertaTokenizer
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset


class DeltaIndJobsCustomVocabDataset(Dataset):
    def __init__(self, data_dir, load, subsample, max_len, is_toy, split):
        self.max_len = max_len
        self.is_toy = is_toy
        self.datadir = data_dir
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.ind_dict = pkl.load(f_name)
        self.rev_ind_dict = {v: k for k, v in self.ind_dict.items()}
        with open(os.path.join(self.datadir, 'vocab_dict.pkl'), 'rb') as f:
            self.vocab_dict = pkl.load(f)
        self.rev_voc_dict = {v: k for k, v in self.vocab_dict.items()}
        with open(os.path.join(self.datadir, "delta_dict.pkl"), 'rb') as f_name:
            self.delta_dict = pkl.load(f_name)
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(subsample, split)
        else:
            self.build_dataset(split, subsample)
        if subsample > 0 and not self.is_toy:
            np.random.shuffle(self.tuples)
            tmp = self.tuples[:subsample]
            self.tuples = tmp
        print("Job dataset Built.")
        print(f"Dataset Length: {len(self.tuples)}")

    def build_dataset(self, split, subsample):
        print("Loading data files...")
        if self.is_toy:
            ppl_file = f"profiles_jobs_ind_title_delta_years_{split}_100.pkl"
        else:
            ppl_file = f"profiles_jobs_ind_title_delta_years_{split}.pkl"
        with open(os.path.join(self.datadir, ppl_file), 'rb') as f_name:
            ppl_reps = pkl.load(f_name)
        print("Data files loaded.")
        # self.rep_dim = 300
        self.tuples, self.user_lookup = self.build_ppl_tuples(ppl_reps, split)
        self.save_dataset(subsample, split)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["indices"], \
               self.tuples[idx]["mask"], \
               self.tuples[idx]["delta_index"], \
               self.tuples[idx]["ind_index"]

    def save_dataset(self, subsample, split):
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
            # ipdb.set_trace()
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = self.get_tgt_file(split, subsample)
        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, subsample, split):
        tgt_file = self.get_tgt_file(split, subsample)
        if os.path.isfile(tgt_file):
            try:
                with open(tgt_file, 'rb') as f:
                    ds_dict = pkl.load(f)
                for key in tqdm(ds_dict, desc="Loading attributes from save..."):
                    vars(self)[key] = ds_dict[key]
                print("Dataset load from : " + tgt_file)
            except EOFError:
                print(f"File {tgt_file} ran out of inputs, building dataset instead.")
                self.build_dataset(split, subsample)
        else:
            print(f"Could not locate file {tgt_file}, building dataset instead.")
            self.build_dataset(split, subsample)

    def build_ppl_tuples(self, ppl_reps, split):
        user_lookup = {}
        tuples = []
        counter = 0
        for person in tqdm(ppl_reps, desc="Building tuples for split " + split + "..."):
            id_person = person[0]
            if person[1] != "" and person[1] in self.rev_ind_dict.keys():
                sorted_jobs = sorted(person[-1], key=lambda k: k['from'])
                for num, job in enumerate(sorted_jobs):
                    new_job = dict()
                    new_job["ind_index"] = self.rev_ind_dict[person[1]]
                    new_job["indices"], new_job["mask"] = self.tokenize_job(job["job"])
                    new_job["delta_index"] = self.attribute_delta_class(job["delta_years"])
                    tuples.append(new_job)
                user_lookup[id_person] = [counter, counter + len(sorted_jobs) - 1]
                counter = counter + len(sorted_jobs)
        return tuples, user_lookup

    def tokenize_job(self, job):
        job_tensor = torch.ones(1, self.max_len).type(torch.LongTensor)
        job_tensor[0, 0] = self.rev_voc_dict["<s>"]
        for num, word in enumerate(job):
            # we check num+1 because index 0 is "<s>"
            if num+1 < self.max_len - 2:
                if word in self.rev_voc_dict.keys():
                    job_tensor[0, num+1] = self.rev_voc_dict[word]
                else:
                    job_tensor[0, num+1] = self.rev_voc_dict['<unk>']
        #  we check num+2 because index 0 is "<s>", and num+1 is last token
        if num+2 >= self.max_len:
            job_tensor[0, -1] = self.rev_voc_dict["</s>"]
        else:
            job_tensor[0, num+2] = self.rev_voc_dict["</s>"]
        if job_tensor.tolist() == [[5, 6, 1, 1, 1, 1, 1, 1, 1, 1]]:
            ipdb.set_trace()
        tmp = (job_tensor == torch.ones(1, self.max_len).type(torch.BoolTensor))
        mask = ~tmp
        return job_tensor, mask.type(torch.LongTensor)

    def get_tgt_file(self, split, subsample):
        if not self.is_toy:
            tgt_file = os.path.join(self.datadir, f"DeltaIndJobsCustomVocabDataset_{self.max_len}_{split}.pkl")
        else:
            tgt_file = os.path.join(self.datadir,
                                    f"DeltaIndJobsCustomVocabToyDataset_{self.max_len}_{split}_{subsample}.pkl")
        return tgt_file

    def attribute_delta_class(self, job_delta):
        for k in self.delta_dict.keys():
            if self.delta_dict[k]["min"] <= job_delta < self.delta_dict[k]["max"]:
                return k


## delta dict
# {0: {'min': 0, 'max': 1}, 1: {'min': 1, 'max': 3}, 2: {'min': 3, 'max': 5}, 3: {'min': 5, 'max': 10}, 4: {'min': 10, 'max': 60}}



## camembert(s base special tokens
# 0 <s>NOTUSED
# 1 <pad>
# 2 </s>NOTUSED
# 3 <unk>
# 4
# 5 <s>
# 6 </s>
# 32004 <mask>