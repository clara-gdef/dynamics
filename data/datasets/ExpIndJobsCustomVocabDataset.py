import os
import torch
import pickle as pkl
from transformers import CamembertTokenizer, RobertaTokenizer
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset


class ExpIndJobsCustomVocabDataset(Dataset):
    def __init__(self, data_dir, light, load, subsample, max_len, is_toy, split):
        self.light = light == "True"
        self.max_len = max_len
        self.is_toy = is_toy
        self.datadir = data_dir
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
        print("Loading transformers...")
        self.tokenizer_en = RobertaTokenizer.from_pretrained("roberta-base")
        print("Transformers loaded.")
        print("Loading data files...")
        if self.max_len == 10:
            self.title = True
            if self.is_toy:
                ppl_file = f"profiles_jobs_ind_title_{split}_100.pkl"
            else:
                ppl_file = f"profiles_jobs_ind_title_{split}.pkl"
        else:
            self.title = False
            ppl_file = f"profiles_jobs_fr_3_{self.max_len}_{split}.pkl"
        with open(os.path.join(self.datadir, ppl_file), 'rb') as f_name:
            ppl_reps = pkl.load(f_name)
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.ind_dict = pkl.load(f_name)
        with open(os.path.join(self.datadir, 'vocab_dict.pkl'), 'rb') as f:
            self.vocab_dict = pkl.load(f)
        self.rev_voc_dict = {v: k for k, v in self.vocab_dict.items()}
        print("Data files loaded.")
        # self.rep_dim = 300
        with open(os.path.join(self.datadir, "exp_dict_en_5.pkl"), 'rb') as f_name:
            self.exp_dict = pkl.load(f_name)
        self.rev_ind_dict = {v: k for k, v in self.industry_dict.items()}
        self.tuples, self.user_lookup = self.build_ppl_tuples(ppl_reps, split)
        self.save_dataset(subsample, split)

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

    def tokenize_attribute(self, att):
        outs = self.tokenizer_en(att, max_length=self.max_len, truncation=True, return_tensors="pt")
        return outs["input_ids"]

    def get_tgt_file(self, split, subsample):
        if not self.light:
            tgt_file = os.path.join(self.datadir, f"ExpIndJobsCustomVocabDataset_{self.max_len}_{split}_.pkl")
        elif not self.is_toy and self.light:
            tgt_file = os.path.join(self.datadir, f"ExpIndJobsCustomVocabDataset_{self.max_len}_{split}_light.pkl")
        else:
            tgt_file = os.path.join(self.datadir,
                                    f"ExpIndJobsCustomVocabToyDataset_{self.max_len}_{split}_{subsample}.pkl")
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