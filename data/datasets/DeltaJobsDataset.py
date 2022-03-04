import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class DeltaJobsDataset(Dataset):
    def __init__(self, data_dir, ppl_file, load, granularity, split):
        self.datadir = data_dir
        self.split = split
        self.granularity = granularity
        self.mean, self.std = self.get_params(ppl_file)
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset()
        else:
            with open(os.path.join(data_dir, ppl_file), 'rb') as f_name:
                ppl_reps = pkl.load(f_name)
            # self.rep_dim = 300
            self.tuples, self.user_lookup = self.build_ppl_tuples(ppl_reps, split)
            self.save_dataset()

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx][f"delta_{self.granularity}"],  self.tuples[idx][f"delta_{self.granularity}_std"]

    def save_dataset(self):
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = os.path.join(self.datadir, f"DeltaJobsDataset_{self.split}.pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self):
        tgt_file = os.path.join(self.datadir, f"DeltaJobsDataset_{self.split}.pkl")
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)
        for key in tqdm(ds_dict, desc="Loading attributes from save..."):
            vars(self)[key] = ds_dict[key]
        print("Dataset load from : " + tgt_file)

    def get_params(self, ppl_file):
        with open(os.path.join(self.datadir, ppl_file), 'rb') as f_name:
            ppl_reps = pkl.load(f_name)
        if self.split == "TRAIN":
            tmp = []
            for person in tqdm(ppl_reps, desc=f"Getting mean and std for Delta Job Dataset for split {self.split} ..."):
                for num, job in enumerate(person[-1]):
                    if num != 0:
                        tmp.append(job[f"delta_{self.granularity}"])
            mean = np.mean(np.stack(tmp))
            std = np.std(np.stack(tmp))
            with open(os.path.join(self.datadir, f"DeltaJobsDataset_{self.granularity}_params.pkl"), 'wb') as f:
                pkl.dump({'mean': mean, "std": std}, f)
        else:
            with open(os.path.join(self.datadir, f"DeltaJobsDataset_{self.granularity}_params.pkl"), 'rb') as f:
                params = pkl.load(f)
            mean = params["mean"]
            std = params["std"]
        return mean, std

    def build_ppl_tuples(self, ppl_reps, split):
        # length of the profile that accounts for 90% of the training dataset
        # max_prof_len = 9
        user_lookup = {}
        counter = 0
        tuples = []
        for person in tqdm(ppl_reps, desc="Building Delta Job Dataset for split " + split + " ..."):
            id_person = person[0]
            sorted_jobs = sorted(person[-1], key=lambda k: k['from'])
            tmp = {}
            for num, job in enumerate(sorted_jobs):
                tmp[f"delta_{self.granularity}"] = job[f"delta_{self.granularity}"]
                tmp[f"delta_{self.granularity}_std"] = (job[f"delta_{self.granularity}"] - self.mean) / self.std
                tuples.append(tmp)
            user_lookup[id_person] = [counter, counter + len(sorted_jobs) - 1]
            counter = counter + len(sorted_jobs)
        return tuples, user_lookup

    def get_tgt_file(self, split):
        if self.granularity == "days":
            tgt_file = os.path.join(self.datadir, f"DeltaJobsDataset_{split}_.pkl")
        elif self.granularity == "years":
            tgt_file = os.path.join(self.datadir,
                                    f"DeltaJobsDataset_years_{split}.pkl")
        return tgt_file