import ipdb
import argparse
import yaml
import os
import pickle as pkl
from tqdm import tqdm
import random
from collections import Counter
from data.datasets import StringDataset, StringIndSubDataset
import itertools


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.DEBUG == "True":
        with ipdb.launch_ipdb_on_exception():
            return main(args)
    else:
        return main(args)


def main(args):
    with open(os.path.join(CFG["gpudatadir"], "20_industry_dict.pkl"), 'rb') as f:
        sub_ind_dict = pkl.load(f)
    datasets = []
    splits = ["TRAIN", "VALID", "TEST"]
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "exp_type": args.exp_type,
                 "is_toy": "False"}
    for split in splits:
        datasets.append(StringDataset(**arguments, split=split))
    all_tuples = datasets[0].tuples.copy()
    all_tuples.extend(datasets[1].tuples.copy())
    all_tuples.extend(datasets[2].tuples.copy())
    print(len(all_tuples))

    with open(os.path.join(CFG["gpudatadir"], "ind_map_to_subsampled.pkl"), 'rb') as f:
        ind_map_to_subsampled = pkl.load(f)
    rev_mapping = {v: k for k, v in ind_map_to_subsampled.items()}
    all_sub_tuples, new_lookup = get_subsampled_tuples(all_tuples, rev_mapping, datasets)
    max_id = max([i[1] for i in new_lookup.values()])
    train, valid, test = [], [], []
    train_lu, valid_lu, test_lu = {}, {}, {}
    train_count, valid_count, test_count = 0, 0, 0
    for person in tqdm(new_lookup.keys(), desc="Splitting tuples into train, valid and test..."):
        tmp = random.random()
        start = new_lookup[person][0]
        end = new_lookup[person][1]
        num_jobs = end - start + 1
        if tmp > .2:
            # train set
            train.extend(all_sub_tuples[start:end+1].copy())
            train_lu[person] = [train_count, train_count + num_jobs - 1]
            train_count += num_jobs
        elif tmp > .1:
            # valid set
            valid.extend(all_sub_tuples[start:end+1].copy())
            valid_lu[person] = [valid_count, valid_count + num_jobs - 1]
            valid_count += num_jobs
        else:
            # test set
            test.extend(all_sub_tuples[start:end+1].copy())
            test_lu[person] = [test_count, test_count + num_jobs - 1]
            test_count += num_jobs
    total_len = len(train) + len(valid) + len(test)
    print(f"percentage of tuples in train: {100*len(train)/total_len} %")
    print(f"Total tuples in train: {len(train)}")
    print(f"percentage of tuples in valid: {100*len(valid)/total_len} %")
    print(f"Total tuples in valid: {len(valid)}")
    print(f"percentage of tuples in test: {100*len(test)/total_len} %")
    print(f"Total tuples in test: {len(test)}")
    # should be 1,178,823
    print(f"total number of jobs: {total_len}")
    ipdb.set_trace()

    dump_list_in_dataset(train, train_lu, "TRAIN", rev_mapping)
    dump_list_in_dataset(valid, valid_lu, "VALID", rev_mapping)
    dump_list_in_dataset(test, test_lu, "TEST", rev_mapping)

    ipdb.set_trace()


def dump_list_in_dataset(tuple_list, lookup, split, rev_mapping):
    for blob in tuple_list:
        assert blob["ind_index"] in rev_mapping.keys()

    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "False",
                 "subsample": -1,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "tuple_list": tuple_list.copy(),
                 'lookup': lookup.copy(),
                 "already_subbed": "False",
                 "exp_type": args.exp_type,
                 "is_toy": "False",
                 "split": split}
    tmp = StringIndSubDataset(**arguments)
    cnt = Counter()
    for blob in tmp.tuples:
        cnt[blob["exp_index"]] += 100 / len(tmp.tuples)
        assert blob["ind_index"] < 20
    print(cnt)
    if split == "TEST":
        ipdb.set_trace()


def get_subsampled_tuples(all_tuples, rev_mapping, datasets):
    new_lookup = {}
    sub_tuples_train, new_lookup = get_subsampled_tuples_per_split(datasets[0], new_lookup, all_tuples, rev_mapping, 0)
    print(len(sub_tuples_train))
    sub_tuples_valid, new_lookup = get_subsampled_tuples_per_split(datasets[1], new_lookup, all_tuples, rev_mapping, len(sub_tuples_train))
    print(len(sub_tuples_valid))
    sub_tuples_test, new_lookup = get_subsampled_tuples_per_split(datasets[2], new_lookup, all_tuples, rev_mapping, len(sub_tuples_valid) + len(sub_tuples_train))
    print(len(sub_tuples_test))
    all_sub_tuples = sub_tuples_train.copy()
    all_sub_tuples.extend(sub_tuples_valid.copy())
    all_sub_tuples.extend(sub_tuples_test.copy())
    return all_sub_tuples, new_lookup


def get_subsampled_tuples_per_split(datasets, new_lookup, all_tuples, rev_mapping, job_offset):
    subsampled_tuples = []
    count_before, count_after = job_offset, job_offset-1
    for person_id in tqdm(datasets.user_lookup.keys(), desc="subsampling tuples according to industry..."):
        current_person = datasets.user_lookup[person_id]
        end = current_person[1] + job_offset
        start = current_person[0] + job_offset
        assert end > start
        num_jobs = end - start + 1
        flag = False
        if all_tuples[start]["ind_index"] in rev_mapping.keys():
            for i in range(start, start + num_jobs):
                if all_tuples[i]["ind_index"] in rev_mapping.keys():
                    subsampled_tuples.append(all_tuples[i])
                    count_after += 1
                    flag = True
        if flag:
            new_lookup[person_id] = [count_before, count_after]
            assert count_after - job_offset == len(subsampled_tuples) - 1
            count_before = count_after+1
    max_id = max([i[1] for i in new_lookup.values()])
    assert len(subsampled_tuples) + job_offset - 1 == max_id
    return subsampled_tuples, new_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str, default="True")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--exp_type", type=str, default="uniform")
    parser.add_argument("--exp_levels", type=int, default=3)
    args = parser.parse_args()
    init(args)
