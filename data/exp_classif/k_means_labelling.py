import os
import ipdb
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import CamembertTokenizer, CamembertModel
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import Counter
import torch
from data.datasets import StringDataset
from torch.utils.data import DataLoader
from utils.models import decode_tokenized_words


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
    splits = ["TRAIN", "VALID"]
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    model = model.cuda()
    exp_dict = get_exp_dict(args)
    if args.exp_type == "kmeans":
        embedded_class_names = embed_class_names(exp_dict, tokenizer, model)
        kmeans = KMeans(n_clusters=len(exp_dict), init=embedded_class_names)
    elif args.exp_type == "kmeansrdm":
        kmeans = KMeans(n_clusters=len(exp_dict))
    else:
        raise Exception("Wrong exp_type specified, can be kmeans or kmeansrdm")
    datasets = load_datasets(args, splits)
    all_data = []
    for num, split in enumerate(splits):
        dataloader = DataLoader(datasets[num], batch_size=args.b_size, collate_fn=collate_fn,
                                num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        processed_data = process_jobs_for_kmeans(dataloader, tokenizer, model, split)
        all_data.extend(processed_data)
    embedded_jobs_numpy = batched_outputs_to_array(all_data, len(datasets[0]) + len(datasets[1]))
    kmeans.fit(embedded_jobs_numpy)
    labels_train = kmeans.labels_[:len(datasets[0])]
    labels_valid = kmeans.labels_[len(datasets[0]):]

    suffix = f"{args.exp_type}_{args.exp_levels}"
    if args.custom_centers == "True":
        suffix += "custom_centers"

    datasets[0].save_with_new_labels(labels_train, suffix)
    datasets[1].save_with_new_labels(labels_valid, suffix)

    dataset_test = load_datasets(args, ["TEST"])[0]
    dataloader_test = DataLoader(dataset_test, batch_size=args.b_size, collate_fn=collate_fn,
                                 num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)
    processed_data_test = process_jobs_for_kmeans(dataloader_test, tokenizer, model, "TEST")
    embedded_jobs_numpy_test = batched_outputs_to_array(processed_data_test, len(dataset_test))
    labels_test = kmeans.predict(embedded_jobs_numpy_test)
    dataset_test.save_with_new_labels(labels_test, suffix)
    # get_visu(splits, exp_dict, embedded_jobs_numpy,  kmeans.labels_)


def batched_outputs_to_array(batched_outputs, expected_len):
    embedded_jobs_numpy = np.zeros((expected_len, 768))
    num = 0
    for block in tqdm(batched_outputs):
        for data in block:
            embedded_jobs_numpy[num, :] = data.reshape(1, 768)
            num += 1
    assert num == expected_len
    return embedded_jobs_numpy


def get_visu(splits, exp_dict, embedded_jobs_numpy, labels):
    num_points = args.subsample * len(splits)
    if num_points <= 5000:
        compute_and_plot_pca(embedded_jobs_numpy, labels, kmeans.cluster_centers_, exp_dict,
                             f"init_not_random_sub{num_points}")
    kmeans_rdn = KMeans(n_clusters=len(exp_dict))
    kmeans_rdn.fit(embedded_jobs_numpy)
    labels_rdn = kmeans_rdn.labels_
    if num_points <= 5000:
        compute_and_plot_pca(embedded_jobs_numpy, labels_rdn, kmeans_rdn.cluster_centers_, exp_dict,
                             f"random_sub{num_points}")
    plot_hist_of_labels(labels, labels_rdn, len(exp_dict), args.subsample)


def plot_hist_of_labels(labels, labels_rdn, num_classes, subsample):
    fig, ax = plt.subplots(2, 1)
    y_init = Counter()
    for val in labels:
        y_init[val] += 1
    height_init = [i[1] for i in sorted(y_init.items())]
    y_rdn = Counter()
    for val in labels_rdn:
        y_rdn[val] += 1
    height_rdn = [i[1] for i in sorted(y_rdn.items())]
    ax[0].bar(np.arange(num_classes), height_init)
    ax[0].set_title("initialized")
    ax[1].bar(np.arange(num_classes), height_rdn)
    ax[1].set_title("random")
    plt.savefig(f"img/histograms_of_labels_sub{subsample}.svg", format='svg')


def compute_and_plot_pca(embedded_jobs_numpy, labels, centers, exp_dict, desc):
    pca = PCA(n_components=2)
    print(f"Fitting PCA...")
    data = pca.fit_transform(embedded_jobs_numpy)
    print(f"PCA fitted.")

    color_map = {0: "blue",
                 1: "green",
                 2: "red",
                 3: "purple",
                 4: "orange"}
    colors_to_print = []
    for i in labels:
        colors_to_print.append(color_map[i])

    fig, ax = plt.subplots()

    for j, i in enumerate(centers):
        projected_center = pca.transform(i.reshape(1, -1))[0]
        ax.scatter(projected_center[0], projected_center[1], marker="x", c=color_map[j])
        plt.text(projected_center[0] + .01, projected_center[1] + .01, exp_dict[j])

    x, y = [], []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    for e in tqdm(range(len(data)), desc="Scattering projected data..."):
        ax.scatter(x[e], y[e], marker="o", c=colors_to_print[e])
    plt.title(f"PCA {desc}")
    print("Saving figure...")
    plt.savefig(f"img/PCA_kmeans_exp_{desc}.svg", format='svg')
    print("Figure saved.")
    plt.show()


def embed_class_names(exp_dict, tokenizer, model):
    tmp = []
    for class_name in exp_dict.values():
        inputs = tokenizer(class_name, return_tensors="pt")["input_ids"].cuda()
        outputs = model(inputs)
        tmp.append(outputs.pooler_output)
    return torch.stack(tmp).squeeze(1).cpu().detach().numpy()


def process_jobs_for_kmeans(dataloader, tokenizer, model, split):
    jobs_embedded = []
    for tup in tqdm(dataloader, desc=f"Converting jobs to camembert embedded tuples for split {split}..."):
        inputs = tokenizer(tup, return_tensors="pt", padding=True, truncation=True)["input_ids"].cuda()
        outputs = model(inputs)
        jobs_embedded.append(outputs.pooler_output.cpu().detach().numpy())
    return jobs_embedded


def collate_fn(batch):
    indices = [i[0] for i in batch]
    return indices


def load_datasets(hparams, splits):
    datasets = []
    arguments = {'data_dir': CFG["gpudatadir"],
                 "load": "True",
                 "subsample": hparams.subsample,
                 "max_len": 10,
                 "exp_levels": args.exp_levels,
                 "exp_type": args.exp_type,
                 "is_toy": hparams.toy_dataset}
    for split in splits:
        datasets.append(StringDataset(**arguments, split=split))
    return datasets


def get_exp_dict(args):
    if args.custom_centers == "True":
        if args.exp_levels == 3:
            exp_dict = {0: "stage",
                        1: "chef",
                        2: "CEO"}
        elif args.exp_levels == 5:
            exp_dict = {0: "stage",
                        1: "assistante",
                        2: "chef",
                        3: "president",
                        4: "CEO"}
        else:
            raise Exception("Wrong exp_levels specified, can be 3 or 5")
    else:
        with open(os.path.join(CFG["gpudatadir"], f"exp_dict_fr_{args.exp_levels}.pkl"), 'rb') as f_name:
            exp_dict = pkl.load(f_name)
    return exp_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running params
    parser.add_argument("--load_dataset", default="False")
    parser.add_argument("--toy_dataset", default="False")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--b_size", type=int, default=120)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--exp_levels", type=int, default=5)
    parser.add_argument("--exp_type", type=str, default="kmeans")  # "kmeans" or "kmeansrdm"
    parser.add_argument("--custom_centers", type=str, default="False")
    # global hyper params
    args = parser.parse_args()
    init(args)
