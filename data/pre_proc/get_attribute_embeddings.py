import os
import ipdb
import torch
import yaml
import pickle as pkl
from data.datasets import ExpIndJobsDataset
from transformers import RobertaModel

def main():
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        arguments = {'data_dir': CFG["gpudatadir"],
                         "subsample": 0,
                         "load": "True",
                         "split": "TRAIN"}
        dataset = ExpIndJobsDataset(**arguments)
        experiences = sorted([v for v in dataset.exp_dict.values()])
        industries = sorted([v for v in dataset.ind_dict.values()])

        embedder = RobertaModel.from_pretrained('roberta-base')

        emb_dim = 768
        exp_embs, exp_indices = get_embeddings_for_attributes(dataset, experiences, embedder, emb_dim)
        ind_embs, ind_indices = get_embeddings_for_attributes(dataset, industries, embedder, emb_dim)
        emb_attributes = torch.cat([exp_embs, ind_embs], dim=0)

        lookup = {}
        counter = 0
        for tmp in zip(experiences, exp_indices):
            lookup[counter] = {"name": tmp[0], 'tokens': tmp[1]}
            counter += 1
        for name, toks in zip(industries, ind_indices):
            lookup[counter] = {"name": name, 'tokens': toks}
            counter += 1
        ipdb.set_trace()

        with open(os.path.join(CFG["gpudatadir"], "embs_exp_ind.t"), "wb") as f:
            torch.save(emb_attributes, f)
        with open(os.path.join(CFG["gpudatadir"], "lookup_att_emb.pkl"), "wb") as f:
            pkl.dump(lookup, f)

def get_embeddings_for_attributes(dataset, attributes, embedder, emb_dim):
    embs = torch.zeros(len(attributes), emb_dim)
    max_token_len = max([len(i) for i in attributes]) + 2
    indices = torch.zeros(len(attributes), max_token_len).type(torch.LongTensor)
    for num_att, att in enumerate(attributes):
        ind = dataset.tokenize_attribute(att)
        indices[num_att, :len(ind[0])] = ind[0].type(torch.LongTensor)
        embs[num_att, :] = torch.mean(embedder(ind, return_dict=True)["last_hidden_state"], dim=1)
    return embs, indices



if __name__ == "__main__":
    main()
