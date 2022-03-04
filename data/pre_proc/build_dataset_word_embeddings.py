import argparse
import ipdb
import yaml
import os
from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from transformers import CamembertModel, CamembertTokenizer
from collections import Counter


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        # word_dict_sorted = get_most_common_words(args)
        # vocab_dict = get_vocab_dict(word_dict_sorted)
        # with open(os.path.join(CFG["gpudatadir"], 'vocab_dict.pkl'), 'wb') as f:
        #     pkl.dump(vocab_dict, f)
        with open(os.path.join(CFG["gpudatadir"], 'vocab_dict.pkl'), 'rb') as f:
            vocab_dict = pkl.load(f)
        print("Vocabulary length:" + str(len(vocab_dict)))
        if args.gpu == "True":
            word_embedder = CamembertModel.from_pretrained('camembert-base').cuda()
        else:
            word_embedder = CamembertModel.from_pretrained('camembert-base')
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        embs = build_embeddings(vocab_dict, word_embedder, tokenizer)
        ipdb.set_trace()
        emb_file = os.path.join(CFG["gpudatadir"], 'word_embs_dynamics.t')
        torch.save(torch.from_numpy(embs), emb_file)


def get_most_common_words(args):
    word_counter = Counter()
    ind_file = "ind_class_dict.pkl"
    with open(os.path.join(CFG["gpudatadir"], ind_file), 'rb') as f_name:
        industry_dict = pkl.load(f_name)
    rev_ind_dict = {v: k for k, v in industry_dict.items()}
    # get relevant data files
    for split in ["VALID", "TRAIN"]:
        ppl_file = "profiles_jobs_fr_3_" + str(args.max_len) + "_" + split + ".pkl"
        with open(os.path.join(CFG["gpudatadir"], ppl_file), 'rb') as f_name:
            ppl_reps = pkl.load(f_name)
            for person in tqdm(ppl_reps, desc="parsing tuples for split " + split + "..."):
                if person[1] != "" and person[1] in rev_ind_dict.keys():
                    for num, job in enumerate(person[-1]):
                        for word in job["job"]:
                            word_counter[word.lower()] += 1
    print("total_words = " + str(len(word_counter)))
    non_unique_words = {}
    for k, v in word_counter.items():
        if v >= args.min_count:
            non_unique_words[k] = word_counter[k]
    print("total words that appear more than " + str(args.min_count) + " times = " + str(len(non_unique_words)))
    return sorted(non_unique_words.items(), key=lambda k: -k[1])

def build_embeddings(vocab, word_embedder, tokenizer):
    embeddings = np.zeros((len(vocab), 768))
    for k, word in tqdm(vocab.items(), desc="computing word embeddings..."):
        if args.gpu == "True":
            inputs = tokenizer(word, return_tensors='pt')["input_ids"].cuda()
        else:
            inputs = tokenizer(word, return_tensors='pt')["input_ids"]
        #ipdb.set_trace()
        if len(inputs[0]) > 2:
            tmp = word_embedder(inputs[0][1:-1].unsqueeze(0), return_dict=True)["last_hidden_state"][0, 0, :]
            embeddings[k, :] = tmp.detach().cpu().numpy()
        #embeddings = embeddings.cpu()
    return embeddings


def get_vocab_dict(word_dict_sorted):
    voc, special_indices, special_tokens = insert_special_tokens()
    print("Special tokens inserted.")
    index = 2 # 0 and 1 are taken already
    for word in tqdm(word_dict_sorted, desc="Building vocab..."):
        if index not in special_indices and word[0] not in special_tokens:
            voc[index] = word[0]
        index += 1
    return voc

def insert_special_tokens():
    ## camembert(s base special tokens
    # 0 <s>NOTUSED
    # 1 <pad>
    # 2 </s>NOTUSED
    # 3 <unk>
    # 4
    # 5 <s>
    # 6 </s>
    # 32004 <mask>
    special_indices = [1, 3, 5, 6, 0]
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    voc = {}
    for ind, tok in zip(special_indices, special_tokens):
        voc[ind] = tok
    return voc, special_indices, special_tokens



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", default="True")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_voc_len", type=int, default=32000)
    parser.add_argument("--min_count", type=int, default=35)
    parser.add_argument("--gpu", type=str, default="True")
    args = parser.parse_args()
    main(args)
