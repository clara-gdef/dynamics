import ipdb
import torch
from os import system
import subprocess

import os
import glob
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def collate_for_delta_jobs(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    return ids, jobs, labels


def collate_for_rewriter(batch):
    b_size = len(batch)
    # padding index in pre trained camembert is "1"
    indices = torch.ones(b_size, 256).type(torch.LongTensor)
    mask = torch.zeros(b_size, 256).type(torch.LongTensor)
    exp_index = torch.zeros(b_size).type(torch.LongTensor)
    ind_index = torch.zeros(b_size).type(torch.LongTensor)
    for num, sample in enumerate(batch):
        indices[num, :len(sample[0][0])] = torch.LongTensor(sample[0][0])
        mask[num, :len(sample[1][0])] = torch.LongTensor(sample[1][0])
        exp_index[num] = sample[3]
        ind_index[num] = sample[5]
    return indices, mask, torch.LongTensor(exp_index), torch.LongTensor(ind_index)


def collate_for_rewriter_light(batch):
    indices = [i[0] for i in batch]
    mask = [i[1] for i in batch]
    exp_index = [i[2] for i in batch]
    ind_index = [i[3] for i in batch]
    return torch.stack(indices).type(torch.LongTensor).squeeze(1), \
           torch.stack(mask).type(torch.LongTensor).squeeze(1), \
           torch.LongTensor(exp_index), \
           torch.LongTensor(ind_index)


def collate_for_causal_lm(batch):
    words = [[i[0]] for i in batch]
    return words


def collate_for_bertgen(batch):
    words = [i[0] for i in batch]
    return words


def collate_for_delta_bert(batch):
    words = [i[0] for i in batch]
    ind = [i[1] for i in batch]
    delta = [i[2] for i in batch]
    return words, torch.LongTensor(ind), torch.LongTensor(delta)


def collate_for_att_classif_exp(batch):
    indices = [i[0] for i in batch]
    exp_index = [i[2] for i in batch]
    return torch.stack(indices).type(torch.LongTensor).squeeze(1), \
           torch.LongTensor(exp_index)


def collate_for_att_classif_ind(batch):
    indices = [i[0] for i in batch]
    ind_index = [i[3] for i in batch]
    return torch.stack(indices).type(torch.LongTensor).squeeze(1), \
           torch.LongTensor(ind_index)


def compute_bleu_score(pred, ref):
    cmd_line = './multi-bleu.perl ' + ref + ' < ' + pred + ''
    return subprocess.check_output(cmd_line, shell=True)


def get_latest_model(modeldir, model_name):
    model_path = os.path.join(modeldir, model_name)
    model_files = glob.glob(os.path.join(model_path, "*.ckpt"))
    latest_file = max(model_files, key=os.path.getctime)
    return latest_file


def is_number(tensor):
    if torch.isnan(tensor).any():
        print("tensor has nan")
    if torch.isinf(tensor).any():
        print("tensor is infinite")


def embed_and_avg_attributes(exp_indices, ind_indices, out_shape, embedder_exp, embedder_ind):
    if exp_indices is not None:
        sample_len = len(exp_indices)
    elif ind_indices is not None:
        sample_len = len(ind_indices)
    else:
        print(f"attribute selection failed")
        ipdb.set_trace()
    embedd_attributes = torch.zeros(sample_len, 1, out_shape).type(torch.cuda.FloatTensor)
    for sample_num in range(sample_len):
        if exp_indices is not None:
            tmp_xp = embedder_exp(exp_indices[sample_num])
        if ind_indices is not None:
            tmp_ind = embedder_ind(ind_indices[sample_num])
        if ind_indices is not None and exp_indices is not None:
            tmp_att = torch.cat((tmp_xp.unsqueeze(0), tmp_ind.unsqueeze(0)))
            embedd_attributes[sample_num, :, :] = torch.mean(tmp_att, dim=0).unsqueeze(0)
        else:
            if exp_indices is not None:
                embedd_attributes[sample_num, :, :] = tmp_xp
            elif ind_indices is not None:
                embedd_attributes[sample_num, :, :] = tmp_ind
            else:
                print(f"attribute selection failed")
                ipdb.set_trace()
    return embedd_attributes


def masked_softmax(logits, mask, seq_len):
    sample_len = logits.shape[0]
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    mask = mask.type(dtype=logits.dtype)
    if seq_len > 1:# we compare the encoder output against the whole sequence
        weights = torch.exp(logits) * mask.unsqueeze(-1).expand(sample_len, seq_len, seq_len-1)
    # we compare the encoder output against only 1 token
    else:
        weights = torch.exp(logits) * mask.unsqueeze(-1)
    denominator = 1e-7 + torch.sum(weights, dim=1, keepdim=True)
    return weights / denominator


def sample_y_tilde(exp_index, ind_index, num_exp, num_ind, debug):
    if debug != "True":
        exp_tilde_index, ind_tilde_index = None, None
        if exp_index is not None:
            sample_len = len(exp_index)
            exp_tilde_index = torch.zeros(sample_len).type(torch.LongTensor)
        if ind_index is not None:
            sample_len = len(ind_index)
            ind_tilde_index = torch.zeros(sample_len).type(torch.LongTensor)
        for num in range(sample_len):
            if exp_index is not None:
                rnd_exp = exp_index[num]
                while rnd_exp == exp_index[num]:
                    rnd_exp = random.randrange(0, num_exp)
                exp_tilde_index[num] = rnd_exp
            if ind_index is not None:
                rnd_ind = ind_index[num]
                while rnd_ind == ind_index[num]:
                    rnd_ind = random.randrange(0, num_ind)
                ind_tilde_index[num] = rnd_ind
        final_exp_tilde = exp_tilde_index.type_as(exp_index) if exp_index is not None else None
        final_ind_tilde = ind_tilde_index.type_as(ind_index) if ind_index is not None else None
    else:
        final_exp_tilde = exp_index
        final_ind_tilde = ind_index
    return final_exp_tilde, final_ind_tilde


def handle_fb_pred(pred):
    return pred[0][0].split("__label__")[-1]


def handle_fb_preds(pred):
    return [i.split("__label__")[-1] for i in pred[0]]


def handle_svm_pred(pred, k):
    return np.argsort(pred, axis=-1)[:, :k]


def get_metrics_at_k(predictions, labels, num_classes, handle):
    out_predictions = []
    for index, pred in enumerate(predictions):
        if labels[index] in pred:
            if (type(labels[index]) == torch.Tensor) or (type(labels[index]) == np.ndarray):
                out_predictions.append(labels[index].item())
            else:
                out_predictions.append(labels[index])
        else:
            if (type(pred[0]) == torch.Tensor) or (type(pred[0]) == np.ndarray):
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle)


def get_metrics(preds, labels, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        # "precision_" + handle: precision_score(labels, preds, average='weighted',
        #                                        labels=num_c, zero_division=0) * 100,
        # "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def decode_tokenized_words(tokenized_words, vocab_dict):
    indices_list = tokenized_words.tolist()
    words = []
    for i in indices_list:
        if i != 6: # eos token
            words.append(vocab_dict[i])
        else:
            break
    return " ".join(words)


def indices_to_words(tensor, vocab):
    if len(tensor.shape) > 1:
        sents = []
        for sample in tensor:
            words = []
            eos_flag = False
            for i in sample:
                if not eos_flag:
                    if i.item() != 6: # eos_token
                        if i.item() != 1:  # pad token
                            words.append(vocab[i.item()])
                    else:
                        eos_flag = True
            sents.append(' '.join(words))
        return sents
    else:
        words = []
        for i in tensor:
            if i.item() != 1: # pad token
                words.append(vocab[i.item()])
        return ' '.join(words)


def get_attribute(att_type, dataset, item):
    if att_type == "ind":
        industry_tmp = dataset.ind_dict[item[-1]]
        if len(industry_tmp.split(" ")) > 1:
            att = "_".join(industry_tmp.split(" "))
        else:
            att = industry_tmp
    elif att_type == "exp":
        att = dataset.exp_dict[item[-2]]
    else:
        raise Exception("Wrong att type specified! Can be \"ind\" or \"exp\" got: " + att_type)
    return att


# https://stackoverflow.com/questions/64356953/batch-wise-beam-search-in-pytorch
def beam_search_decoder(post, k):
    """Beam Search Decoder
    Parameters:
        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.
    Outputs:
        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.
    Shape:
        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).
    Examples:
        post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        indices, log_prob = beam_search_decoder(post, 3)

    """
    batch_size, seq_length, vocab_size = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1) % vocab_size], dim=-1)
    if indices.max() >= vocab_size:# voc_size
        ipdb.set_trace()
    return indices, log_prob


def prettify_bleu_score(bbleu):
    bleu_dict = {}
    str_bleu = bbleu.decode()
    bleu_dict["all"] = str_bleu
    bleu_dict["bleu"] = float(str_bleu.split(" ")[2][:-1])
    bleu_dict["bleu1"] = float(str_bleu.split("/")[0].split(',')[-1])
    bleu_dict["bleu2"] = float(str_bleu.split("/")[1])
    bleu_dict["bleu3"] = float(str_bleu.split("/")[2])
    bleu_dict["bleu4"] = float(str_bleu.split("/")[3].split(" ")[0])
    return bleu_dict


def plot_grad_flow(named_parameters, desc):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for name, param in named_parameters:
        if (param.requires_grad) and ("bias" not in name):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean())
            max_grads.append(param.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), torch.stack(max_grads).cpu().numpy(), alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), torch.stack(ave_grads).cpu().numpy(), alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f"img/plot_grad_flow_{desc}.png")


def print_cm(preds, labels, desc, label_dict, att_type):
    all_lab_indices = sorted([i for i in label_dict.keys()])
    if att_type == "ind":
        all_lab_names = sorted([i for i in label_dict.values()])
    else:
        all_lab_names = all_lab_indices
    ipdb.set_trace()
    cm = confusion_matrix(labels, preds, labels=all_lab_indices, sample_weight=None, normalize="all")
    if len(all_lab_indices) > 10:
        size = 100
    else:
        size = 10
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    xaxis = np.arange(len(all_lab_indices))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(all_lab_names, rotation="vertical")
    ax.set_yticklabels(all_lab_names)
    fig.savefig(f"/local/gainondefor/work/data/{desc}_{att_type}.svg")
    fig.show()

