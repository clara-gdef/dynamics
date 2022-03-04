import ipdb
import csv
from utils import decode_tokenized_words
from tqdm import tqdm


def print_tilde_to_csv(att_type, real_jobs, real_exp, real_ind, generated_job, exp_tilde, ind_tilde, exp_pred, ind_pred, desc, ind_dict, exp_dict):
    tgt_file = f"csv/{desc}.csv"
    if att_type == "both":
        to_csv_for_both_att(tgt_file, real_jobs, real_exp, real_ind, generated_job, exp_tilde, ind_tilde, exp_pred, ind_pred, ind_dict, exp_dict)
    elif att_type == "exp":
        to_csv_for_exp(tgt_file, real_jobs, real_exp, generated_job, exp_tilde, exp_pred, exp_dict)
    elif att_type == "ind":
        to_csv_for_ind(tgt_file, real_jobs, real_ind, generated_job, ind_tilde, exp_pred, ind_pred, ind_dict)
    else:
        raise Exception("wrong att_type, can be either \"exp\", \"ind\" or \"both\", " + str(att_type) + " was given.")
    return tgt_file


def to_csv_for_exp(tgt_file, real_jobs, real_exp, generated_job, exp_tilde, exp_pred,  exp_dict):
    headers = ["initial job", "initial exp", "generated job", "exp tilde", "exp predicted"]
    assert len(real_jobs) == len(generated_job)
    assert len(real_exp) == len(exp_tilde)
    assert len(real_exp) == len(exp_pred)
    with open(tgt_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for job, exp, gen_job, exp_t, exp_p in tqdm(zip(real_jobs, real_exp, generated_job, exp_tilde, exp_pred)):
            row = tuple_to_row_exp(job, exp, gen_job, exp_t, exp_p, exp_dict)
            writer.writerow(row)


def to_csv_for_ind(tgt_file, real_jobs, real_ind, generated_job, ind_tilde, ind_pred, ind_dict):
    headers = ["initial job",  "initial ind", "generated job", "ind tilde", "ind predicted"]
    assert len(real_jobs) == len(generated_job)
    assert len(real_ind) == len(ind_tilde)
    assert len(real_ind) == len(ind_pred)
    with open(tgt_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for job, ind, gen_job, ind_t, ind_p in tqdm(zip(real_jobs, real_ind, generated_job, ind_tilde, ind_pred)):
            row = tuple_to_row_both_att(job, ind, gen_job, ind_t, ind_p, ind_dict)
            writer.writerow(row)


def to_csv_for_both_att(tgt_file, real_jobs, real_exp, real_ind, generated_job, exp_tilde, ind_tilde, exp_pred, ind_pred, ind_dict, exp_dict):
    headers = ["initial job", "initial exp", "initial ind", "generated job", "exp tilde", "ind tilde", "exp predicted", "ind predicted"]
    assert len(real_jobs) == len(generated_job)
    assert len(real_exp) == len(exp_tilde)
    assert len(real_exp) == len(exp_pred)
    assert len(real_ind) == len(ind_tilde)
    assert len(real_ind) == len(ind_pred)
    with open(tgt_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for job, exp, ind, gen_job, exp_t, ind_t, exp_p, ind_p in tqdm(zip(real_jobs, real_exp, real_ind, generated_job, exp_tilde, ind_tilde, exp_pred, ind_pred)):
            row = tuple_to_row_both_att(job, exp, ind, gen_job, exp_t, ind_t, exp_p, ind_p, ind_dict, exp_dict)
            writer.writerow(row)


def tuple_to_row_both_att(job, exp, ind, gen_job, exp_t, ind_t, exp_p, ind_p, ind_dict, exp_dict):
    row = []
    row.append(job.strip())
    row.append((exp, exp_dict[exp]))
    row.append((ind, ind_dict[ind]))
    row.append(gen_job)
    row.append((exp_t.item(), exp_dict[exp_t.item()]))
    row.append((ind_t.item(), ind_dict[ind_t.item()]))
    row.append((exp_p[0], exp_dict[exp_p[0]]))
    row.append((ind_p[0], ind_dict[ind_p[0]]))
    return row


def tuple_to_row_exp(job, exp, gen_job, exp_t, exp_p, exp_dict):
    row = []
    row.append(job.strip())
    row.append((exp, exp_dict[exp]))
    row.append(gen_job)
    row.append((exp_t.item(), exp_dict[exp_t.item()]))
    row.append((exp_p[0], exp_dict[exp_p[0]]))
    return row


def tuple_to_row_ind(job, ind, gen_job, ind_t, ind_p, ind_dict):
    row = []
    row.append(job.strip())
    row.append((ind, ind_dict[ind]))
    row.append(gen_job)
    row.append((ind_t.item(), ind_dict[ind_t.item()]))
    row.append((ind_p[0], ind_dict[ind_p[0]]))
    return row
# def tuple_to_row(job, exp, ind, gen_job, exp_t, ind_t, exp_p, ind_p, vocab, ind_dict, exp_dict):
#     row = []
#     row.append(decode_tokenized_words(job[0], vocab).split("<s>")[-1].strip())
#     row.append((exp.item(), exp_dict[exp.item()]))
#     row.append((ind.item(), ind_dict[ind.item()]))
#     row.append(decode_tokenized_words(gen_job[0], vocab))
#     row.append((exp_t.item(), exp_dict[exp_t.item()]))
#     row.append((ind_t.item(), ind_dict[ind_t.item()]))
#     row.append((exp_p[0].item(), exp_dict[exp_p[0].item()]))
#     row.append((ind_p[0].item(), ind_dict[ind_p[0].item()]))
#     return row


def print_steps_to_csv(person_list, filename):
    tgt_file = f"csv/{filename}.csv"
    headers = ["id", "num steps", "exp level as labelled", "exp level as detected", "job"]
    with open(tgt_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        current_id = None
        for person in tqdm(person_list):
            id_person, num_steps, exp_level, pred_exp, job = person[0], person[1], person[2], person[3], person[4]
            if current_id != id_person:
                writer.writerow([id_person, "=========", "========", "========", person[-1]])
                current_id = id_person
            row = [id_person, num_steps, exp_level, pred_exp, job]
            writer.writerow(row)
    return tgt_file