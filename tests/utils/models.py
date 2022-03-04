import torch
import ipdb
from utils.models import *

def main():
    with ipdb.launch_ipdb_on_exception():
        print("Testing beam seach")
        test_beam_search()

def test_beam_search():
    beam_size = 3
    voc_size = 5
    max_len = 3
    predictions = torch.zeros(1, max_len, voc_size)
    counter = 0
    for i, _ in enumerate(predictions[0]):
        for j, _ in enumerate(predictions[0][i]):
            predictions[0][i][j] += counter
            counter += 1
    softmaxed_pred = torch.softmax(predictions, dim=-1)
    output, pba = beam_search_decoder(softmaxed_pred, beam_size)
    ipdb.set_trace()

    assert output[0][0].tolist() == [voc_size-1] * max_len
    assert output[0][1].tolist() == predictions

main()
