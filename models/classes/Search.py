## adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/search.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional

import torch
import torch.nn as nn
import ipdb
from torch import Tensor


class Search(nn.Module):
    def __init__(self, tgt_dict):
        super().__init__()
        self.pad = tgt_dict['<pad>']
        self.unk = tgt_dict['<unk>']
        self.eos = tgt_dict['</s>']
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(
        self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None
    ):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point
            prev_output_tokens: (bsz x step)
                the previously generated oputput tokens
            original_batch_idxs: (bsz)
                the tensor with the batch indices, in the range [0, bsz)
                this is useful in case there has been applied a re-ordering
                and we need to know the orignal indices

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        """Initialize constraint states for constrained decoding (if supported).

        Args:
            batch_constraints: (torch.Tensor, optional)
                the list of constraints, in packed form
            beam_size: (int)
                the beam size
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        pass

    def prune_sentences(self, batch_idxs: Tensor):
        """
        Removes constraint states for completed sentences (if supported).
        This is called from sequence_generator._generate() when sentences are
        deleted from the batch.

        Args:
            batch_idxs: Indices of *sentences* whose constraint state should be *kept*.
        """
        pass

    def update_constraints(self, active_hypos: Tensor):
        """
        Updates the constraint states by selecting the beam items that are retained.
        This is called at each time step of sequence_generator._generate() when
        the set of 2 * {beam_size} candidate hypotheses are reduced to the beam size.

        Args:
            active_hypos: (batch size, beam size)
              list of integers denoting, for each sentence, which beam candidate items
              should be kept.
        """
        pass


class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None

    @torch.jit.export
    def step(self, step_in_seq, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step_in_seq == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step_in_seq - 1].unsqueeze(-1))

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf

    # def step(
    #     self,
    #     step_in_seq: int,
    #     lprobs,
    #     scores: Optional[Tensor],
    #     prev_output_tokens: Optional[Tensor] = None,
    #     original_batch_idxs: Optional[Tensor] = None,
    # ):
    #     super()._init_buffers(lprobs)
    #
    #     bsz, beam_size, vocab_size = lprobs.size()
    #
    #     if step_in_seq == 0:
    #         # at the first step all hypotheses are equally likely, so use
    #         # only the first beam
    #         lprobs = lprobs[:, ::beam_size, :].contiguous()
    #     else:
    #         # make probs contain cumulative scores for each hypothesis
    #         assert scores is not None
    #         lprobs = lprobs + scores[:, :, step_in_seq - 1].unsqueeze(-1)
    #
    #     torch.topk(
    #         lprobs.view(bsz, -1),
    #         k=min(
    #             # Take the best 2 x beam_size predictions. We'll choose the first
    #             # beam_size of these which don't predict eos to continue with.
    #             beam_size * 2,
    #             lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
    #         ),
    #         out=(self.scores_buf, self.indices_buf),
    #     )
    #     torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
    #     self.indices_buf.fmod_(vocab_size)
    #     return self.scores_buf, self.indices_buf, self.beams_buf
