# Copyright (c) InternLM. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from internlm.apis import InferenceParams, process_parallel_output
from internlm.core.context import ParallelMode  # noqa: E402
from internlm.core.context import global_context as gpc  # noqa: E402
from internlm.core.trainer import Trainer

__all__ = ["SequenceGenerator"]


def _get_model_device(model):
    """
    obtain the device of an nn.Module.model

    Args:
        model: nn.Module

    Return: torch.device. if None, the parameters of this model is None.
    """
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


class SequenceGenerator:
    """
    Sequence Generator.
    """

    def __init__(
        self,
        decoder,
        eos_token_id,
        pad_token_id,
        bos_token_id,
        additional_eos_token_list=None,
        add_eos_when_return=False,
    ):
        self.decoder = decoder
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.additional_eos_token_list = additional_eos_token_list
        self.add_eos_when_return = add_eos_when_return

    @torch.no_grad()
    def generate(
        self,
        tokens: "torch.LongTensor" = None,
        num_return_sequences=1,
        max_length: int = 20,
        num_beams: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        min_new_tokens: int = 1,
    ):
        """
        Args:
            tokens: the beginning tokens whose shape is [bsz, length]. If shape is None, default ''bos_token'' will be
                added to conduct generation.
            num_return_sequences: number of returned sequences.
            max_length: the max length of generated sequence.
            num_beams: the size of beam search.
            do_sample: whether using sample.
            temperature: it's meaningful when do_sample is True.
            top_k: sampling from top_k.
            top_p: sampling from top_p tokens(nucleus sampling).
            repetition_penalty: the penalty degree for repetition tokens
            length_penalty: the penalty for length.
            min_new_tokens: minimum number of generated tokens.

        Return:
            the token sequence whose shape is [bsz, num_return_sequences, max_length]. If eos_token_id is not None,
                the ending of each sequence must be eos_token_id.
        """
        assert num_return_sequences <= num_beams, f"The `{num_return_sequences}` must be less than `{num_beams}`..."
        if do_sample:
            return sample_generate(
                self.decoder,
                tokens=tokens,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.eos_token_id,  # the ending token id
                additional_eos_token_list=self.additional_eos_token_list,
                add_eos_when_return=self.add_eos_when_return,
                pad_token_id=self.pad_token_id,
                repetition_penalty=repetition_penalty,  # the penalty degree for repetition tokens
                length_penalty=length_penalty,  # the penalty for length. if it > 1, then encourages long sequence.
                # Otherwise, encourages short sequence.
                bos_token_id=self.bos_token_id,
                min_new_tokens=min_new_tokens,
            )
        else:
            return greedy_generate(
                self.decoder,
                tokens=tokens,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.eos_token_id,
                additional_eos_token_list=self.additional_eos_token_list,
                add_eos_when_return=self.add_eos_when_return,
                pad_token_id=self.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                bos_token_id=self.bos_token_id,
                min_new_tokens=min_new_tokens,
            )

    @torch.no_grad()
    def streaming_generate(
        self,
        tokens: "torch.LongTensor" = None,
        max_length: int = 20,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        min_new_tokens: int = 1,
    ):
        if not do_sample:
            temperature = 1
            top_k = 50
            top_p = 1
        yield from _streaming_no_beam_search_generate(
            self.decoder,
            tokens=tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.eos_token_id,
            additional_eos_token_list=self.additional_eos_token_list,
            add_eos_when_return=self.add_eos_when_return,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            min_new_tokens=min_new_tokens,
        )


@torch.no_grad()
def greedy_generate(
    decoder,
    tokens=None,
    max_length=20,
    num_beams=1,
    num_return_sequences=1,
    eos_token_id=None,
    additional_eos_token_list=None,
    add_eos_when_return=False,
    pad_token_id=1,
    repetition_penalty=1,
    length_penalty=1.0,
    bos_token_id=1,
    min_new_tokens=1,
):
    """
    Search sequence greedily.

    Args:
        decoder: the Decoder object.
        tokens: the shape is [batch size, length]. If decoder is None, generating begins with bos_token_id.
        max_length: the max length for generated sequence.
        num_beams: the size of beam to decode.
        eos_token_id: the ending token id. If None, the decode length is max_length.
        pad_token_id: the token id of pad.
        repetition_penalty: the penalty degree for repetition tokens
        length_penalty: the penalty for length.
        min_new_tokens: minimum number of generated tokens.

    """
    if num_beams == 1:
        token_ids = _no_beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            temperature=1,
            top_k=50,
            top_p=1,
            eos_token_id=eos_token_id,
            additional_eos_token_list=additional_eos_token_list,
            add_eos_when_return=add_eos_when_return,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            min_new_tokens=min_new_tokens,
        )
    else:
        token_ids = _beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1,
            top_k=50,
            top_p=1,
            eos_token_id=eos_token_id,
            additional_eos_token_list=additional_eos_token_list,
            add_eos_when_return=add_eos_when_return,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            min_new_tokens=min_new_tokens,
        )

    return token_ids


@torch.no_grad()
def sample_generate(
    decoder,
    tokens,
    max_length=20,
    num_beams=1,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    additional_eos_token_list=None,
    add_eos_when_return=False,
    pad_token_id=1,
    repetition_penalty=1.0,
    length_penalty=1.0,
    bos_token_id=1,
    min_new_tokens=1,
):
    """
    generate sequence in sampling way.

    Args:
        decoder: the Decoder object.
        tokens: the shape is [batch size, length]. If decoder is None, generating begins with bos_token_id.
        max_length: the max length for generated sequence.
        num_beams: the size of beam to decode.
        num_return_sequences: number of returned sequence.
        temperature: annealing magnitude during sampling.
        top_k: sampling from top_k. (Default: 50)
        top_p: sampling from top_p tokens(nucleus sampling). (Default: 1.0)
        eos_token_id: the ending token id. If None, the decode length is max_length.
        pad_token_id: the token id of pad.
        repetition_penalty: the penalty degree for repetition tokens
        length_penalty: the penalty for length.
        min_new_tokens: minimum number of generated tokens.

    """
    if num_beams == 1:
        token_ids = _no_beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            additional_eos_token_list=additional_eos_token_list,
            add_eos_when_return=add_eos_when_return,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            min_new_tokens=min_new_tokens,
        )
    else:
        token_ids = _beam_search_generate(
            decoder,
            tokens=tokens,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            additional_eos_token_list=additional_eos_token_list,
            add_eos_when_return=add_eos_when_return,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            min_new_tokens=min_new_tokens,
        )
    return token_ids


@torch.no_grad()
def _streaming_no_beam_search_generate(
    decoder,
    tokens,
    inference_params=None,
    max_length=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    additional_eos_token_list=None,
    add_eos_when_return=False,
    do_sample=True,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=1,
    bos_token_id=1,
    min_new_tokens=1,
):
    batch_size, cur_len = tokens.shape
    real_max_length = max_length
    real_min_length = cur_len + min_new_tokens
    if eos_token_id is not None:
        if not isinstance(eos_token_id, (List, Tuple)):
            eos_token_id = [eos_token_id]
        if additional_eos_token_list is not None:
            if not isinstance(additional_eos_token_list, (List, Tuple)):
                additional_eos_token_list = [additional_eos_token_list]
            eos_token_id.extend(additional_eos_token_list)
        eos_token_id = torch.LongTensor(eos_token_id).to(tokens.device)

    assert bos_token_id == pad_token_id, "bos_token_id should be equal to left pad_token_id!"
    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))

    attention_mask = get_attention_mask(tokens, has_bos, bos_token_id=bos_token_id)

    if inference_params is None:
        inference_params = InferenceParams(
            max_sequence_len=max_length,
            max_batch_size=tokens.size(0),
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=attention_mask,
        )

    if isinstance(decoder, torch.nn.Module):
        scores = decoder(**{"input_ids": tokens, "inference_params": inference_params})
    elif isinstance(decoder, Trainer):
        data = {"input_ids": tokens, "inference_params": inference_params}
        model_output, _, _ = decoder.execute_schedule(
            (data, None), forward_only=True, return_loss=False, return_output_label=True
        )
        scores = torch.cat(model_output, dim=0)
    else:
        raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    scores = scores[:, -1].float()
    inference_params.sequence_len_offset += tokens.size(1)
    if eos_token_id is not None and min_new_tokens > 0:
        scores[:, eos_token_id] = -float("inf")

    # The first token generated.
    next_tokens = scores.argmax(dim=-1, keepdim=True)
    dones = next_tokens.new_zeros(batch_size, 1).eq(1)

    if eos_token_id is not None:
        end_mask = torch.any(next_tokens[:, None].eq(eos_token_id), dim=-1)
        dones = dones.__or__(end_mask)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

    while cur_len < real_max_length and dones.min() != 1:
        # batch_size x vocab_size
        attention_mask = get_attention_mask(token_ids, has_bos, bos_token_id=bos_token_id)

        if isinstance(decoder, torch.nn.Module):
            inference_params.attention_mask = attention_mask
            scores = decoder(**{"input_ids": token_ids[:, -1:], "inference_params": inference_params})
        elif isinstance(decoder, Trainer):
            inference_params.set_attention_mask(attention_mask)
            data = {"input_ids": token_ids[:, -1:], "inference_params": inference_params}
            model_output, _, _ = decoder.execute_schedule(
                (data, None), forward_only=True, return_loss=False, return_output_label=True
            )
            scores = torch.cat(model_output, dim=0)
        else:
            raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        inference_params.sequence_len_offset += 1

        if eos_token_id is not None and cur_len < real_min_length:
            scores[..., eos_token_id] = -float("inf")

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = (
                lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            )
            scores.scatter_(dim=1, index=token_ids, src=token_scores)
        # scores: [bsz, vocab_size]
        if eos_token_id is not None and length_penalty != 1.0:
            # batch_size x vocab_size
            eos_token_scores = scores[:, eos_token_id].clone()
            scores = scores / cur_len**length_penalty
            scores[:, eos_token_id] = eos_token_scores
            del eos_token_scores

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=2)
            # add 1e-12 to avoid https://github.com/pytorch/pytorch/pull/27523
            probs = F.softmax(scores, dim=-1) + 1e-12

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # batch_size
        else:
            next_tokens = torch.argmax(scores, dim=-1)  # batch_size

        if eos_token_id is not None:
            # When the generated result exceeds the length, its eos_token_id is set to the most basic terminator.
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1), eos_token_id[0])

        tokens = next_tokens.unsqueeze(1)
        tokens = tokens.masked_fill(dones, pad_token_id)
        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        yield token_ids

        if eos_token_id is not None:
            end_mask = torch.any(next_tokens[:, None].eq(eos_token_id), dim=-1)
            dones = dones.__or__(end_mask)

        cur_len += 1

        if dones.min() == 1:
            break

    # token_ids: [bsz, seqlen]
    if eos_token_id is not None and add_eos_when_return:
        token_ids = torch.cat([token_ids, token_ids.new_full((token_ids.size(0), 1), eos_token_id[0])], dim=1)

    yield token_ids


@torch.no_grad()
def _no_beam_search_generate(
    decoder,
    tokens,
    inference_params=None,
    max_length=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    additional_eos_token_list=None,
    add_eos_when_return=False,
    do_sample=True,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=1,
    bos_token_id=1,
    min_new_tokens=1,
):

    batch_size, cur_len = tokens.shape
    real_max_length = max_length
    real_min_length = cur_len + min_new_tokens
    if eos_token_id is not None:
        if not isinstance(eos_token_id, (List, Tuple)):
            eos_token_id = [eos_token_id]
        if additional_eos_token_list is not None:
            if not isinstance(additional_eos_token_list, (List, Tuple)):
                additional_eos_token_list = [additional_eos_token_list]
            eos_token_id.extend(additional_eos_token_list)
        eos_token_id = torch.LongTensor(eos_token_id).to(tokens.device)

    assert bos_token_id == pad_token_id, "bos_token_id should be equal to left pad_token_id!"
    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))

    attention_mask = get_attention_mask(tokens, has_bos, bos_token_id)

    if inference_params is None:
        inference_params = InferenceParams(
            max_sequence_len=max_length,
            max_batch_size=tokens.size(0),
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=attention_mask,
        )

    if isinstance(decoder, torch.nn.Module):
        scores = decoder(**{"input_ids": tokens, "inference_params": inference_params})
    elif isinstance(decoder, Trainer):
        data = {"input_ids": tokens, "inference_params": inference_params}
        model_output, _, _ = decoder.execute_schedule(
            (data, None), forward_only=True, return_loss=False, return_output_label=True
        )
        scores = process_parallel_output(model_output)
    else:
        raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

    if gpc.is_last_rank(ParallelMode.PIPELINE):
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        if eos_token_id is not None and min_new_tokens > 0:
            scores[:, eos_token_id] = -float("inf")

        # The first token generated.
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    else:
        next_tokens = tokens.new_zeros([batch_size, 1])

    if gpc.is_initialized(ParallelMode.PIPELINE):
        # broadcast to other rank in PP group
        torch.distributed.broadcast(
            next_tokens,
            src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
            group=gpc.get_group(ParallelMode.PIPELINE),
        )

    dones = next_tokens.new_zeros(batch_size, 1).eq(1)

    if eos_token_id is not None:
        end_mask = torch.any(next_tokens[:, None].eq(eos_token_id), dim=-1)
        dones = dones.__or__(end_mask)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)

    inference_params.sequence_len_offset += tokens.size(1)

    real_max_length = max_length
    max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

    while cur_len < real_max_length and dones.min() != 1:
        # batch_size x vocab_size
        attention_mask = get_attention_mask(token_ids, has_bos, bos_token_id=bos_token_id)

        if isinstance(decoder, torch.nn.Module):
            inference_params.attention_mask = attention_mask
            scores = decoder(**{"input_ids": token_ids[:, -1:], "inference_params": inference_params})
        elif isinstance(decoder, Trainer):
            inference_params.set_attention_mask(attention_mask)
            data = {"input_ids": token_ids[:, -1:], "inference_params": inference_params}
            model_output, _, _ = decoder.execute_schedule(
                (data, None), forward_only=True, return_loss=False, return_output_label=True
            )
            scores = process_parallel_output(model_output)
        else:
            raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

        inference_params.sequence_len_offset += 1

        if eos_token_id is not None and cur_len < real_min_length:
            scores[..., eos_token_id] = -float("inf")

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            if isinstance(scores, (list, tuple)):
                scores = scores[0]
            scores = scores[:, -1].float()

            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = (
                    lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                )
                scores.scatter_(dim=1, index=token_ids, src=token_scores)
            # scores: [bsz, vocab_size]
            if eos_token_id is not None and length_penalty != 1.0:
                # batch_size x vocab_size
                eos_token_scores = scores[:, eos_token_id].clone()
                scores = scores / cur_len**length_penalty
                scores[:, eos_token_id] = eos_token_scores
                del eos_token_scores

            if do_sample:
                if temperature > 0 and temperature != 1:
                    scores = scores / temperature

                scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=2)
                # add 1e-12 to avoid https://github.com/pytorch/pytorch/pull/27523
                probs = F.softmax(scores, dim=-1) + 1e-12

                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # batch_size
            else:
                next_tokens = torch.argmax(scores, dim=-1)  # batch_size
        else:
            next_tokens = tokens.new_zeros(batch_size)

        if gpc.is_initialized(ParallelMode.PIPELINE):
            # broadcast to other rank in PP group
            torch.distributed.broadcast(
                next_tokens,
                src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
                group=gpc.get_group(ParallelMode.PIPELINE),
            )

        if eos_token_id is not None:
            # When the generated result exceeds the length, its eos_token_id is set to the most basic terminator.
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1), eos_token_id[0])

        tokens = next_tokens.unsqueeze(1)
        tokens = tokens.masked_fill(dones, pad_token_id)
        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        if eos_token_id is not None:
            end_mask = torch.any(tokens[:, None].eq(eos_token_id), dim=-1)
            dones = dones.__or__(end_mask)

        cur_len += 1

        if dones.min() == 1:
            break

    # token_ids: [bsz, seqlen]
    if eos_token_id is not None and add_eos_when_return:
        token_ids = torch.cat([token_ids, token_ids.new_full((token_ids.size(0), 1), eos_token_id[0])], dim=1)

    # In order to maintain consistency with the results returned by beam search,
    #  a new dimension is added here representing num_return_sequences.
    # token_ids: [bsz, num_return_sequences, seqlen]
    return token_ids[:, None]


@torch.no_grad()
def _beam_search_generate(
    decoder,
    tokens,
    inference_params=None,
    max_length=20,
    num_beams=4,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    eos_token_id=None,
    additional_eos_token_list=None,
    add_eos_when_return=False,
    do_sample=True,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=1,
    bos_token_id=1,
    min_new_tokens=1,
) -> torch.LongTensor:

    device = tokens.device
    batch_size, cur_len = tokens.shape
    real_max_length = max_length
    real_min_length = cur_len + min_new_tokens

    if eos_token_id is not None:
        if not isinstance(eos_token_id, (List, Tuple)):
            eos_token_id = [eos_token_id]
        if additional_eos_token_list is not None:
            if not isinstance(additional_eos_token_list, (List, Tuple)):
                additional_eos_token_list = [additional_eos_token_list]
            eos_token_id.extend(additional_eos_token_list)
        eos_token_id = torch.LongTensor(eos_token_id).to(tokens.device)

    assert bos_token_id == pad_token_id, "bos_token_id should be equal to left pad_token_id!"
    has_bos = torch.all(tokens[:, 0].eq(bos_token_id))

    attention_mask = get_attention_mask(tokens, has_bos, bos_token_id=bos_token_id)

    if inference_params is None:
        inference_params = InferenceParams(
            max_sequence_len=max_length,
            max_batch_size=tokens.size(0),
            sequence_len_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=None,
            lengths_per_sample=None,
            attention_mask=attention_mask,
        )

    if isinstance(decoder, torch.nn.Module):
        scores = decoder(**{"input_ids": tokens, "inference_params": inference_params})
    elif isinstance(decoder, Trainer):
        data = {"input_ids": tokens, "inference_params": inference_params}
        model_output, _, _ = decoder.execute_schedule(
            (data, None), forward_only=True, return_loss=False, return_output_label=True
        )
        scores = process_parallel_output(model_output)
    else:
        raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

    inference_params.sequence_len_offset += tokens.size(1)

    if gpc.is_last_rank(ParallelMode.PIPELINE):
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores[:, -1].float()
        if eos_token_id is not None and min_new_tokens > 0:
            scores[:, eos_token_id] = -float("inf")
        vocab_size = scores.size(1)
        assert vocab_size >= num_beams, "num_beams should be smaller than " "the number of vocabulary size."

        # The first token generated.
        if do_sample:
            probs = F.softmax(scores, dim=-1) + 1e-12
            # (batch_size, num_beams)
            next_tokens = torch.multinomial(probs, num_samples=num_beams)
            logits = probs.log()
            # (batch_size, num_beams)
            next_scores = logits.gather(dim=1, index=next_tokens)
        else:
            scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
            # obtain (batch_size, num_beams), (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)
    else:
        next_tokens = tokens.new_zeros([batch_size, num_beams])
        next_scores = torch.zeros([batch_size, num_beams], dtype=torch.float32, device=next_tokens.device)

    if gpc.is_initialized(ParallelMode.PIPELINE):
        # broadcast to other rank in PP group
        torch.distributed.broadcast(
            next_tokens,
            src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
            group=gpc.get_group(ParallelMode.PIPELINE),
        )
        torch.distributed.broadcast(
            next_scores,
            src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
            group=gpc.get_group(ParallelMode.PIPELINE),
        )

    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    inference_params.reorder_state(indices)

    # batch_size * num_beams x length
    tokens = tokens.index_select(dim=0, index=indices)
    # genrated token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    cur_len = token_ids.size(1)

    max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)
    hypos = [
        BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]
    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    for batch_idx in range(batch_size):
        dones[batch_idx] = (
            dones[batch_idx]
            or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item())
            or max_lengths[batch_idx * num_beams] == cur_len + 1
        )

    while cur_len < real_max_length and not all(dones):
        attention_mask = get_attention_mask(token_ids, has_bos, bos_token_id=bos_token_id)

        if isinstance(decoder, torch.nn.Module):
            inference_params.attention_mask = attention_mask
            scores = decoder(**{"input_ids": token_ids[:, -1:], "inference_params": inference_params})
        elif isinstance(decoder, Trainer):
            inference_params.set_attention_mask(attention_mask)
            data = {"input_ids": token_ids[:, -1:], "inference_params": inference_params}
            model_output, _, _ = decoder.execute_schedule(
                (data, None), forward_only=True, return_loss=False, return_output_label=True
            )
            scores = process_parallel_output(model_output)
        else:
            raise NotImplementedError(f"Unsupported decoder type: {type(decoder)}")

        inference_params.sequence_len_offset += 1

        if gpc.is_last_rank(ParallelMode.PIPELINE):

            if isinstance(scores, (list, tuple)):
                scores = scores[0]
            scores = scores[:, -1].float()
            inference_params.sequence_len_offset += 1
            if eos_token_id is not None and cur_len < real_min_length:
                scores[..., eos_token_id] = -float("inf")

            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = (
                    lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                )

                scores.scatter_(dim=1, index=token_ids, src=token_scores)

            if eos_token_id is not None:
                max_len_eos_mask = max_lengths.eq(cur_len + 1)
                # When the generated result exceeds the length, its eos_token_id is set to the most basic terminator.
                eos_scores = scores[:, eos_token_id[0]]
                scores[:, eos_token_id[0]] = torch.where(max_len_eos_mask, eos_scores + 1e32, eos_scores)

            if do_sample:
                if temperature > 0 and temperature != 1:
                    scores = scores / temperature

                scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=num_beams + 1)
                # add 1e-12 to avoid https://github.com/pytorch/pytorch/pull/27523
                probs = F.softmax(scores, dim=-1) + 1e-12

                # batch_size' x (num_beams+1)
                _tokens = torch.multinomial(probs, num_samples=num_beams + 1)

                logits = probs.log()
                # batch_size' x (num_beams+1)
                _scores = logits.gather(dim=1, index=_tokens)
                # batch_size' x (num_beams+1)
                _scores = _scores + beam_scores[:, None]
                _scores = _scores.view(batch_size, num_beams * (num_beams + 1))
                next_scores, ids = _scores.topk(2 * num_beams, dim=1, largest=True, sorted=True)
                _tokens = _tokens.view(batch_size, num_beams * (num_beams + 1))
                # (batch_size, 2*num_beams)
                next_tokens = _tokens.gather(dim=1, index=ids)
                # (batch_size, 2*num_beams)
                from_which_beam = torch.floor(ids.float() / (num_beams + 1)).long()
            else:
                # (batch_size * num_beams, vocab_size)
                scores = F.log_softmax(scores, dim=-1)
                # (batch_size * num_beams, vocab_size)
                _scores = scores + beam_scores[:, None]
                # (batch_size, num_beams*vocab_size)
                _scores = _scores.view(batch_size, -1)
                # (bsz, 2*num_beams)
                next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
                # (batch_size, 2*num_beams)
                from_which_beam = torch.floor(ids.float() / vocab_size).long()
                next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)
        else:
            next_tokens = tokens.new_zeros([batch_size, 2 * num_beams])
            next_scores = torch.zeros([batch_size, 2 * num_beams], dtype=torch.float32, device=next_tokens.device)
            from_which_beam = torch.zeros([batch_size, 2 * num_beams], dtype=torch.int64, device=next_tokens.device)

        if gpc.is_initialized(ParallelMode.PIPELINE):
            # broadcast to other rank in PP group
            torch.distributed.broadcast(
                next_tokens,
                src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
                group=gpc.get_group(ParallelMode.PIPELINE),
            )
            torch.distributed.broadcast(
                next_scores,
                src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
                group=gpc.get_group(ParallelMode.PIPELINE),
            )
            torch.distributed.broadcast(
                from_which_beam,
                src=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1],
                group=gpc.get_group(ParallelMode.PIPELINE),
            )

        not_eos_mask = torch.all(next_tokens[..., None].ne(eos_token_id), dim=-1)
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
        keep_mask = not_eos_mask.__and__(keep_mask)

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        flag = True
        if cur_len + 1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)
        else:
            effective_eos_mask = torch.any(
                next_tokens[:, :num_beams][..., None].eq(eos_token_id), dim=-1
            )  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(
                eos_batch_idx.tolist(), eos_beam_ind.tolist(), eos_beam_idx.tolist()
            ):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    if eos_token_id is not None:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)
        inference_params.reorder_state(reorder_inds)
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = (
                dones[batch_idx]
                or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item())
                or max_lengths[batch_idx * num_beams] == cur_len + 1
            )

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size, num_return_sequences)
    best = []

    for i, hypotheses in enumerate(hypos):
        # best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        sorted_hyp = list(sorted(hypotheses.hyp, key=lambda x: x[0], reverse=True))
        _best = []
        for j, hyp in zip(range(num_return_sequences), sorted_hyp):
            hyp = hyp[1]
            if eos_token_id is not None and add_eos_when_return:
                # When forcing eos to be added at the end of the generated result, use the most basic text terminator.
                hyp = torch.cat([hyp, token_ids.new_ones(1) * eos_token_id[0]])
            tgt_len[i, j] = len(hyp)
            _best.append(hyp)
        best.append(_best)

    # generate target batch
    decoded = token_ids.new_zeros(batch_size, num_return_sequences, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        for j, _hypo in enumerate(hypo):
            decoded[i, j, : tgt_len[i, j]] = _hypo

    # decoded: [bsz, num_return_sequences, seqlen]
    return decoded


class BeamHypotheses(object):
    """
    BeamHypotheses
    """

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """Initialize n-best list of hypotheses."""
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we are
        done with this sentence."""
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length**self.length_penalty


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Based on the values of top_k and top_p, set the values that do not meet the criteria to the filter_value.

    Args:
        logits: logit value, shape is [bsz, vocab_size].
        top_k: If it is greater than 0, only the probabilities of the top_k vocabulary are kept, and the rest of
            the positions are set to filter_value.
        top_p: according to http://arxiv.org/abs/1904.09751.
        filter_value: filter value
        min_tokens_to_keep: The probability of words in each sample‘s returned distribution will not be
            lower than this value.

    """
    if top_k > 0:
        # Safety check
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of
        # the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            # (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def get_attention_mask(tokens, has_bos, bos_token_id=1):
    if has_bos:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        bos_sum = bos_pos.cumsum(dim=-1)
        bos_pos = torch.where(bos_sum.eq(bos_sum[:, -1:]), 0, 1)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
    else:
        bos_pos = torch.where(tokens.eq(bos_token_id), 1, 0)
        to_atten_x = bos_pos[:, :, None]
        to_atten_y = bos_pos[:, None, :]
    # attention_mask = torch.logical_or(to_atten_x, to_atten_y).eq(1)
    to_atten_y_new = to_atten_y.repeat(1, to_atten_x.shape[1], 1)
    to_atten_x_new = to_atten_x.repeat(1, 1, to_atten_y.shape[2])
    attention_mask = torch.logical_or(to_atten_x_new, to_atten_y_new).eq(1)

    return attention_mask


def batch_tokenize_process_fn(
    batch: Union[List[str], List[Dict], Dict], tokenizer, add_bos: bool = True, add_eos: bool = False
) -> Union[List, Dict]:
    """Data post-processing function for tokenize.

    This function can be directly used in the map function of ``DatasetDict`` and supports batched=True.

    Args:
        batch (Union[List[str], List[Dict], Dict]): Data used to tokenize which can be of the following
         categories:
            (a) A list whose content can be a string or a dictionary. If it is a dictionary,
                it needs to contain the "content" field;
            (b) A dictionary-like object, which should contain the "content" field.
        tokenizer : Currently only sentencepiece is supported.
        add_bos (bool, optional): Whether to add bos token. Defaults to True.
        add_eos (bool, optional): Whether to add eos token. Defaults to False.

    Returns:
        Union[List, Dict]: tokenized data.
    """

    def _tokenize(text):
        tokens = [tokenizer.bos_id()] if add_bos else []
        tokens += tokenizer.encode(text)
        if add_eos:
            tokens.append(tokenizer.eos_id())
        return tokens

    if isinstance(batch, (List, Tuple)):
        if len(batch) == 0:
            return None
        if isinstance(batch[0], str):
            return [_tokenize(w) for w in batch]
        if isinstance(batch[0], Dict):
            for sample in batch:
                sample["input_ids"] = _tokenize(sample["content"])
            return batch
    elif isinstance(batch, str):
        raise NotImplementedError("Do not support a single str as input.")
    else:
        try:
            batch["input_ids"] = [_tokenize(w) for w in batch["content"]]
            batch.pop("content")
            return batch
        except Exception as e:
            print(f"The type of parameter ``batch`` is wrong, type:{type(batch)}, batch: {batch}.")
            raise e


def pad_input_ids(batch: List[Dict], pad_token_id: int = 0, return_dict: bool = False) -> Union[Dict, torch.Tensor]:
    """Tokenize a list of prompts with Left Padding.

    Args:
        batch (List[Dict, List]): if batch[0] is a dict, then key 'input_ids' must exist,
            and value must be a list of integers.
        pad_token_id (int, optional): Defaults to 0.
        return_dict (bool, optional): Defaults to False.

    Returns:
        Union[Dict, torch.Tensor]: input_ids or dict(input_ids=input_ids)
    """
    assert isinstance(batch, list), "batch must be a list"

    input_ids = []
    max_length = max([len(w["input_ids"] if isinstance(w, Dict) else w) for w in batch])
    for sample in batch:
        cur_input_ids = sample["input_ids"] if isinstance(sample, Dict) else sample
        assert len(cur_input_ids) > 0, "got empty list"
        assert isinstance(cur_input_ids[0], int), f"only support a list of integers, but got {type(cur_input_ids[0])}"
        cur_input_ids = torch.LongTensor(cur_input_ids)
        # left padding for generation
        input_ids.append(
            torch.cat(
                [
                    cur_input_ids.new_full((max_length - len(cur_input_ids),), fill_value=pad_token_id),
                    cur_input_ids,
                ]
            )
        )
    input_ids = torch.stack(input_ids)
    return input_ids if not return_dict else {"input_ids": input_ids}


def batch_tokenize(
    prompts: List[str], tokenizer, return_dict: bool = False, pad_token_id: int = 1
) -> Union[Dict, torch.Tensor]:
    """Tokenize a list of prompts with Left Padding. Return the tokens.

    Args:
        prompts (List[str]):  a list of prompts
        tokenizer : Currently only sentencepiece is supported.
        return_dict (bool, optional): Defaults to False.
        pad_token_id (int, optional): Defaults to 1.

    Returns:
        Union[Dict, torch.Tensor]: input_ids or dict(input_ids=input_ids)
    """

    tokenizer_out = batch_tokenize_process_fn(prompts, tokenizer)

    tokens = pad_input_ids(tokenizer_out, return_dict=return_dict, pad_token_id=pad_token_id)

    return tokens
