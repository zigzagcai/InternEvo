import os

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from internlm.apis.inference import SequenceGenerator, batch_tokenize
from internlm.initialize import initialize_distributed_env  # noqa: E402
from internlm.train import initialize_model, initialize_parallel_communicator


def load_and_generate(path, model_type="INTERNLM2_PUBLIC", tokenizer_path=""):
    model_cfg = os.path.join(path, "model_config.pt")
    model_wt = os.path.join(path, "model_tp0_pp0.pt")
    model_config = torch.load(model_cfg)
    model_config["apply_post_layer_norm"] = False
    model_config.pop("adapt_hf")
    evo_cfg = dict(
        model_type=model_type,
        model=model_config,
        parallel=dict(
            zero1=dict(size=1, fsdp=False),
            pipeline=dict(size=1, interleaved_overlap=True),
            tensor=dict(size=1, mode="mtp"),
            sequence_parallel=0,
        ),
    )
    initialize_distributed_env(evo_cfg, master_port=23574, args_check=False)

    tokenizer = SentencePieceProcessor(tokenizer_path)  # pylint: disable=E1121
    model = initialize_model()
    _ = initialize_parallel_communicator(model)
    # Directly get the origin model without NativeAMP wrapper.
    model = model.model

    state_dict = torch.load(model_wt)
    load_info = model.load_state_dict(state_dict, strict=False)
    print(load_info)
    prompt = [
        "user\nHow can I keep flys away from my house\nassistant\n",
        "user\nHow can I keep flys away from my house\nassistant\nThe best way is to keep your house clean, "
        "and sweep away from where your meals are prepared, since flys tend to seek out food particles.\n"
        "user\nAny other advice?\nassistant\n",
    ]

    sequenece_generator = SequenceGenerator(
        decoder=model,
        eos_token_id=tokenizer.eos_id(),
        pad_token_id=tokenizer.bos_id(),
        bos_token_id=tokenizer.bos_id(),
        additional_eos_token_list=None,
    )

    input_ids = batch_tokenize(prompt, tokenizer, pad_token_id=tokenizer.bos_id()).cuda()
    generate_kwargs = {}
    output_ids = sequenece_generator.generate(
        input_ids,
        num_return_sequences=generate_kwargs.get("num_return_sequences", 1),
        max_length=generate_kwargs.get("max_length", 100),
        num_beams=generate_kwargs.get("num_beams", 1),
        do_sample=generate_kwargs.get("do_sample", True),
        temperature=generate_kwargs.get("temperature", 1.0),
        top_k=generate_kwargs.get("top_k", 50),
        top_p=generate_kwargs.get("top_p", 1.0),
        repetition_penalty=generate_kwargs.get("repetition_penalty", 1),
        length_penalty=generate_kwargs.get("repetition_penalty", 1.0),
    )

    output_tokens = output_ids.tolist()
    all_output_str = []
    for b in range(len(output_tokens)):
        for sent_idx in range(len(output_tokens[b])):
            cur_output_tokens = output_tokens[b][sent_idx]
            cur_sent = tokenizer.decode(cur_output_tokens)
            all_output_str.append(cur_sent)
    print(all_output_str)
    gt = [
        "user\nHow can I keep flys away from my house\nassistant\nThere are several ways to keep flies away from your\
 house:\n\n1. Use citronella: Place citronella candles or essential oil in areas where flies likes to\
 congregate. This can be a temporary solution,",
        "Here are some tips to eliminate food sources, keep it's a lot of people eat. \nAlso, sprays with essential\
 oils like garlic or vinegar can help?\nTo Keep flies away:\nHere are some more tips",
    ]
    assert all_output_str[0] == gt[0], all_output_str[0]
    assert all_output_str[1][len(prompt[1]) :] == gt[1], all_output_str[1][len(prompt[1]) :]


def test_internlm2_1_8B_generate():
    base_model_dir = os.environ.get("qa_data")
    if base_model_dir is not None:
        model_dir = os.path.join(base_model_dir, "internlm2_1_8B")
        tokenizer_path = os.path.join(base_model_dir, "InternLM_CI_assets/v13.model")
        if os.path.exists(model_dir) and os.path.exists(tokenizer_path):
            load_and_generate(model_dir, tokenizer_path=tokenizer_path)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "-v", "test_generate.py"])
