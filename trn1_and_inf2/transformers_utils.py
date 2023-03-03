import time
import torch

def elapsed_time(fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f'### {fn.__name__} elapsed time : {end_time-start_time:.3f}secs')
        return result
    return wrapper_fn

def amp_callback_opt(model, dtype):
    # cast attention and mlp to low precision only; layernorms stay as f32
    for block in model.model.decoder.layers:
        block.self_attn.to(dtype)
        block.fc1.to(dtype)
        block.fc2.to(dtype)
    model.lm_head.to(dtype)

def amp_callback_gpt2(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)

@elapsed_time  
def batch_generate_texts(args, batch_prompts, tokenizer, neuron_model):
    encodings = tokenizer.batch_encode_plus(batch_prompts, padding='longest', pad_to_max_length=True, return_tensors='pt')
    batch_input_ids, batch_attention_masks = encodings["input_ids"], encodings["attention_mask"]

    with torch.inference_mode():
        generated_ids = neuron_model.sample(batch_input_ids, sequence_length=args.seq_length)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts
    