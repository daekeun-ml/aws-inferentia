import torch
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers_neuronx.module import save_pretrained_split

from transformers import AutoModelForCausalLM
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_utils import elapsed_time, amp_callback_gpt2, batch_generate_texts

batch_prompts = [
    "Hello, I'm a language model,",
    "Welcome to Amazon Elastic Compute Cloud,",
    "Amazon SageMaker is ", 
    "Machine Learning is ", 
    "GPT is ", 
    "Robot is ", 
    "Deep Learning ", 
    "Language model's future ", 
]

batch_prompts_ko = [
    "근육이 커지기 위해서는",
    "내일 날씨는",
    "수학은",
    "다윈의 진화론은"
]
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--model_id", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--save_path", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.save_path is None: 
        args.save_path = args.model_id.split('/')[-1]
        args.save_path = f'./{args.save_path}-split'
        
    return args

@elapsed_time
def save_splited_model(args):
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, low_cpu_mem_usage=True)
    amp_callback_gpt2(hf_model, torch.float16)
    save_pretrained_split(hf_model, args.save_path)
    
@elapsed_time
def get_neuron_model(args):
    # load HF model to NeuronCores with n-way tensor parallel
    # enable float16 casting     
    neuron_model = GPTJForSampling.from_pretrained(
        args.save_path, 
        batch_size=args.batch_size, 
        tp_degree=args.tp_degree, 
        amp='f16'
    )    
    neuron_model.to_neuron()
    return neuron_model
    
def main(args):
    print(args)
    save_splited_model(args)
    neuron_model = get_neuron_model(args)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    generated_texts = batch_generate_texts(args, batch_prompts[:args.batch_size], tokenizer, neuron_model)
    print(generated_texts)

if __name__ == "__main__":
    main(parse_args())   
