import os
import torch
import torch.neuron
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def generate_sample_inputs(tokenizer, inputs, max_length=128):
    embeddings = tokenizer(inputs, max_length=max_length, padding="max_length", return_tensors="pt")
    return tuple(embeddings.values())
     
def get_model_neuron(args):
    neuron_model_dir = os.path.join(args.save_path, args.model_id)
    neuron_model_filepath = os.path.join(neuron_model_dir, "model_neuron.pt")
    if os.path.exists(neuron_model_filepath):
        print("Load pre-compiled model")
        tokenizer = AutoTokenizer.from_pretrained(neuron_model_dir)
        model_neuron = torch.load(neuron_model_filepath)
    else:
        print("Compile model")
        os.makedirs(neuron_model_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_id, torchscript=True)
        model.eval()
        model_neuron = compile_model(model, tokenizer, args.max_length)
        model_neuron.save(neuron_model_filepath)
        tokenizer.save_pretrained(neuron_model_dir)
    return model_neuron, tokenizer, neuron_model_filepath

def compile_model(model, tokenizer, max_length=128):
    dummy_inputs = generate_sample_inputs(tokenizer, "dummy", max_length)
    torch.neuron.analyze_model(model, example_inputs=dummy_inputs)
    model_neuron = torch.neuron.trace(model, example_inputs=dummy_inputs)        
    return model_neuron

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="neuron_model")
    parser.add_argument("--model_id", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")  
    
    parser_args, _ = parser.parse_known_args()  
    return parser_args

def predict(model, tokenizer, inputs, postprocess_output=False):
    payloads = generate_sample_inputs(tokenizer, inputs)
    outputs = model(*payloads)
    
    if postprocess_output:
        softmax_fn = torch.nn.Softmax(dim=1)
        softmax_outputs = softmax_fn(outputs[0])
        _, pred = torch.max(softmax_outputs, dim=1)
        return (outputs, pred)
    else:
        return outputs

def main(args):
    print(args)
    model_neuron, tokenizer, neuron_model_filepath = get_model_neuron(args)        
    inputs = "I do not like you"
    outputs = predict(model_neuron, tokenizer, inputs, postprocess_output=True)
    print(outputs)
        
if __name__ == "__main__":
    main(parse_args()) 
