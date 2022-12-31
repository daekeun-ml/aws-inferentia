import os
import csv
import torch
import torch.neuron
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from time import perf_counter
import neuronperf as npf
import neuronperf.torch  # or tensorflow, mxnet

logger = logging.getLogger(__name__)

def generate_sample_inputs(tokenizer, inputs, max_length=128):
    embeddings = tokenizer(inputs, max_length=max_length, padding="max_length", return_tensors="pt")
    return tuple(embeddings.values())

def compile_model(model, tokenizer, max_length=128):
    dummy_inputs = generate_sample_inputs(tokenizer, "dummy", max_length)
    torch.neuron.analyze_model(model, example_inputs=dummy_inputs)
    model_neuron = torch.neuron.trace(model, example_inputs=dummy_inputs)        
    return model_neuron

def benchmark_latency(model, tokenizer, max_length, num_infers=1000):
    logger.info(f"Measuring latency for sequence length={max_length}, num_infers={num_infers}")
    payload = generate_sample_inputs(tokenizer, "dummy", max_length)
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(*payload)
    # Timed run
    for _ in range(num_infers):
        start_time = perf_counter()
        _ =  model(*payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    time_p99_ms = 1000 * np.percentile(latencies,99)    
    return {"avg_ms": time_avg_ms, "std_ms": time_std_ms, "p95_ms": time_p95_ms, "p99_ms": time_p95_ms, "max_length": max_length}
 
def generate_sample_inputs(tokenizer, inputs, max_length=128):
    embeddings = tokenizer(inputs, max_length=max_length, padding="max_length", return_tensors="pt")
    return tuple(embeddings.values())

def load_model_neuron(save_path, model_neuron_name="model_neuron.pt"):
    model_neuron = torch.load(os.path.join(save_path, model_neuron_name))
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    return model_neuron, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_infers", type=int, default=1000)    
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="neuron_model")
    parser.add_argument("--model_id", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    #parser.add_argument("--model_id", type=str, default="roberta-base")
    
    parser.add_argument("--use_neuronperf", type=int, default=0)
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
        
def benchmark_byo(args):
    print(args)
    model_neuron, tokenizer, neuron_model_filepath = get_model_neuron(args)     
    outputs = predict(model_neuron, tokenizer, "i love you")

    benchmark_latency(model_neuron, tokenizer, args.max_length, args.num_infers)
    res = benchmark_latency(model_neuron, tokenizer, args.max_length, args.num_infers)
    print(res)
    benchmark_dict = []
    benchmark_dict.append({**res, "instance_type": "inf1"})    
    
    # write results to csv
    keys = benchmark_dict[0].keys()
    with open(f'benchmmark_{args.model_id.split("/")[-1].replace("-","_")}.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(benchmark_dict)
        
def benchmark_neuronperf(args, batch_sizes=[1]):
    model_neuron, tokenizer, neuron_model_filepath = get_model_neuron(args)   
    payload = generate_sample_inputs(tokenizer, "dummy", max_length=args.max_length)
    reports = npf.torch.benchmark(neuron_model_filepath, payload, batch_sizes)

    # View and save results
    neuronperf.print_reports(reports)
    csv_file = f'benchmmark_neuronperf_{args.model_id.split("/")[-1].replace("-","_")}.csv'
    neuronperf.write_csv(reports, csv_file)
                    
if __name__ == "__main__":
    args = parse_args()    
    if args.use_neuronperf == 1:
        benchmark_neuronperf(args)
    else:
        benchmark_byo(args) 
