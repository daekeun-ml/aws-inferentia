# Getting Started for Trn1/Inf2

## Install AWS Neuron Drivers
If AMI (Amazon Machihe Image) is Amazon Linux 2, please run `install_neuron_amznlinux2.sh`
```bash
$ ./install_neuron_amznlinux2.sh
```
If AMI (Amazon Machihe Image) is Ubuntu Server 18.04 or Ubuntu Server 20.04, please run `install_neuron_ubuntu.sh`
```bash
$ ./install_neuron_ubuntu.sh
```
It is also good to check if dependencies are installed successfully with the command below.

```bash
$ source pytorch_venv/bin/activate
$ pip list|grep -e neuron -e torch
>>>
libneuronxla                  0.5.101
neuronx-cc                    2.4.0.21+b7621be18
neuronx-hwm                   2.4.0.1+90172456c
torch                         1.13.1
torch-neuronx                 1.13.0.1.4.0
torch-xla                     1.13.0+torchneuron3
torchvision                   0.14.1
```

## Sample codes
- `compile_cv.py`: You can compile VGG, ResNet, ResNeXt, EfficientNet, and ViT models with this example code.
- `compile_nlp.py`: You can compile BERT-based encoder models like BERT, DistilBERT, ALBERT, RoBERTa with this example code.
- `benchmark_nlp.py`: Perform latency and throughput benchmarking of BERT-based classification models.

```bash
# BERT Example
$ source pytorch_venv/bin/activate
$ python3 benchmark_nlp.py --max_length 128 --model_id distilbert-base-uncased-finetuned-sst-2-english

# Image classification Example
$ source pytorch_venv/bin/activate
$ python3 compile_cv.py
```

## Caution
The input shape you use when compiling with `torch_neuronx.trace()` must match the input shape used during inference. This means that variable length input data cannot be used when generating sentences in transformer decoder-based model like GPT. Therefore, please be sure to feed sample data with a fixed token length (e.g. 128) and change the padding of the tokenizer to left. (`tokenizer.padding_side = "left"`) For more information, please see https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2.

## References
- AWS Neuron SDK GitHub: https://github.com/aws-neuron/aws-neuron-sdk
- AWS Neuron Samples GitHub: https://github.com/aws-neuron/aws-neuron-samples
- Getting Started AWS Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/torch-neuron.html
* PyTorch Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html
- TensorFlow Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/tensorflow/index.html