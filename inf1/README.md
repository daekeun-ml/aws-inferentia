# Getting Started for Inf1

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
neuron-cc               1.13.5.0+7dcf000a6
neuronperf              1.6.1.0+a63399af5
torch                   1.12.1
torch-neuron            1.12.1.2.5.8.0
torchvision             0.13.1
```

## Sample codes
- `compile_cv.py`: You can compile VGG, ResNet, ResNeXt, EfficientNet, YOLO-v5, and ViT models with this example code.
- `compile_nlp.py`: You can compile BERT-based encoder models like BERT, DistilBERT, ALBERT, RoBERTa with this example code.
- `benchmark_nlp.py`: Perform latency and throughput benchmarking of BERT-based classification models. If you set `use_neuronperf=0`, only basic latency benchmarking is performed. Change `use_neuronperf=1` to perform benchmarking considering more diverse scenarios and throughput.

```bash
# BERT Example
$ cd ~ && source pytorch_venv/bin/activate && cd aws-inferentia/inf1
$ python3 benchmark_nlp.py --num_infers 1000 --max_length 128 --model_id distilbert-base-uncased-finetuned-sst-2-english --use_neuronperf 1

# Image classification Example
$ cd ~ && source aws_neuron_venv_pytorch/bin/activate && cd aws-inferentia/trn1_and_inf2
$ python3 compile_cv.py
```

## References
- AWS Neuron SDK GitHub: https://github.com/aws-neuron/aws-neuron-sdk
- AWS Neuron Samples GitHub: https://github.com/aws-neuron/aws-neuron-samples
- Getting Started AWS Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/torch-neuron.html
- PyTorch Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html
- TensorFlow Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/tensorflow/index.html
- NeuronPerf: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuronperf