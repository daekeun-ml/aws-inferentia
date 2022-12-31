# AWS Inferentia Hands-on Labs

This repository provides an easy hands-on way to get started with AWS Inferentia.
A demonstration of this hands-on can be seen in the AWS Innovate 2023 - AIML Edition session.
Please refer to the AWS Neuron SDK official developer guide for details.

## Requirements
Before starting, you have met the following requirements:
- AWS Inf1 EC2 instances (https://aws.amazon.com/ko/ec2/instance-types/inf1/)

A CPU instance (e.g., `c5.xlarge`) is also possible as long as you only compile the model with the Neuron SDK.

## Getting Started

### Install AWS Neuron Drivers
If AMI (Amazon Machihe Image) is Amazon Linux 2, please run `install_neuron_amznlinux2.sh`
```bash
$ ./install_neuron_amznlinux2.sh
```
If AMI (Amazon Machihe Image) is Ubuntu 18.04/20.04, please run `install_neuron_ubuntu.sh`
```bash
$ ./install_neuron_ubuntu.sh
```

### Sample codes
- `compile_cv.py`: You can compile VGG, ResNet, ResNeXt, EfficientNet, YOLO-v5, and ViT models with this example code.
- `compile_nlp.py`: You can compile BERT-based encoder models like BERT, DistilBERT, ALBERT, RoBERTa with this example code.
- `benchmark_nlp.py`: Perform latency and throughput benchmarking of BERT-based classification models. If you set `use_neuronperf=0`, only basic latency benchmarking is performed. Change `use_neuronperf=1` to perform benchmarking considering more diverse scenarios and throughput.

```bash
# Example
$ source pytorch_venv/bin/activate
$ python3 benchmark_nlp.py --num_infers 1000 --max_length 128 --model_id distilbert-base-uncased-finetuned-sst-2-english --use_neuronperf 1
```

## References
- AWS Neuron SDK GitHub: https://github.com/aws-neuron/aws-neuron-sdk
- AWS Neuron Samples GitHub: https://github.com/aws-neuron/aws-neuron-samples
- Getting Started AWS Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/torch-neuron.html
* PyTorch Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html
- TensorFlow Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/tensorflow/index.html
- NeuronPerf: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuronperf


## License Summary
This sample code is provided under the MIT-0 license. See the LICENSE file.