# AWS Inferentia Hands-on Labs

This repository provides an easy hands-on way to get started with AWS Inferentia.
A demonstration of this hands-on can be seen in the AWS Innovate 2023 - AIML Edition session.
Please refer to the AWS Neuron SDK official developer guide for details.

## Requirements
Before starting, you have met the following requirements:
- AWS Inf1 EC2 instances (https://aws.amazon.com/ko/ec2/instance-types/inf1/)

A CPU instance (e.g., `c5.xlarge`) is also possible as long as you only compile the model with the Neuron SDK.

## Getting Started

If AMI (Amazon Machihe Image) is Amazon Linux 2, please run `install_neuron_amznlinux2.sh`
```bash
$ ./install_neuron_amznlinux2.sh
```
If AMI (Amazon Machihe Image) is Ubuntu 18.04/20.04, please run `install_neuron_ubuntu.sh`
```bash
$ ./install_neuron_ubuntu.sh
```

## References
- AWS Neuron SDK GitHub: https://github.com/aws-neuron/aws-neuron-sdk
- Getting Started AWS Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/torch-neuron.html
* PyTorch Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html
- TensorFlow Neuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/tensorflow/index.html


## License Summary
This sample code is provided under the MIT-0 license. See the LICENSE file.