# Configure Linux for Neuron repository updates
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

# Update OS packages
sudo yum update -y

################################################################################################################
# Remove older versions of Neuron
################################################################################################################
sudo yum remove aws-neuron-dkms -y
sudo yum remove aws-neuronx-dkms -y
sudo yum remove aws-neuron-tools -y
sudo yum remove aws-neuronx-tools -y

################################################################################################################
# To install or update to Neuron versions 2.5 and newer from previous releases:
# - DO NOT skip 'aws-neuronx-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
################################################################################################################

# Install OS headers
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

# Install Neuron Driver
sudo yum install aws-neuronx-dkms -y

####################################################################################
# Warning: If Linux kernel is updated as a result of OS package update
#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
####################################################################################

# Install Neuron Tools
sudo yum install aws-neuronx-tools -y

export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv and activate Python virtual environment to install
# Neuron pip packages.

sudo yum install -y python3.7-venv gcc-c++
python3.7 -m venv pytorch_venv
source pytorch_venv/bin/activate
pip install -U pip

# Instal Jupyter notebook kernel
pip install ipykernel
python3.7 -m ipykernel install --user --name pytorch_venv --display-name "Python (Neuron PyTorch)"
pip install jupyter notebook
pip install environment_kernels

# Activate a Python 3.7 virtual environment where Neuron pip packages were installed
source pytorch_venv/bin/activate

# Set Pip repository  to point to the Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install Neuron PyTorch
pip install torch-neuron neuron-cc[tensorflow] "protobuf<4" torchvision

# Install transformers
pip install transformers==4.25.1

# Install NeuronPerf
pip install neuronperf --extra-index-url=https://pip.repos.neuron.amazonaws.com
