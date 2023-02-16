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

# Install git
sudo yum install git -y


# Install OS headers
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

# Remove preinstalled packages and Install Neuron Driver and Runtime
sudo yum remove aws-neuron-dkms -y
sudo yum remove aws-neuronx-dkms -y
sudo yum remove aws-neuronx-oci-hook -y
sudo yum remove aws-neuronx-runtime-lib -y
sudo yum remove aws-neuronx-collectives -y
sudo yum install aws-neuronx-dkms-2.*  -y
sudo yum install aws-neuronx-oci-hook-2.*  -y
sudo yum install aws-neuronx-runtime-lib-2.*  -y
sudo yum install aws-neuronx-collectives-2.*  -y

# install Neuron Driver
sudo yum install aws-neuronx-dkms-2.* -y

# Install Neuron Runtime
sudo yum install aws-neuronx-collectives-2.* -y
sudo yum install aws-neuronx-runtime-lib-2.* -y

# Install EFA Driver(only required for multiinstance training)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
cat aws-efa-installer.key | gpg --fingerprint
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
cd
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# Remove pre-installed package and Install Neuron Tools
sudo yum remove aws-neuron-tools  -y
sudo yum remove aws-neuronx-tools  -y
sudo yum install aws-neuronx-tools-2.*  -y

export PATH=/opt/aws/neuron/bin:$PATH


####################################################################################
# Warning: If Linux kernel is updated as a result of OS package update
#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
####################################################################################

# Install Neuron Tools
sudo yum install aws-neuronx-tools-2.* -y

export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv and activate Python virtual environment to install
# Neuron pip packages.

sudo yum install -y python3.7-venv gcc-c++
python3.7 -m venv aws_neuron_venv_pytorch
source aws_neuron_venv_pytorch/bin/activate
pip install -U pip

# Instal Jupyter notebook kernel
pip install ipykernel
python3.7 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (Neuron PyTorch)"
pip install jupyter notebook
pip install environment_kernels

# Activate a Python 3.7 virtual environment where Neuron pip packages were installed
source aws_neuron_venv_pytorch/bin/activate

# Set Pip repository  to point to the Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
pip install wget awscli

# Install Neuron Compiler
pip install neuronx-cc==2.*

# Install Neuron Framework
pip install torch-neuronx torchvision

# Install additional packages
pip install transformers==4.25.1
pip install opencv-python-headless==4.6.0.66
pip install matplotlib scikit-learn seaborn