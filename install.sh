#!/bin/sh --login
# This is an optional installation script for setting up the nle package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch and Pytorch Geometric (PYG).
#
# Example:
# $ source install.sh cu102  # install nle with CUDA 10.2
#
# To uninstall and remove the nle environment:
# $ conda remove -n nle --all

# Check input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu102,cu113]"
    return 1
fi

# Torch related dependency versions
PYTORCH_VERSION=1.12.1
PYG_VERSION=2.0.4

# Set CUDA variable (use CPU if not set)
CUDA_VERSION=${1:-cpu}
echo "CUDA_VERSION=${CUDA_VERSION}"
case $CUDA_VERSION in
    cpu)
        TORCH_OPT="cpuonly"
        ;;
    cu102)
        TORCH_OPT="cudatoolkit=10.2"
        ;;
    cu113)
        TORCH_OPT="cudatoolkit=11.3"
        ;;
    *)
        echo "ERROR: Unrecognized CUDA_VERSION=${CUDA_VERSION}"
        return 1
        ;;
esac

# Create environment
conda create -n nle python=3.8 -y
conda activate nle

# Install CUDA enabled dependencies
conda install pytorch=${PYTORCH_VERSION} torchvision ${TORCH_OPT} -c pytorch -y
pip install "torch-geometric[full]"==${PYG_VERSION} torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html

# Finally, install nle
pip install -e .

printf "Successfully installed nle, be sure to activate the nle conda environment via:\n"
printf "\n    \$ conda activate nle\n"
printf "\nTo uninstall and remove the nle environment:\n"
printf "\n    \$ conda remove -n nle --all\n\n"
