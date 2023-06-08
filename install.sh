#!/bin/sh --login
# This is an optional installation script for setting up the obnb package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch and Pytorch Geometric (PYG).
#
# Example:
# $ source install.sh cu117  # install obnb with CUDA 11.7
#
# To uninstall and remove the obnb environment:
# $ conda remove -n obnb --all

# Check input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu117,cu118]"
    return 1
fi

# Torch related dependency versions
PYTORCH_VERSION=2.0.0
PYG_VERSION=2.3.0

# Set CUDA variable (use CPU if not set)
CUDA_VERSION=${1:-cpu}
echo "CUDA_VERSION=${CUDA_VERSION}"
case $CUDA_VERSION in
    cpu)
        TORCH_OPT="cpuonly -c pytorch"
        ;;
    cu117)
        TORCH_OPT="pytorch-cuda=11.7 -c pytorch -c nvidia"
        ;;
    cu118)
        TORCH_OPT="pytorch-cuda=11.8 -c pytorch -c nvidia"
        ;;
    *)
        echo "ERROR: Unrecognized CUDA_VERSION=${CUDA_VERSION}"
        return 1
        ;;
esac

# Create environment
conda create -n obnb python=3.8 -y
conda activate obnb

# Install CUDA enabled dependencies
conda install pytorch=${PYTORCH_VERSION} torchvision torchaudio ${TORCH_OPT} -y
conda install pyg=${PYG_VERSION} -c pyg -y

# Finally, install obnb
pip install -e .

printf "Successfully installed obnb, be sure to activate the obnb conda environment via:\n"
printf "\n    \$ conda activate obnb\n"
printf "\nTo uninstall and remove the obnb environment:\n"
printf "\n    \$ conda remove -n obnb --all\n\n"
