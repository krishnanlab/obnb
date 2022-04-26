#!/bin/bash --login

source ~/.bashrc

homedir=$(dirname $(realpath $0))
cd $homedir
echo homeidr=$homedir

if [[ -z $1 ]]; then
    echo No conda environment specified.
else
    echo Using conda environment $1
    conda activate $1
fi

for script in $(ls *.py); do
    echo Start runing example script: $script
    python $script
done
