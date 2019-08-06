#!/bin/bash
home_dir=$(dirname $(realpath $0))
echo home_dir=$home_dir
cd $home_dir

for i in $(ls test_*.py); do
	echo $i
	python $i
	echo 
	echo
done
