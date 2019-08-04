#!/bin/bash

fp=$(dirname $(realpath $0))
echo $fp

cd $fp
if [ -d data ]; then
	rm -rf data/*
else
	mkdir data/
fi

mkdir data/networks
mkdir data/labels
mkdir data/properties

data_path=/mnt/research/compbio/krishnanlab/data
network_path=$data_path/networks

ln -s $network_path/string/v10/String_experiments.edg data/networks/STRING-EXP.edg
ln -s $network_path/BioGRID/BioGrid_3.4.136.edgelist data/networks/BioGRID

ln -s $data_path/functional_annotations/kegg/c2.cp.kegg.v6.1.entrez.BP.gsea-min10-max200-ovlppt7-jacpt5.nonred.gmt data/labels/KEGGBP.gmt
ln -s $data_path/disease-gene_annotations/disgenet/disgenet_disease-genes_prop.gsea-min10-max600-ovlppt8-jacpt5.nonred.gmt data/labels/DisGeNet.gmt

ln -s $data_path/pubmed/gene2pubmed_human_gene-counts.txt data/properties/pubcnt.txt

