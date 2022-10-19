#!/bin/bash
# for every file in the inputs_for_validation folder
# move it to file input_val.bin using bash script 

counter=0
# for file in $(ls ./inputs_for_validataion/)
for file in $(cat left_file.txt)
do
    cp "./inputs_for_validataion/${file}" ./input_val.bin
    make all run platform=gvsoc
    mv ./output.bin ./outputs_for_validation/"$file"
    counter=$((counter+1))
    echo "$counter FIELS FINISHED"
done
