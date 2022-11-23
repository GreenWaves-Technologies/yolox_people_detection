#!/bin/bash
# for every file in the inputs_for_validation folder
# move it to file input_val.bin using bash script 

# counter=0
# # for file in $(ls ./inputs_for_gvsoc)
# for file in $(ls ./nntool_model_outputs)
# do
#     # cp "./inputs_for_gvsoc/$file" ./input.ppm
#     cp "./nntool_model_outputs/$file" ./input.bin
#     make all run platform=gvsoc

#     # mv ./output.bin "./outputs_from_gvsoc/$file.bin"
#     mv ./output.bin "./nntool_outputs_from_gvsoc/$file"
#     counter=$((counter+1))
#     echo "$counter : $file FIELS FINISHED"
#     rm ./input.bin
# done

counter=0
for file in $(ls ./inputs_for_gvsoc)
do
    cp "./inputs_for_gvsoc/$file" ./input.ppm
    make all run platform=gvsoc

    mv ./output.bin "./outputs_from_gvsoc/$file.bin"
    counter=$((counter+1))
    echo "$counter : $file FIELS FINISHED"
    rm ./input.bin
done