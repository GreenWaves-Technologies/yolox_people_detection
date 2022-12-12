#!/bin/bash
mkdir -p $2
counter=0
for file in $(ls ./$1)
do
    echo "./$1/$file" 
    cp "./$1/$file" ./input.pgm
    make all run platform=gvsoc

    mv ./output.bin "./$2/$file.bin"
    counter=$((counter+1))
    echo "$counter : $file FIELS FINISHED"
    rm ./input.pgm
done