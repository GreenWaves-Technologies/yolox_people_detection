DIR_IN=$1
DIR_OUT=$2
mkdir -p DIR_IN 
counter=0
for file in $(ls ./${DIR_IN})
do
    echo "./${DIR_IN}/$file" 
    cp "./${DIR_IN}/$file" ./input.pgm
    make all run platform=gvsoc

    mv ./output.bin "./${DIR_OUT}/$file.bin"
    counter=$((counter+1))
    echo "$counter : $file FIELS FINISHED"
    rm ./input.pgm
done
