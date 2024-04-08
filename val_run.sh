DIR_IN=$1
DIR_OUT=$2
mkdir -p ${DIR_OUT}
counter=0

if [[ $DIR_IN == */ ]] 
then
    DIR_IN=${DIR_IN: 0: -1}
fi


for file in $(ls ${DIR_IN})
do
    echo "${DIR_IN}/$file"
    cmake -B build \
            -DCONFIG_PLATFORM_GVSOC=y \
            -DCONFIG_INFERENCE=y \
            -DCONFIG_CI=n \
            -DCONFIG_IMAGE_ABS_PATH=${DIR_IN}/$file

    cmake --build build --target run -j

    mv ./build/output.bin "${DIR_OUT}/$file.bin"
    counter=$((counter+1))
    echo "$counter : $file FILES FINISHED"
done
