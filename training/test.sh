#!/usr/bin/env bash

./build.sh
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# make the artifact folder writable
read -r -p "I will now recursively delete  $SCRIPTPATH/../inference/artifact/*  and   $2/* . Are you sure you want to continue? (type 'yes')? " CONT
if [ "$CONT" != "yes" ]
then
  echo "Aborting."
  exit 0
fi

chmod 777 $SCRIPTPATH/../inference/artifact/
chmod 777 $2/

# Clear the artifact folder
rm -r $SCRIPTPATH/../inference/artifact/*
rm -r $2/*

# Run the algorithm
MEMORY="32g"

docker run --rm --gpus all \
        --memory=$MEMORY --memory-swap=$MEMORY \
        --cap-drop=ALL --cap-add SYS_NICE --security-opt="no-new-privileges" \
        --network none --shm-size=1G --pids-limit 256 \
        -v /mnt/NAS_25/RAW/STOIC2021/TestDocker/:/input/ \
        -v $SCRIPTPATH/../inference/artifact/:/output/ \
		-v $2:/scratch/ \
        stoictrain
