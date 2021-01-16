#!/bin/bash

let PORT=$1

# GPUS per node
let LOCAL_SIZE=$DLWS_NUM_GPU_PER_WORKER

# World size (total GPUs number)
let WORLD_SIZE=$DLWS_WORKER_NUM*$DLWS_NUM_GPU_PER_WORKER

# Master node ip (here we use node 0's ip)
MASTER_IP=$DLWS_SD_worker0_IP

echo 'WORLD_SIZE:'$WORLD_SIZE
echo 'LOCAL_SIZE:'$LOCAL_SIZE
echo

FILE_ROOT=$(cd "$(dirname "$0")"; pwd) 

for (( i=0; i<$LOCAL_SIZE; i++ ))
do
    # Global rank
    let RANK=DLWS_ROLE_IDX*LOCAL_SIZE+i

    python $FILE_ROOT/toy.py --rank ${RANK} --world-size ${WORLD_SIZE} --ip ${MASTER_IP} --port $PORT &
done
