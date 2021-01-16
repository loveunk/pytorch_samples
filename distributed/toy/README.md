# 分布式例子

## toy with `torch.distributed`

这是一个简单的测试`Pytorch`分布式的Python例子，并没有使用到模型训练。

### 单机 共2卡

```shell
Terminal 1
$ python toy.py --rank 0 --local_rank 0 --world-size 2

Terminal 2
$ python toy.py --rank 1 --local_rank 1 --world-size 2
```

### 两机 共2卡

例如 master node ip: `192.168.0.100`

#### Node 0

```shell
$ python toy.py --rank 0 --local_rank 0 --world-size 2 --ip 192.168.0.100 --port 22000
```

#### Node 1

```shell
$ python toy.py --rank 1 --local_rank 0 --world-size 2 --ip 192.168.0.100 --port 22000
```

### 两机 共8卡

有两种形式：

#### 方法1: 在每个node上luanch 4个进程

为了方便操作，使用一个脚本`multi-nodes-multi-gpus.sh` 来处理需要在每个`Node`上拉起 `LOCAL_RANK` 个processes。

注意，需要这里从环境变量里获取了三个值，你也可以自己修改脚本。

```shell
# GPUS per node
let LOCAL_SIZE=$DLWS_NUM_GPU_PER_WORKER

# World size (total GPUs number)
let WORLD_SIZE=$DLWS_WORKER_NUM*$DLWS_NUM_GPU_PER_WORKER

# Master node ip (here we use node 0's ip)
MASTER_IP=$DLWS_SD_worker0_IP
```

执行：

Node 0

```shell
$ multi-nodes-multi-gpus.sh 22000
```

Node 1

```shell
$ multi-nodes-multi-gpus.sh 22000
```

#### 方法2: 使用torch.distributed.launch在每个node拉起一个进程

在每个node上执行：

```shell
python -m torch.distributed.launch \
        --nproc_per_node=${DLWS_NUM_GPU_PER_WORKER} \
        --nnodes=${DLWS_WORKER_NUM} \
        --node_rank=${DLWS_ROLE_IDX} \
        --master_addr="${DLWS_SD_worker0_IP}" \
        --master_port=22000 toy_torch_dist_launch.py
```

## MNIST with `torch.distributed`

在每个node上执行：

```shell
python ~/code/pytorch_samples/distributed/MNIST/mnist-dist.py \
                --init-method tcp://${DLWS_SD_worker0_IP}:22000 \
                --rank ${DLWS_ROLE_IDX} \
                --backend nccl \
                --world-size ${DLWS_WORKER_NUM} \
                --data_root data-${DLWS_ROLE_IDX}
```
