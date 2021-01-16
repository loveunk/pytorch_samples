# 分布式例子

## toy.py with `torch.distributed`

这是一个简单的测试`Pytorch`分布式的Python例子，并没有使用到模型训练。

### 单机 共2卡

```shell
Terminal 1
$ python toy.py --rank 0 --world-size 2

Terminal 2
$ python toy.py --rank 1 --world-size 2
```

### 两机 共2卡

master node ip `192.168.0.100`

#### Node 0

```shell
$ python toy.py --rank 0 --world-size 2 --ip 192.168.0.100 --port 22222
```

#### Node 1

```shell
$ python toy.py --rank 1 --world-size 2 --ip 192.168.0.100 --port 22222
```

### 两机 共8卡

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

#### Node 0

```shell
$ multi-nodes-multi-gpus.sh 22222
```

#### Node 1

```shell
$ multi-nodes-multi-gpus.sh 22222
```
