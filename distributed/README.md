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
$ python toy.py --rank 0 --world-size 2 --ip 192.168.0.100 --port 22000
```

#### Node 1

```shell
$ python toy.py --rank 1 --world-size 2 --ip 192.168.0.100 --port 22000
```
