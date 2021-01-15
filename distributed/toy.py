import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint

def foo(rank, world_size):
    for step in range(20):
        # get random int
        value = randint(0, 10)
        # group all ranks
        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)
        # compute reduced sum

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor = torch.IntTensor([value]).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(
              rank,step,value,float(tensor)))
        sleep(1)

def initialize(backend, rank, world_size, ip, port):
    dist.init_process_group(backend=backend,
                            init_method='tcp://{}:{}'.format(ip, port),
                            rank=rank,
                            world_size=world_size)
    foo(rank, world_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='nccl', choices=['gloo', 'nccl'])
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--world-size', '-s', type=int)
    args = parser.parse_args()
    print(args)
    initialize(args.backend, args.rank, args.world_size, args.ip, args.port)


'''
python toy.py --rank ${DLWS_ROLE_IDX} --world-size ${DLWS_WORKER_NUM} --ip ${DLWS_SD_worker0_IP} --port 22000
'''
if __name__ == '__main__':
    main()
