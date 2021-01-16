import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint

def foo(local_rank):
    for step in range(20):
        # get random int
        value = randint(0, 10)
        # group all ranks
        #ranks = list(range(world_size))
        #group = dist.new_group(ranks=ranks)
        # compute reduced sum

        device = torch.device("cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu")
        tensor = torch.IntTensor([value]).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)#, group=group)
        print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(
              dist.get_rank(), step, value, float(tensor)))
        sleep(1)

def initialize(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')

    foo(local_rank)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='nccl', choices=['gloo', 'nccl'])
    parser.add_argument('--local_rank', '-r', type=int)
    args = parser.parse_args()
    print(args)
    initialize(args.local_rank)


'''
python -m torch.distributed.launch \
        --nproc_per_node=${DLWS_NUM_GPU_PER_WORKER} \
        --nnodes=${DLWS_WORKER_NUM} \
        --node_rank=${DLWS_ROLE_IDX} \
        --master_addr="${DLWS_SD_worker0_IP}" \
        --master_port=22000 toy_torch_dist_launch.py
'''
if __name__ == '__main__':
    main()
