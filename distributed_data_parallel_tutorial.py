import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp
import argparse
import random
import numpy as np
import time

def setup(rank, world_size):
# initial distributed group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def cuda_set_device(rank):
# set cuda device of current proccess
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return device

def set_seed(seed):
# control random numbers in multiple processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

        print('[INFO] model initialized')

    def forward(self, x):
        return self.customize_method(x)
    
    def customize_method(self, x):
    # test whether a customized method can be used
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        x = x * 2
        return x
    
def main(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    device = cuda_set_device(rank)

    # random test
    ## In each process you will see the same generated random numbers, if comment the set_seed(), random numbers will be different

    set_seed(0)
    rand_num = torch.randint(0, 10, (5,))
    print(f"[Rank{rank}] Generated random number: {rand_num}")

    # training data preparation
    training_data = torch.randn((2000, 10), dtype=torch.float32, device=device)
    print(f"[Rank{rank}] trining_data is in device: {training_data.device}")
    true_transform = torch.tensor(list(range(50)), dtype=torch.float32, device=device).reshape(-1, 5)
    label_data = training_data.matmul(true_transform)

    model = ToyModel().to(device)
    ddp_model = ddp(model, device_ids=[device])

    epochs = args.eps
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.0001) # use a extremely small learning rate to test training efficiency between single GPU and multi-GPU

    start_t = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = ddp_model(training_data)
        loss = loss_fn(outputs, label_data)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0 and rank == 0:
            end_t = time.time()
            print(f'[INFO] epoch {epoch + 1} takes {(end_t - start_t):.4f} seconds.: loss = {loss}')
            start_t = end_t

    # evaluation
    if rank == 0:
        eval_tensor = torch.ones((1, 10), dtype=torch.float32, device=device)
        output = model(eval_tensor)
        print(f"[INFO] true results: {eval_tensor.matmul(true_transform)}")
        print(f"[INFO] eval results: {output}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--eps', default=1000, type=int)
    opt=parser.parse_args()
    
    world_size = opt.gpu_num

    mp.spawn(main, args=(world_size, opt), nprocs=world_size)