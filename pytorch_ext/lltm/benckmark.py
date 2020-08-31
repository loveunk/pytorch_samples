import time
import torch
from cpp.lltm import LLTM as cpp_lltm
from py import LLTM as py_lltm


def benckmark(lstm_func, description, device='cpu', steps=1000):
    batch_size = 16
    input_features = 32
    state_size = 128
    torch_device = torch.device(device)

    X = torch.randn(batch_size, input_features, device=torch_device)
    h = torch.randn(batch_size, state_size, device=torch_device)
    C = torch.randn(batch_size, state_size, device=torch_device)

    rnn = lstm_func(input_features, state_size).to(torch_device)

    forward = 0
    backward = 0
    for _ in range(steps):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        if device == 'cuda':
            torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        backward += time.time() - start

    print('{}, Device: {}, Forward: {:.3f} s | Backward {:.3f} s'.format(
        description, device, forward, backward))


if __name__ == "__main__":

    benckmark(cpp_lltm, 'C++', device='cuda', steps=10000)
    benckmark(py_lltm, 'Py ', device='cuda', steps=10000)

    benckmark(cpp_lltm, 'C++', device='cpu', steps=10000)
    benckmark(py_lltm, 'Py ', device='cpu', steps=10000)
