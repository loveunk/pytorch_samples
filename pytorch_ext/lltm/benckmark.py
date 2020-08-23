import time
import torch
from cpp.lltm import LLTM as cpp_lltm
from py.lltm import LLTM as py_lltm


def benckmark(lstm_func, description, steps=1000):
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    rnn = lstm_func(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(steps):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start

    print('{}, Forward: {:.3f} s | Backward {:.3f} s'.format(
        description, forward, backward))


benckmark(cpp_lltm, 'C++', steps=10000)
benckmark(py_lltm, 'Py ', steps=10000)
