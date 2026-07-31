from contextlib import contextmanager

import numpy as np


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
