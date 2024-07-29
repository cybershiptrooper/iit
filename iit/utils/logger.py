import sys
import os
import time

import numpy as np
import torch as t
from torch import Tensor
np.set_printoptions(threshold=sys.maxsize)

class LoggingDict(dict):
    def __init__(self, *args, **kwargs): #type: ignore
        dirname = "logs"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._log_filename = os.path.join(dirname, f"log_{time.strftime('%m-%d_%H-%M')}.log")
        
        super().__init__(*args, **kwargs)

    def compare(self, x: t.Any, y: t.Any) -> bool:
        if isinstance(x, (Tensor)):
            assert isinstance(y, (Tensor)), "x and y are not the same type"
            return bool((x == y).all())
        elif isinstance(x, (np.ndarray)):
            assert isinstance(y, (np.ndarray)), "x and y are not the same type"
            return bool((x == y).all())
        elif isinstance(x, (list)):
            assert isinstance(y, (list)), "x and y are not the same type"
            return bool(all(self.compare(x[i], y[i]) for i in range(len(x))))
        else:
            return x == y
    
    def convert_tensor_to_numpy(self, x: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, (Tensor)):
            return x.cpu().detach().numpy()
        return x
    
    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        if key not in self:
            with open(self._log_filename, "a") as f:
                f.write(f"{key}\n initial value: {value}\n")
        elif not self.compare(self[key], value):
            with open(self._log_filename, "a") as f:
                f.write(f"{key}\n changed from {self[key]} to {value}\n")
        super().__setitem__(key, value)


if __name__ == "__main__":
    logger = LoggingDict() # type: ignore
    logger["a"] = 1
    logger["b"] = 2
    logger["c"] = 3
    logger["a"] = 4
    filename = logger._log_filename
    file = open(logger._log_filename, "r")
    print(file.read())