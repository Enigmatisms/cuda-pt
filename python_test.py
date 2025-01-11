import sys
import torch

sys.path.append("./build/Release")
from pyrender import PythonRenderer

if __name__ == "__main__":
    rdr = PythonRenderer("./scene/xml/vader.xml", 0)

    tensor: torch.Tensor = rdr.render()
    print(tensor.shape, tensor.dtype, tensor.device)