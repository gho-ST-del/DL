import torch
import torchvision.transforms as trasforms
import torch.autograd as autograd
import torch.utils.data as Data
from matplotlib import pyplot as plt
import numpy as np
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features  =torch.tensor(np.random.normal((0,1,(num_examples,num_inputs),)),dtype=torch.float32)
Data.DataLoader