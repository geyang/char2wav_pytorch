import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class RecursiveDilation(nn.Module):
    def __init__(self):
        super(RecursiveDilation, self).__init__()
