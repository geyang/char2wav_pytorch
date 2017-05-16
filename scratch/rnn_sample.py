import torch
import torch.nn as nn
from torch.autograd import Variable

# test script
rnn = nn.GRU(input_size=10, hidden_size=200, num_layers=2, bidirectional=True)

x0 = Variable(torch.randn(15, 100, 10))
h0 = Variable(torch.randn(2 * 200, 100, 200))
x, h = rnn(x0, h0)
