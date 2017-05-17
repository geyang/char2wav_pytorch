import torch
import torch.nn as nn
from torch.autograd import Variable


# NOTE: Considering removing this class, since the only thing it does is to add up-sampling.
# done: add up-sampling
# todo: add multiple layers
class PerforatedRNN(nn.Module):
    """This fun little module up-samples its output by `k` in a way similar to *perforated* up-sampling done in CNN.
    NOTE:
        - Does NOT support multiple layers.
        - Does NOT support batch_first, always assumes seq_length to be first index.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        k: the up-sampling within each layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, h_0
        - **input** (seq_len, batch, input_size): **Same as RNN**. tensor containing the features of the input sequence.
          The input can also be a packed variable length sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** (num_layers * num_directions, batch, hidden_size): **Same as RNN** tensor containing the initial
          hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor containing the output features h_t from
          the last layer of the RNN, for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the
          input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, output_size, k, bidirectional=False, activation=nn.ReLU):
        """k is the up-sampling factor."""
        super(PerforatedRNN, self).__init__()
        self.char_set = input_size
        self.hidden_size = output_size
        self.k = k
        self.n_layers = 1
        self.bi_multiplier = 2 if bidirectional else 1

        self.gru = nn.GRU(output_size, output_size, batch_first=True)
        self.activation = activation()

    def forward(self, x, hidden):
        """
        Perforated Up-sampling: add zeros in-between real outputs.
        
        NOTE: 
            - batch index goes first.
            - no support for `batch_first` yet.
        """
        output, hidden = self.gru(x, hidden)
        seq_len, batch_size, feature_n = output.size()
        # done: up-sampling
        output_perforated = Variable(
            torch.zeros(
                int(seq_len * self.k),
                batch_size,
                feature_n
            )
        )
        # NOTE: no support for sequence index advanced indexing.
        # output_perforated[::k] = output
        for i in range(seq_len):
            output_perforated[i*k] = output
        return output_perforated, hidden

    def init_hidden(self, batch_size, rand: bool = False):
        """remember, hidden layer always has batch_size at index = 1, regardless of the batch_first flag."""
        if rand:
            return Variable(
                torch.randn(
                    self.bi_multiplier * self.n_layers,
                    batch_size,
                    self.hidden_size))
        else:
            return Variable(
                torch.zeros(
                    self.bi_multiplier * self.n_layers,
                    batch_size,
                    self.hidden_size))


class SampleRNN(nn.Module):
    """The sample RNN module, built on-top of perforated RNN.
    params:
        - input_size
        - output_size (one-hot)
        - ks: an array of up-sampling
    input: 
        - 
    """

    def __init__(self, input_size, output_size, hidden_size, k, n_layers=1, bidirectional=False):
        super(SampleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # NOTE: accept both [k*] as well as k:<int>
        self.ks = k if type(k) is 'list' else [k] * self.n_layers
        assert (len(k) == n_layers)
        self.layers = []
        self.layers.append(PerforatedRNN(input_size, hidden_size))
        for i in range(1, self.n_layers - 1):
            self.layers.append(PerforatedRNN(hidden_size, hidden_size))
        if i < self.n_layers:
            self.layers.append(PerforatedRNN(hidden_size, output_size))

        self.bidirectional = bidirectional

    def init_hidden(self, batch_size, *args, **kwrags):
        """takes in the batch_size of the input."""
        return [layer.init_hidden(batch_size, *args, **kwrags) for layer in self.layers]

    # todo: add teacher_forcing to the wavelet output?
    def forward(self, input, hidden, target=None):
        """
        hidden is a list where each item is the hidden vector for each layer.
        ```
        hidden = [
            Size(seq_len, batch_size, hidden_size)
        ]
        ```
        We get these parameters from the layer array.
        """
        # NOTE: hidden always has second index being the batch index.

        x = input
        hidden_updated = []
        for ind, (layer, hidden) in enumerate(zip(self.layers, hidden if type(hidden) is 'list' else [hidden])):
            x, _hidden = self.layer.forward(x, hidden)
            hidden_updated.append(_hidden)
        return x, hidden_updated

    # todo: need to evaluate this method.
    def setup_training(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()

    def load(self, fn):
        # TODO: load the input and output language as well, to maintain the char list.
        checkpoint = torch.load(fn)
        self.load_state_dict(checkpoint['state_dict'])
        self.input_lang.load_dict(checkpoint['input_lang'])
        self.output_lang.load_dict(checkpoint['output_lang'])
        return checkpoint

    def save(self, fn="seq-to-seq.cp", meta=None, **kwargs):
        # TODO: save input and output language as well, to maintain the char list.
        d = {k: kwargs[k] for k in kwargs}
        d["state_dict"] = self.state_dict()
        d["input_lang"] = vars(self.input_lang)
        d["output_lang"] = vars(self.output_lang)
        if meta is not None:
            d['meta'] = meta
        torch.save(d, fn)
