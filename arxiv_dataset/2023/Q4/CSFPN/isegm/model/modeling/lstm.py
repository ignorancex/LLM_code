import os
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, use_bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        use_bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=hidden_dim,
                                            kernel_size=kernel_size, padding=padding, bias=use_bias),
                                  nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                                            kernel_size=1, padding=0, bias=self.bias),
                                  nn.Conv2d(in_channels=hidden_dim, out_channels=4 * hidden_dim,
                                            kernel_size=kernel_size, padding=padding, bias=use_bias))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, use_bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim

        # self.gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
        #                        out_channels=2 * hidden_dim,  # for update_gate,reset_gate respectively
        #                        kernel_size=kernel_size,
        #                        padding=padding,
        #                        bias=use_bias)
        latten_dim = (input_dim + hidden_dim) // 4
        self.gates = nn.Sequential(nn.Conv2d(input_dim + hidden_dim, latten_dim, kernel_size=1,
                                             bias=use_bias),
                                   nn.Conv2d(latten_dim, latten_dim, kernel_size=kernel_size,
                                             padding=padding, bias=use_bias),
                                   nn.Conv2d(latten_dim, 2 * hidden_dim, kernel_size=1,
                                             bias=use_bias),
                                   nn.ReLU()
                                   )

        # self.out_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
        #                            out_channels=hidden_dim,  # for candidate neural memory
        #                            kernel_size=kernel_size,
        #                            padding=padding,
        #                            bias=use_bias)
        self.out_gates = nn.Sequential(nn.Conv2d(input_dim + hidden_dim, latten_dim, kernel_size=1,
                                                 bias=use_bias),
                                       nn.Conv2d(latten_dim, latten_dim, kernel_size=kernel_size,
                                                 padding=padding, bias=use_bias),
                                       nn.Conv2d(latten_dim, hidden_dim, kernel_size=1,
                                                 bias=use_bias),
                                       nn.ReLU()
                                       )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = torch.sigmoid(self.gates(combined))

        reset_gate, update_gate = torch.split(combined_conv, self.hidden_dim, dim=1)

        hhat = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        hhat = torch.tanh(self.out_gates(hhat))

        h_next = (1 - update_gate) * h_cur + update_gate * hhat
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, use_bias=False, return_all_layers=False, device=None):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param batch_first: bool
            if the first position of array is batch or not
        :param use_bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         use_bias=use_bias))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """

        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=input_tensor.size(0))

        last_state_list = [input_tensor]
        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            # input current hidden and cell state then compute the next hidden and cell state
            # through ConvLSTMCell forward function
            h = self.cell_list[layer_idx](input_tensor=last_state_list[-1], h_cur=h)
            last_state_list.append(h)

        return last_state_list[1:]

    def init_hidden(self, batch_size, device=None):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    height = width = 6
    channels = 256
    hidden_dim = [32, 64]
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 2  # number of stacked hidden layer
    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    use_bias=True,
                    return_all_layers=False)

    batch_size = 1
    time_steps = 2
    input_tensor = torch.rand(batch_size, channels, height, width)  # (b,t,c,h,w)
    last_state_list = model(input_tensor)
    print(last_state_list[0].shape, last_state_list[1].shape)

    last_state_list = model(input_tensor, last_state_list)
    print(last_state_list[0].shape, last_state_list[1].shape)