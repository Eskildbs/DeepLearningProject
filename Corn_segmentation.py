import data_util
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, MaxUnpool2d
import torch.nn.functional as F

class flatten(nn.Module):
    def forward(self, x):
        return(x.view(x.shape[0], -1))


class MRF_loss(torch.nn.Module):

    def __init__(self, lambda_val):
        super(MRF_loss, self).__init__()
        self.structure = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]])
        self.lambda_val = lambda_val

    def forward(self, y_hat, y):
        """
        :param y_hat:   Batch of estimated map of classes
        :param y:       Batch of true map of classes
        :return:
        """
        y_byte_inner = (y[:, :, 1:-1, 1:-1]).byte()
        y_byte = y.byte()
        neighbour_energy = y_byte_inner ^ y_byte[:, :, 2:, 2:] + y_byte_inner ^ y_byte[:, :, 1:-1, 2:] + \
                           y_byte_inner ^ y_byte[:, :, :-2, 2:] + y_byte_inner ^ y_byte[:, :, :-2, 1:-1] + \
                           y_byte_inner ^ y_byte[:, :, :-2, :-2] + y_byte_inner ^ y_byte[:, :, 1:-1, :-2] + \
                           y_byte_inner ^ y_byte[:, :, 2:, :-2] + y_byte_inner ^ y_byte[:, :, 2:, 1:-1]

        return ((y_hat - y) ** 2).sum() + (neighbour_energy.float().sum() * self.lambda_val)

class Net(nn.Module):
    def __init__(self, load_text = False):
        super(Net, self).__init__()
        self.number_of_classes = 3
        self.GPU_ENABLED = torch.cuda.is_available()

        if load_text != False:

            """ Loading network from job text file """

            layers = load_text[0]
            parameters = load_text[1]
            network = "self.number_of_classes=" + str(self.number_of_classes)
            network += "\nself.encoder_list=nn.Sequential(OrderedDict(["
            conv_cnt = 1
            pool_cnt = 1
            unpool_cnt = 1
            decoder = 0
            for i, layer in enumerate(layers):
                if layer == 1: #conv2d
                    network+='(\'conv' + str(conv_cnt) + '\', Conv2d(in_channels=' + str(parameters[i-decoder][0]) +\
                             ',out_channels=' + str(parameters[i-decoder][1]) + ',kernel_size=' + str(parameters[i-decoder][2]) +\
                             ',stride=' + str(parameters[i-decoder][3]) + ',padding=' + str(parameters[i-decoder][4]) + ')),'
                    conv_cnt +=1
                elif layer == 2: #pool
                    network+='(\'pool' + str(pool_cnt) + '\',MaxPool2d(kernel_size=' + str(parameters[i-decoder][0]) +\
                             ',stride=' + str(parameters[i-decoder][1]) + ',padding=' + str(parameters[i-decoder][2]) +\
                             ',return_indices=True)),'
                    pool_cnt += 1
                elif layer == 3: #unpool
                    network+='(\'unpool' + str(unpool_cnt) + '\', MaxUnpool2d(kernel_size=' + str(parameters[i-decoder][0]) +\
                             ',stride=' + str(parameters[i-decoder][1]) + ')),'

                    unpool_cnt += 1
                elif layer == 4:
                    network = network[:-1]
                    network += ']))\nself.decoder_list=nn.Sequential(OrderedDict(['
                    decoder = 1

            network = network[:-1]
            network += ']))\nself.output = nn.Softmax(dim=1)'
            exec(network)
        else:

            """ Default network load """

            self.encoder_list = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(in_channels=3,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv2', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('pool1', MaxPool2d(kernel_size=2,  # 256x256
                                    stride=2,
                                    padding=0,
                                    return_indices=True)),
                ('conv3', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv4', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('pool2', MaxPool2d(kernel_size=2,  # 128x128
                                    stride=2,
                                    padding=0,
                                    return_indices=True)),
                ('conv5', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv6', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv7', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('pool3', MaxPool2d(kernel_size=2,  # 64x64
                                    stride=2,
                                    padding=0,
                                    return_indices=True))
                ]))
            self.decoder_list = nn.Sequential(OrderedDict([
                ('unpool1', MaxUnpool2d(kernel_size=2,  # 128x128
                                      stride=2)),
                ('conv1', Conv2d(in_channels=8+8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv2', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv3', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('unpool2', MaxUnpool2d(kernel_size=2,  # 256x256
                                        stride=2)),
                ('conv4', Conv2d(in_channels=8+8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv5', Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('unpool3', MaxUnpool2d(kernel_size=2,  # 512x512
                                        stride=2)),
                ('conv6', Conv2d(in_channels=8+8,
                                 out_channels=8,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)),
                ('conv7', Conv2d(in_channels=8,
                                 out_channels=self.number_of_classes,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2))
            ]))

            """ Output """
            self.output = nn.Softmax(dim=1)

    def run_list(self, list, action):
        if action == 'encoder':
            self.indices_input = []
            self.encoder_features = []
        i = 1
        j = 1
        for cnt, op in enumerate(list):
            str_op = str(op).split('(')[0]
            if str_op == 'MaxPool2d':
                self.input, index = op(self.input)
                self.indices_input.append(index)
            elif str_op == 'MaxUnpool2d':
                self.input = op(self.input, indices=self.indices_input[-i])
                i += 1
            else:
                if action == 'decoder': # decoderpart
                    # if element is the first element after a upooling, previous features should be used.
                    # Does not check first element, due to index error
                    if cnt != 0 and str(list[cnt-1]).split('(')[0] == 'MaxUnpool2d':
                        self.input = torch.cat((self.input, self.encoder_features[-j]), dim=1)
                        if self.GPU_ENABLED:
                            self.input = self.input.cuda()
                        j+=1
                    self.input = op(self.input)
                else: # encoder part
                    # if last element in encoder part or the conv layer just before pooling, save its features.
                    self.input = op(self.input)
                    if ((cnt+1 == len(list)) or str(list[cnt+1]).split('(')[0] == 'MaxPool2d'):
                        self.encoder_features.append(self.input)

    def forward(self, image):
        # permuting image, such that [batch_size, channels, height, width]
        if image.ndim == 4:
            self.input = image.permute(0, 3, 1, 2)
        elif image.ndim == 3:
            self.input = image.permute(2, 0, 1)

        self.run_list(list=self.encoder_list, action='encoder')
        self.run_list(list=self.decoder_list, action='decoder')
        y_out = self.output(self.input)
        return y_out

    def output_conv_parameters(self, output_path, sampling='average'):
        # creaing directory
        if not os.path.isdir(output_path + 'model'):
            os.mkdir(output_path + 'model')
        if not os.path.isdir(output_path + 'model/conv_filters'):
            os.mkdir(output_path + 'model/conv_filters')

        # filtering conv modules
        conv_modules = dict()
        for i, (key, value) in enumerate(self._modules.items()):
            if (i+1) != len(self._modules):
                conv_modules[key] = value

        for seq_number, seq in enumerate(conv_modules): # going through conv modules
            conv_layers = list(filter(lambda elm: isinstance(elm, Conv2d), conv_modules[seq])) # filtering conv operations
            for conv_number, conv in enumerate(conv_layers): # going through conv layers

                # extracting bias' and weights
                bias = conv._parameters['bias'].data.cpu().numpy()
                weight = conv._parameters['weight'].data.cpu().numpy()

                # calculating values fro grid
                if sampling == 'average':
                    weight = np.mean(weight, axis=1)
                    values = weight

                # plotting filters
                n_filters = len(values[:, 0, 0])
                sqrt_n = int(np.ceil(np.sqrt(n_filters)))
                fig, axs = plt.subplots(sqrt_n, sqrt_n)
                cnt = 0
                for i in range(sqrt_n):
                    for j in range(sqrt_n):
                        if cnt >= n_filters: # removes empty subplots
                            axs[i, j].set_axis_off()
                        else:
                            axs[i, j].matshow(values[cnt, :, :], cmap=plt.get_cmap('Greys'))
                            cnt += 1

                # saving filters as files
                fig.suptitle('Seq_' + str(seq_number) + '_convLayer_' + str(conv_number), y=0)
                fig.savefig(output_path + 'model/conv_filters/filter_conv_' + str(seq_number) + '_' + str(conv_number))

class Stopping_criteria():
    """
    Stopping criterias return True, if network should stop training due to the convergence criteria.
    It should return False otherwise.
    """
    def __init__(self):
        pass

    def test_for_stop(self, job):
        self.training_error = job.training_loss
        self.training_accuracy = job.training_accs
        self.validation_error = job.validation_loss
        self.validation_accuracy = job.validation_accs

class Moving_average_stopping(Stopping_criteria):
    def __init__(self, lag=35):
        super().__init__()
        self.lag = lag

    def test_for_stop(self, job):
        super().test_for_stop(job)
        if len(self.validation_error) > self.lag:
            self.moving_average = np.mean(self.validation_error[-self.lag-5:-5])
            self.last_error = np.mean(self.validation_error[-5:])
            if self.last_error > self.moving_average:
                print('Finished job due to stopping criteria.')
                return True
        else:
            return False

class No_stopping_criteria(Stopping_criteria):
    def __init__(self):
        super().__init__()

    def test_for_stop(self, job):
        super().test_for_stop(job)
        return False

def get_accuracy(output, target):
    predictions = torch.max(output, 1)[1]
    target = torch.max(target, 1)[1]
    correct_prediction = torch.eq(predictions, target)
    return torch.mean(correct_prediction.float()).cpu().numpy()