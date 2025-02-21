import torch
import torch.nn as nn

class Convolutional_neuralnet(nn.Module):
  def __init__(self, dataloader):
    super(Convolutional_neuralnet, self).__init__()
    # this automatically calculates the correct shape for the data (only works for square images)
    # such that the data can automatically be converted into the right shape
    # (batch_size, depth, width, height) 
    # and number of classes (nodes in output channel)
    sample_batch = next(iter(dataloader))
    images, labels = sample_batch
    self.num_output_classes = labels.unique().numel()
    self.input_shape = int(torch.sqrt(torch.tensor(images.size(1), dtype=torch.float32)))

    # input channels should be equal to the amount of color channels in the imaage
    # for mnist, this is 1 because it is greyscale
    # for cifar-10, it should be 3 for the red, green and blue channels
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    # to calculate the amount of nodes for the first layer of the neural net
    # we take the original size (28 x 28) divided by the stride of the first pooling layer = 14 * 14
    # the 14 * 14 output is again divided by 2, because of the stride of 2 in the second pooling layer = 7 * 7
    # to calculate the amount of nodes, this has to be multiplied by the depth of the last convolutional layer
    # with a depth of 64, there would be 7 * 7 * 64 = 3136 nodes
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, self.num_output_classes)

  def forward(self, x):
    # -1 is the batch size, which is automatically done based on the data
    # 1 stands for the depth, which is 1 for grey-scaled
    x = x.view(-1, 1, self.input_shape, self.input_shape)
    x = self.relu(self.conv1(x))
    x = self.pool1(x)

    x = self.relu(self.conv2(x))
    x = self.pool2(x)

    # flatten the outputs to be able to go into the fc layer(s)
    # it should be flattened the same dimensions as the amount of nodes in the fc layer (like explained above)
    x = x.view(-1, 64 * 7 * 7)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x