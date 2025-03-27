# adjusted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
# and from https://github.com/jimmyyhwu/resnet18-tf2/blob/master/resnet.py
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model, Input, Sequential

# resnet_kernel_initializer = keras.initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform')

class BasicBlock(Layer):
    expansion = 1
    def __init__(self, depth, input_depth, strides=1):
        super().__init__()
        self.conv_1 = Conv2D(depth, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(depth, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn_2 = BatchNormalization()
        self.shortcut = Sequential()
        if strides != 1 or input_depth != self.expansion*depth:
            self.shortcut.add(
                Conv2D(self.expansion*depth, kernel_size=1, strides=strides, padding="same", use_bias=False))
            self.shortcut.add(BatchNormalization())
        self.config = {
            "depth":depth,
            "input_depth":input_depth,
            "strides":strides,
            "expansion":self.expansion,
        }
    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config
    def call(self, x):
        temp_x = tf.nn.relu(self.bn_1(self.conv_1(x)))
        temp_x = self.bn_2(self.conv_2(temp_x))
        temp_x += self.shortcut(x)
        out = tf.nn.relu(temp_x)
        return out

class Bottleneck(Layer):
    expansion = 4
    def __init__(self, depth, input_depth, strides=1):
        super().__init__()
        self.conv_1 = Conv2D(depth, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(depth, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn_2 = BatchNormalization()
        self.conv_3 = Conv2D(depth*self.expansion, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.bn_3 = BatchNormalization()
        self.shortcut = Sequential()
        if strides != 1 or input_depth != self.expansion*depth:
            self.shortcut.add(
                Conv2D(self.expansion*depth, kernel_size=1, strides=strides, padding="same", use_bias=False))
            self.shortcut.add(BatchNormalization())
        self.config = {
            "depth":depth,
            "input_depth":input_depth,
            "strides":strides,
            "expansion":self.expansion,
        }
    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config
    def call(self, x):
        temp_x = tf.nn.relu(self.bn_1(self.conv_1(x)))
        temp_x = tf.nn.relu(self.bn_2(self.conv_2(temp_x)))
        temp_x = self.bn_3(self.conv_3(temp_x))
        temp_x += self.shortcut(x)
        out = tf.nn.relu(temp_x)
        return out

class ResNet(Model):
    '''
    The model is different to the origin resnet.
    To fit cifar10, the kernel_size of the start layer is 3 rather than 7, 
    there is no max_pool after the start layer,
    and the depth is 32 rather than 64
    The output is feature after avgpooling
    '''
    def __init__(self, block, num_blocks_list):
        super().__init__()
        start_depth = 64
        self.start_conv = Conv2D(start_depth, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.start_bn = BatchNormalization()
        self.layer1 = self._make_layer(block, 64, start_depth, num_blocks_list[0], strides=1)
        self.layer2 = self._make_layer(block, 128, 64*block.expansion, num_blocks_list[1], strides=2)
        self.layer3 = self._make_layer(block, 256, 128*block.expansion, num_blocks_list[2], strides=2)
        self.layer4 = self._make_layer(block, 512, 256*block.expansion, num_blocks_list[3], strides=2)
        self.config = {
            # "block":block,
            "num_blocks_list":num_blocks_list,
        }
    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config
        # return self.config
    def _make_layer(self, block, depth, start_depth, num_blocks, strides):
        temp_layer = Sequential()
        strides_list = [1 for i in range(num_blocks)]
        strides_list[0] = strides
        input_depth = start_depth
        for now_strides in strides_list:
            temp_layer.add(block(depth, input_depth, strides=now_strides))
            input_depth = depth*block.expansion
        return temp_layer
    
    def call(self, x):
        temp_x = tf.nn.relu(self.start_bn(self.start_conv(x)))
        temp_x = self.layer1(temp_x)
        temp_x = self.layer2(temp_x)
        temp_x = self.layer3(temp_x)
        out = self.layer4(temp_x)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
# kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

# def conv3x3(x, out_planes, stride=1, name=None):
#     x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
#     return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

# def basic_block(x, planes, stride=1, downsample=None, name=None):
#     identity = x

#     out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
#     out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
#     out = layers.ReLU(name=f'{name}.relu1')(out)

#     out = conv3x3(out, planes, name=f'{name}.conv2')
#     out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

#     if downsample is not None:
#         for layer in downsample:
#             identity = layer(identity)

#     out = layers.Add(name=f'{name}.add')([identity, out])
#     out = layers.ReLU(name=f'{name}.relu2')(out)

#     return out

# def make_layer(x, planes, blocks, stride=1, name=None):
#     downsample = None
#     inplanes = x.shape[3]
#     if stride != 1 or inplanes != planes:
#         downsample = [
#             layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
#             layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
#         ]

#     x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
#     for i in range(1, blocks):
#         x = basic_block(x, planes, name=f'{name}.{i}')

#     return x

# def resnet(x, blocks_per_layer, num_classes=1000):
#     x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
#     x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
#     x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
#     x = layers.ReLU(name='relu1')(x)
#     x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
#     x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

#     x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
#     x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
#     x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
#     x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

#     feature = layers.GlobalAveragePooling2D(name='avgpool')(x)
#     initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
#     y = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(feature)

#     return y, feature

# def resnet18(x, **kwargs):
#     return resnet(x, [2, 2, 2, 2], **kwargs)

# def resnet34(x, **kwargs):
#     return resnet(x, [3, 4, 6, 3], **kwargs)
if __name__ == "__main__":
    test = Bottleneck(32, 32)
    print(test.get_config())