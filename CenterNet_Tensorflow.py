from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.models import Model, load_model, save_model
from keras.utils import plot_model
import tensorflow as tf
import tensorflow.python.keras.layers as tks


def relu(x):
    return tks.Activation(activation='relu')(x)


class CenterNet:
    def __init__(self, data_format, is_training, num_classes):
        self.data_format = data_format
        self.is_training = is_training
        self.num_classes = num_classes
        self.img_input_shape = [None, None, 3]
        self.inputs = []
        self.outputs = []

    def get_model(self):
        img_input = tks.Input(shape=self.img_input_shape)
        self.inputs.append(img_input)
        with tf.variable_scope('backone'):
            conv = self._conv_bn_activation(
                bottom=img_input,
                filters=16,
                kernel_size=7,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=16,
                kernel_size=3,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=32,
                kernel_size=3,
                strides=2,
            )
            dla_stage3 = self._dla_generator(conv, 64, 1, self._basic_block)
            dla_stage3 = self._max_pooling(dla_stage3, 2, 2)

            dla_stage4 = self._dla_generator(dla_stage3, 128, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage3, 128, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage4 = self._max_pooling(dla_stage4, 2, 2)
            dla_stage4 = tks.Add()([dla_stage4, residual])

            dla_stage5 = self._dla_generator(dla_stage4, 256, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage5 = self._max_pooling(dla_stage5, 2, 2)
            dla_stage5 = tks.Add()([dla_stage5, residual])

            dla_stage6 = self._dla_generator(dla_stage5, 512, 1, self._basic_block)
            residual = self._conv_bn_activation(dla_stage5, 512, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage6 = self._max_pooling(dla_stage6, 2, 2)
            dla_stage6 = tks.Add()([dla_stage6, residual])
        with tf.variable_scope('upsampling'):
            dla_stage6 = self._conv_bn_activation(dla_stage6, 256, 1, 1)
            dla_stage6_5 = self._dconv_bn_activation(dla_stage6, 256, 4, 2)
            dla_stage6_4 = self._dconv_bn_activation(dla_stage6_5, 256, 4, 2)
            dla_stage6_3 = self._dconv_bn_activation(dla_stage6_4, 256, 4, 2)

            dla_stage5 = self._conv_bn_activation(dla_stage5, 256, 1, 1)
            dla_stage5_4 = self._conv_bn_activation(tks.Add()([dla_stage5, dla_stage6_5]), 256, 3,
                                                    1)
            dla_stage5_4 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)
            dla_stage5_3 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)

            dla_stage4 = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            dla_stage4_3 = self._conv_bn_activation(
                tks.Add()([dla_stage4, dla_stage5_4, dla_stage6_4]), 256, 3, 1)
            dla_stage4_3 = self._dconv_bn_activation(dla_stage4_3, 256, 4, 2)

            features = self._conv_bn_activation(
                tks.Add()([dla_stage6_3, dla_stage5_3, dla_stage4_3]), 256, 3,
                1)
            features = self._conv_bn_activation(features, 256, 1, 1)

        with tf.variable_scope('center_detector'):
            keypoints = self._conv_bn_activation(features, self.num_classes, 3, 1, None)
            offset = self._conv_bn_activation(features, 2, 3, 1, None)
            size = self._conv_bn_activation(features, 2, 3, 1, None)
        self.outputs.append(keypoints)
        self.outputs.append(offset)
        self.outputs.append(size)
        model = Model(inputs=self.inputs, outputs=self.outputs)
        return model

    def _bn(self, bottom):
        bn = tks.BatchNormalization(
            axis=3 if self.data_format == 'channels_last' else 1,
        )(inputs=bottom, training=self.is_training)
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=relu):
        conv = tks.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=relu):
        conv = tks.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _basic_block(self, bottom, filters):
        conv = self._conv_bn_activation(bottom, filters, 3, 1)
        conv = self._conv_bn_activation(conv, filters, 3, 1)
        axis = 3 if self.data_format == 'channels_last' else 1
        input_channels = bottom.get_shape().as_list()[axis]
        if input_channels == filters:
            short_cut = bottom
        else:
            short_cut = self._conv_bn_activation(bottom, filters, 1, 1)
        return tks.Add()([conv, short_cut])

    def _dla_generator(self, bottom, filters, levels, stack_block_fn):
        if levels == 1:
            block1 = stack_block_fn(bottom, filters)
            block2 = stack_block_fn(block1, filters)
            aggregation = tks.Add()([block1, block2])
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        else:
            block1 = self._dla_generator(bottom, filters, levels - 1, stack_block_fn)
            block2 = self._dla_generator(block1, filters, levels - 1, stack_block_fn)
            aggregation = tks.Add()([block1, block2])
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        return aggregation

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tks.MaxPool2D(
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )(bottom)

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
        return tks.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )(bottom)


if __name__ == "__main__":
    dla = CenterNet(data_format="channels_last", is_training=True, num_classes=20)
    model = dla.get_model()
    plot_model(model, to_file='model1.png', show_shapes=True)
