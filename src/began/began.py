import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class NoValidValueException(Exception):
    def __init__(self, name, valid_value):
        self.name = str(name)
        self.valid_value = str(valid_value)
    def __str__(self):
        msg = "Training with '%s=%s'"
        return repr(msg % (self.name, self.valid_value))

tf.reset_default_graph()


class NeuralNetworkModel(object):
    def __init__(self, *args, **kwargs):
        super().__init__()


    def input(self, *args, **kwargs):
        pass


    def _layer_rnn(self, *args, **kwargs):
        pass

    def _graph_rnn(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        pass


    def train(self, *args, **kwargs):
        pass


    def evaluate(self, *args, **kwargs):
        pass


    def loss_plot(self, *args, **kwargs):
        pass


# Functions in module --------------------------------------------------------



def get_conv_output_size(input_size, filter_size, stride_size):
    "Assumption: Input & Filter are Square-shaped."
    output_size = ((input_size - filter_size) / stride_size) + 1
    return output_size


# Model Classes --------------------------------------------------------------
class BEGAN(NeuralNetworkModel):
    """Varitional RNN.

    TBD

    Attributes
    ----------
    TBD


    Example
    -------
    >>> "TBD"

    """
    def __init__(
        self,
        input_x_ext='png',
        input_x_dtype=tf.float32,
        input_z_dtype=tf.float32,
        input_x_shape=(None, 64, 64, 3),
        input_z_shape=(None, 100),
        use_gpu=True,
        # input_width=12,
        # input_height=12,
        # input_channel=21,
        # output_width=None,
        # output_height=None,
        # output_channel=None,
        # class_num=None,
        # z_dim=100,
        filter_dim=64,
        # g_filter_dim=64,
        # d_filter_dim=64,
        g_fc_dim=1024,
        d_fc_dim=1024,
        batch_size=128,
        batch_norm_ok=False,
        conv_activation=tf.nn.elu,
        dropout=.3,
        lr_decaying=True,
        decay_lr_step=100_000,
        buffer_size=1000,
        learning_rate=.0002,
        adam_beta1=.8,
        adam_beta2=.8,
        validation_ratio=.2,
        ano_lambda_=.1,
        ):

        super().__init__()
        self.USE_GPU = use_gpu

        assert input_x_ext in ['jpg', 'jpeg', 'png']
        self.input_x_ext = input_x_ext

        self.input_x_dtype = input_x_dtype
        self.input_z_dtype = input_z_dtype

        self.input_x_shape = input_x_shape
        self.input_z_shape = input_z_shape

        (_,
         input_width,
         input_height,
         input_channel) = input_x_shape

        (_,
         output_width,
         output_height,
         output_channel) = input_x_shape

        _, self.Z_DIM = self.input_z_shape        # Random Variable Z Dimension

        assert input_width == input_height        # Check INPUT_WIDTH = INPUT_HEIGHT

        self.INPUT_WIDTH = input_width            # 5mins x 12 = 60mins = 1hour
        self.INPUT_HEIGHT = input_height          # 1hour x 12 = 12hours
        self.INPUT_CHANNEL = input_channel        # Each pixel has KPIs as channels

        self.CLASS_NUM = None

        self.OUTPUT_WIDTH = output_width
        self.OUTPUT_HEIGHT = output_height
        self.OUTPUT_CHANNEL = output_channel

        g_filter_dim = d_filter_dim = filter_dim
        assert g_filter_dim == d_filter_dim
        assert g_fc_dim == d_fc_dim
        self.G_FILTER_DIM = g_filter_dim
        self.D_FILTER_DIM = d_filter_dim
        self.G_FC_DIM = g_fc_dim
        self.D_FC_DIM = d_fc_dim
        self.FILTER_DIM = filter_dim

        self.BATCH_SIZE = batch_size              # Training Batch Size
        self.BATCH_NORM_OK = batch_norm_ok
        self.CONV_ACTIVATION = conv_activation
        self.DROPOUT = dropout
        self.LR_DECAYING = lr_decaying
        self.DECAY_LR_STEP = decay_lr_step
        self.BUFFER_SIZE = buffer_size            # For tf.Dataset.suffle(buffer_size)
        self.LEARNING_RATE = learning_rate        # Learning rate (Fixed for now)
        self.ADAM_BETA1 = adam_beta1
        self.ADAM_BETA2 = adam_beta2
        self.VALIDATION_RATIO = validation_ratio  # Validation Ratio
        self.EPOCH_NUM = 0                        # Cumulative Epoch Number

        self.ANO_LAMBDA_ = ano_lambda_

        self._x_input_tensor = None
        self._z_input_tensor = None
        self._x_batch_tensor = None
        self._z_batch_tensor = None
        self.data_init_op = None

        self.writer = None
        self.variable_init_op = None
        self.train_op = None
        self.train_loss_history = None
        self.train_valid_history = None

        self.prediction = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        self.metrics_init_op = None
        self.metrics = None
        self.update_metrics_op = None

        self.summary_op = None

        self.history = {}


        tf.reset_default_graph()

        # self.input(
        #     batch_size=self.BATCH_SIZE,
        #     input_x_dtype=self.input_x_dtype,
        #     input_z_dtype=self.input_z_dtype,
        #     input_x_shape=self.input_x_shape,
        #     input_z_shape=self.input_z_shape,
        #     is_training=True,
        # )

        self.build_BEGAN(
            batch_norm_ok=self.BATCH_NORM_OK,
            conv_activation=self.CONV_ACTIVATION,
            dropout=self.DROPOUT,
            lr_decaying=self.LR_DECAYING,
            learning_rate=self.LEARNING_RATE,
            adam_beta1=self.ADAM_BETA1,
            adam_beta2=self.ADAM_BETA2,
        )
        # self.build_AnoGAN(
        #     learning_rate = self.LEARNING_RATE,
        #     adam_beta1 = self.ADAM_BETA1,
        #     dropout=self.DROPOUT,
        #     ano_lambda_=self.ANO_LAMBDA_,
        #     use_featuremap=self.USE_FEATUREMAP,
        # )


    def _img_parse_fn(self, filenames, data_z):
        image_string = tf.read_file(filenames)
        # image_decoded = tf.image.decode_jpeg(image_string)
        extension_string = self.input_x_ext
        if extension_string in ['jpg', 'jpeg']:
            image_decoded = tf.image.decode_jpeg(
                image_string,
                channels=3,
            )
        elif extension_string in ['png']:
            image_decoded = tf.image.decode_png(
                image_string,
                channels=3,
                dtype=tf.uint8,
            )
        image_cropped = tf.image.central_crop(
            image_decoded,
            central_fraction=.7,
        )
        # image_resized = tf.image.crop_and_resize(
        #     image_decoded,
        #     boxes,
        #     box_ind,
        #     crop_size,
        #     method='bilinear',
        #     extrapolation_value=0,
        #     name='image_crop_and_resize',
        # )
        image_resized = tf.image.resize_images(
            image_cropped,
            [self.INPUT_HEIGHT, self.INPUT_WIDTH],
        )
        # tf.image.crop_and_resize(
        #     image_decoded,
        #     boxes=[[0, 0, 1, 1]],
        #     box_ind=[0],
        #     crop_size=[self.INPUT_HEIGHT, self.INPUT_WIDTH],
        #     method='bilinear',
        #     extrapolation_value=0,
        #     name='image_cropping_and_resizing',
        # )

        image_scaled = tf.subtract(
            tf.multiply(
                tf.div(
                    tf.cast(
                        image_resized,
                        tf.float32,
                    ),
                    255.,
                ),
                2.,
            ),
            1.,
            name='image_scaling',
        )

        return image_scaled, data_z

    def _print_scope(
        self,
        name,
        sep_length=47,
        ):
        print(
            '{{{name:^15s}}}'.format(name=name) + ' ' + '-' * sep_length
        )

    def _print_layer(
        self,
        name,
        input_shape=None,  # input_.get_shape(),
        output_shape=None,  # final_layer.get_shape(),
        ):

        string_format = " ".join(
            [
                "{name:15s}\t|",
                "{input_shape:20s}",
                "-> ",
                "{output_shape:20s}",
                "|",
            ]
        )
        print(
            string_format.format(
                name=name,
                input_shape=str(input_shape),
                output_shape=str(output_shape),
            )
        )

    def _print_parameter(
        self,
        parameter_dict,
        ):

        string_format = ''.join(
            [
                '| {prmt_key:18s}',
                ': '
                '{prmt_val:>8s} |',
            ]
        )
        parameter_str = "\n".join(
            # f"| {prmt_key:18s}: {str(prmt_value):>8s} |"
            string_format.format(
                prmt_key=prmt_key,
                prmt_val=str(prmt_val),
            )
            for prmt_key, prmt_val in parameter_dict.items()
        )
        print(
            "=" * 7 + " Given Parameters " +  "=" * 7,
            parameter_str,
            "=" * 32,
            sep='\n',
        )

    def input(
        self,
        batch_size=64,
        input_x_dtype=tf.float32,
        input_z_dtype=tf.float32,
        input_x_shape=(None, 64, 64, 3),
        input_z_shape=(None, 100),
        is_training=True,
        drop_remainder=False,
        ):
        """Input function.

        TBD

        Parameter
        ---------
        mode: (string) {'train', 'eval'}
            or any other mode you can think of.
            At training, we shuffle the data and have multiple epochs
        input_x: numpy.ndarray

        input_y: numpy.ndarray

        Example
        -------
        >>> vrnn = VRNN()
        >>> vrnn.input_fn('train', x_train, y_train)
        >>> eval_input_spec = input_fn('evaluate', x_test, y_test)

        """
        with tf.name_scope('input'):
            if batch_size is None:
                batch_size = self.BATCH_SIZE

            buffer_size = self.BUFFER_SIZE if is_training else 1

            assert self.INPUT_WIDTH == self.INPUT_HEIGHT

            # X_t = tf.placeholder(self.input_x_dtype, self.input_x_shape,
            #                      name='x_tensor_interface')
            X_file_t= tf.placeholder(tf.string, (None,),
                                 name='x_filename_interface')
            Z_t = tf.placeholder(self.input_z_dtype, self.input_z_shape,
                                 name='z_tensor_interface')

            dataset = tf.data.Dataset.from_tensor_slices((X_file_t, Z_t))
            dataset = dataset.map(self._img_parse_fn)
            dataset = dataset.shuffle(buffer_size=buffer_size)  # reshuffle_each_iteration=True as default.
            dataset = dataset.batch(
                batch_size,
                drop_remainder=drop_remainder,
            )
            dataset = dataset.flat_map(
                lambda data_x, data_z: tf.data.Dataset.zip(
                    (
                        tf.data.Dataset.from_tensors(data_x),
                        tf.data.Dataset.from_tensors(data_z),
                    )
                )#.repeat(repeat_num)
            )

            data_op = dataset.make_initializable_iterator()
            data_init_op = data_op.initializer
            next_batch = X_batch, Z_batch = data_op.get_next()

            print('[dtype] X: %s , Z: %s' % (X_batch.dtype, Z_batch.dtype))
            print('[shape] X: %s , Z: %s' % (X_batch.get_shape(), Z_batch.get_shape()))

        #self._x_input_tensor = X_t
        self._x_input_filename_tensor = X_file_t
        self._z_input_tensor = Z_t
        self._x_batch_tensor = X_batch
        self._z_batch_tensor = Z_batch
        self.next_batch = next_batch
        self.data_init_op = data_init_op


    def _layer_dense(
        self,
        input_,
        output_size,
        is_training=True,
        stddev=0.02,
        bias_start=0.,
        batch_norm_ok=False,
        dropout=False,
        activation=None,
        reuse=tf.AUTO_REUSE,
        name='dense',
        ):

        with tf.variable_scope(name, reuse=reuse):

            with tf.variable_scope('dense', reuse=reuse):
                densed = tf.layers.dense(
                    input_,
                    units=output_size,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.,
                        stddev=stddev,
                    ),
                    bias_initializer=tf.constant_initializer(bias_start),
                    name=name,
                )
                final_layer = densed

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        densed,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                    final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                activated = activation(
                    final_layer,
                    name='activation',
                )
                final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.nn.dropout(
                    final_layer,
                    keep_prob=keep_prob,
                    name='dropout',
                )
                final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=final_layer.get_shape(),
        )

        return final_layer


    def _layer_linear(
        self,
        input_,
        output_size,
        is_training=True,
        stddev=0.02,
        bias_start=0.,
        batch_norm_ok=False,
        activation=None,
        dropout=.3,
        return_weight=False,
        reuse=tf.AUTO_REUSE,
        name='linear',
        ):

        with tf.variable_scope(name, reuse=reuse):

            input_shape = input_.get_shape().as_list()

            with tf.variable_scope('linear', reuse=reuse):
                weight = tf.get_variable(
                    'weight',
                    shape=(input_shape[1], output_size),
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(
                        #mean=0.,
                        stddev=stddev,
                    ),
                )
                bias = tf.get_variable(
                    'bias',
                    shape=(output_size,),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start),
                )
                lineared = tf.matmul(
                    input_,
                    weight,
                )
                lineared_biased = tf.nn.bias_add(
                    lineared,
                    bias,
                    #data_format='NHWC',
                    name='linear_function',
                )
                final_layer = lineared_biased

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                activated = activation(
                    final_layer,
                    name='activation',
                )
                final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.nn.dropout(
                    final_layer,
                    keep_prob=keep_prob,
                    name='dropout',
                )
                final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=final_layer.get_shape(),
        )

        if return_weight:
            return final_layer, weight, bias
        else:
            return final_layer


    def _layer_reshape(
        self,
        input_,
        output_shape,
        batch_norm_ok=True,
        is_training=True,
        activation=tf.nn.relu,
        dropout=.3,
        reuse=tf.AUTO_REUSE,
        name='reshaped_proj_z',
        ):
        """Reshape Layer.
        """
        with tf.variable_scope(name, reuse=reuse):

            reshaped = tf.reshape(
                input_,
                shape=output_shape,
                # shape=(
                #     -1,  # `tf.reshape` needs -1, not `None`
                #     height_0,
                #     width_0,
                #     channel_0,
                # ),
                name='reshape',
            )
            final_layer = reshaped

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                activated = activation(
                    final_layer,
                    name='activation',
                )
                final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                keep_prob = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda : tf.constant((1. - dropout)),
                    lambda : tf.constant(1.),
                    name='choose_prob_if_training'
                )
                # keep_prob = tf.case(
                #     {
                #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                #     },
                #     default=tf.constant(1.),
                #     name='choose_prob_if_training',
                #     exclusive=True,
                # )

                # TBD: tf.layers.dropout()
                dropoutted = tf.nn.dropout(
                    final_layer,
                    keep_prob=keep_prob,
                    name='dropout',
                )
                final_layer = dropoutted

            self._print_layer(
                name=name,
                input_shape=input_.get_shape(),
                output_shape=final_layer.get_shape(),
            )

        return final_layer


    def _layer_conv2d(
        self,
        input_,
        output_size,
        is_training=True,
        filter_size=3,
        stride_size=1,
        stddev=.02,
        batch_norm_ok=True,
        activation=tf.nn.relu,
        dropout=.3,
        reuse=tf.AUTO_REUSE,
        name='conv2d',
        ):
        """Convolution Layer.
        """
        with tf.variable_scope(name, reuse=reuse):

            # Get Shapes
            (batch_size,
             input_height,
             input_width,
             input_channels) = input_.get_shape()
            (batch_size,
             output_height,
             output_width,
             output_channels) = output_size

            if output_size[0] in (None, -1):
                output_size = (
                    tf.shape(input_)[0],
                    output_height,
                    output_width,
                    output_channels,
                )
            with tf.variable_scope('conv', reuse=reuse):
                # Filter shape : [height, width, output_channels, in_channels]
                filter_weight = tf.get_variable(
                    'conv2d_filter_weight',
                    shape=(
                        filter_size,
                        filter_size,
                        input_channels,
                        output_channels,
                    ),
                    initializer=tf.truncated_normal_initializer(
                        mean=0.0,
                        stddev=stddev,
                    )
                )

                conv2d = tf.nn.conv2d(
                    input_,
                    filter=filter_weight,
                    strides=[
                        1,
                        stride_size,
                        stride_size,
                        1,
                    ],
                    padding='SAME',
                    #use_cudnn_on_gpu=True,
                    data_format='NHWC',
                    name='conv2d',
                    )

                bias = tf.get_variable(
                    'conv2d_bias',
                    shape=[output_channels],
                    initializer=tf.constant_initializer(
                        value=0.0,
                    ),
                )
                conv2d_biased = tf.nn.bias_add(conv2d, bias)
                # conv2d = tf.reshape(
                #     tf.nn.bias_add(conv2d, bias),
                #     shape=(
                #         -1,
                #         output_height,
                #         output_width,
                #         output_channels,
                #     ),
                # )
                final_layer = conv2d_biased

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                with tf.variable_scope('activation', reuse=reuse):
                    activated = activation(
                        final_layer,
                        name='activation',
                    )
                    final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                with tf.variable_scope('dropout', reuse=reuse):
                    # keep_prob = tf.constant(
                    #     1. - (
                    #         tf.to_float(is_training) * dropout
                    #     )
                    # )
                    keep_prob = tf.cond(
                        tf.cast(is_training, tf.bool),
                        lambda : tf.constant((1. - dropout)),
                        lambda : tf.constant(1.),
                        name='choose_prob_if_training'
                    )
                    # keep_prob = tf.case(
                    #     {
                    #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                    #     },
                    #     default=tf.constant(1.),
                    #     name='choose_prob_if_training',
                    #     exclusive=True,
                    # )

                    # TBD: tf.layers.dropout()
                    dropoutted = tf.nn.dropout(
                        final_layer,
                        keep_prob=keep_prob,
                        name='dropout',
                    )
                    final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=final_layer.get_shape(),
        )

        return final_layer


    def _layer_conv2d_t(
        self,
        input_,
        output_size,
        is_training=True,
        filter_size=3,
        stride_size=1,
        stddev=.02,
        batch_norm_ok=True,
        activation=tf.nn.relu,
        dropout=.3,
        name='conv2d_t',
        reuse=tf.AUTO_REUSE,
        ):
        """Transposed-Convolutional Layer.

        It is often called
        Transposed Convolutions or
        Fractionally-strided Convolutions.

        Parameters
        ----------
        input_: tf.Tensor
            shape: `[batch, height, width, in_channels]`

        output_size: int
            shape: `[batch, height, width, in_channels]`

        filter_size: int
            A int number as a filter height & width.

        strides: int
            A int number to stride.

        stddev: float
            A float number for the standard deviation of an initializer.

        name: str
            A name string for variable scope & conv2d_t layer.

        Return
        ------
        tf.Tensor `conv2d_t`
        """
        with tf.variable_scope(name, reuse=reuse):

            # Get Shapes
            (batch_size,
             input_height,
             input_width,
             input_channels) = input_.get_shape()
            (batch_size,
             output_height,
             output_width,
             output_channels) = output_size

            if output_size[0] in (None, -1):
                output_size = (
                    tf.shape(input_)[0],
                    output_height,
                    output_width,
                    output_channels,
                )

            # Transposed Convolutional Layer with Filter and Bias
            with tf.variable_scope('conv2d_t', reuse=reuse):
                # Filter shape : [height, width, output_channels, in_channels]
                filter_weight = tf.get_variable(
                    'conv2d_t_filter_weight',
                    shape=(
                        filter_size,
                        filter_size,
                        output_channels,
                        input_channels,
                    ),
                    initializer=tf.random_normal_initializer(
                        mean=0.0,
                        stddev=stddev)
                )

                conv2d_t = tf.nn.conv2d_transpose(
                    input_,
                    filter=filter_weight,
                    output_shape=output_size,
                    strides=[
                        1,
                        stride_size,
                        stride_size,
                        1,
                    ],
                    padding='SAME',
                    data_format='NHWC',
                    name='conv2d_transpose',
                )

                bias = tf.get_variable(
                    'conv2d_t_bias',
                    shape=[output_channels],
                    initializer=tf.constant_initializer(
                        value=0.0,
                    ),
                )
                conv2d_t_biased = tf.nn.bias_add(conv2d_t, bias)
            # conv2d_t = tf.reshape(
            #     tf.nn.bias_add(conv2d_t, bias),
            #     shape=(
            #         -1,
            #         output_height,
            #         output_width,
            #         output_channels,
            #     ),
            # )
                final_layer = conv2d_t_biased

            # (Optional) Batch Normalization
            if batch_norm_ok:
                # TBD: tf.layers.batch_normalization()
                with tf.variable_scope('batch_norm', reuse=reuse) as scope:
                    batch_normed = tf.contrib.layers.batch_norm(
                        final_layer,
                        epsilon=1e-5,
                        decay=.9,
                        updates_collections=None,
                        scale=True,
                        is_training=is_training,
                        scope=scope,
                    )
                final_layer = batch_normed

            # (Optional) Activation Layer
            if activation:
                with tf.variable_scope('activation', reuse=reuse):
                    activated = activation(
                        final_layer,
                        name='activation',
                    )
                    final_layer = activated

            # (Optional) Dropout Layer
            if dropout:

                with tf.variable_scope('dropout', reuse=reuse):
                    keep_prob = tf.cond(
                        tf.cast(is_training, tf.bool),
                        lambda : tf.constant((1. - dropout)),
                        lambda : tf.constant(1.),
                        name='choose_prob_if_training'
                    )
                    # keep_prob = tf.case(
                    #     {
                    #         tf.equal(is_training, tf.constant(True)): tf.constant((1. - dropout)),
                    #     },
                    #     default=tf.constant(1.),
                    #     name='choose_prob_if_training',
                    #     exclusive=True,
                    # )

                    # TBD: tf.layers.dropout()
                    dropoutted = tf.nn.dropout(
                        final_layer,
                        keep_prob=keep_prob,
                        name='dropout',
                    )
                    final_layer = dropoutted

        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=final_layer.get_shape(),
        )

        return final_layer


    def _subgraph_encoder(
        self,
        input_x,
        # n,
        # z_dim,
        # channel_dim,
        filter_size=3,
        stride_size=1,
        conv_activation=tf.nn.elu,
        dropout=None,
        is_training=True,
        reuse=tf.AUTO_REUSE,
        name='encoder',
        ):
        """
        Reconstruction network
        """
        # h = tf.layers.dense(input_h, img_dim * img_dim * n, activation=None)
        # h = tf.reshape(h, (-1, img_dim, img_dim, n))
        with tf.variable_scope(name, reuse=reuse):
            output_size_arr = np.array(
                [self.INPUT_HEIGHT, self.INPUT_WIDTH]
            )
            stacked_num = int(np.log2(self.INPUT_HEIGHT) - 2)
            output_size_arr = np.array(
                [output_size_arr // (2 ** i) for i in range(stacked_num)]
            )
            height_arr = output_size_arr[:, 0]
            width_arr = output_size_arr[:, 1]


            # channel_gen_x = self.OUTPUT_CHANNEL       #   3
            # channel_4 = self.G_FILTER_DIM * (2 ** 0)  #  32
            # channel_3 = self.G_FILTER_DIM * (2 ** 1)  #  64
            # channel_2 = self.G_FILTER_DIM * (2 ** 2)  # 128
            # channel_1 = self.G_FILTER_DIM * (2 ** 3)  # 256
            # channel_0 = self.G_FILTER_DIM * (2 ** 4)  # 512
            channel = self.FILTER_DIM


            final_layer = input_x
            for i in range(stacked_num - 1):
                conv2d = self._layer_conv2d(
                    final_layer,
                    output_size=(
                        None,
                        height_arr[i],
                        width_arr[i],
                        channel * (2 ** i),
                    ),
                    filter_size=filter_size,
                    stride_size=stride_size + 1,
                    batch_norm_ok=False,
                    activation=conv_activation,
                    dropout=dropout,
                    name='d_sampling_%s' % (i),
                    is_training=is_training,
                )
                final_layer = conv2d

                for _ in range(3):
                    conv2d = self._layer_conv2d(
                        final_layer,
                        output_size=(
                            None,
                            height_arr[i],
                            width_arr[i],
                            channel * (i + 2),
                        ),
                        filter_size=filter_size,
                        stride_size=stride_size,
                        batch_norm_ok=False,
                        activation=conv_activation,
                        dropout=dropout,
                        name='conv2d_%s_%s' % (i, _),
                        is_training=is_training,
                    )
                    final_layer = conv2d

            projection_size = np.prod(final_layer.shape[1:])
            reshaped_proj_z = self._layer_reshape(
                final_layer,
                output_shape=(
                    -1,  # `tf.reshape` needs -1, not `None`
                    projection_size,
                ),
                batch_norm_ok=False,
                activation=None,
                dropout=dropout,
                name='reshaped_proj_z',
                is_training=is_training,
            )
            encoded = self._layer_dense(
                reshaped_proj_z,
                output_size=self.Z_DIM,
                batch_norm_ok=False,
                activation=None,
                dropout=dropout,
                is_training=is_training,
                name='proj_z_dense',
            )
            final_layer = encoded

        return final_layer


    def _subgraph_decoder(
        self,
        input_z,
        # n,
        # img_dim,
        # channel_dim,
        filter_size=3,
        stride_size=1,
        conv_activation=tf.nn.elu,
        dropout=None,
        is_training=True,
        reuse=tf.AUTO_REUSE,
        name='decoder',
        ):
        """
        Reconstruction network
        """
        # h = tf.layers.dense(input_h, img_dim * img_dim * n, activation=None)
        # h = tf.reshape(h, (-1, img_dim, img_dim, n))
        with tf.variable_scope(name, reuse=reuse):
            output_size_arr = np.array(
                [self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH]
            )
            stacked_num = int(np.log2(self.OUTPUT_HEIGHT) - 2)
            output_size_arr = np.array(
                [output_size_arr // (2 ** i) for i in range(stacked_num)]
            )[::-1]
            height_arr = output_size_arr[:, 0]
            width_arr = output_size_arr[:, 1]


            # channel_gen_x = self.OUTPUT_CHANNEL       #   3
            # channel_4 = self.G_FILTER_DIM * (2 ** 0)  #  32
            # channel_3 = self.G_FILTER_DIM * (2 ** 1)  #  64
            # channel_2 = self.G_FILTER_DIM * (2 ** 2)  # 128
            # channel_1 = self.G_FILTER_DIM * (2 ** 3)  # 256
            # channel_0 = self.G_FILTER_DIM * (2 ** 4)  # 512
            channel = self.FILTER_DIM


            projection_size = (
                height_arr[0] * width_arr[0] * channel
            )

            with tf.name_scop('projection'):
                proj_z = self._layer_dense(
                    input_z,
                    output_size=projection_size,
                    batch_norm_ok=False,
                    activation=None,
                    dropout=dropout,
                    is_training=is_training,
                    name='proj_z_dense',
                )
                reshaped_proj_z = self._layer_reshape(
                    proj_z,
                    output_shape=(
                        -1,  # `tf.reshape` needs -1, not `None`
                        height_arr[0],
                        width_arr[0],
                        channel,
                    ),
                    batch_norm_ok=False,
                    activation=None,
                    dropout=dropout,
                    name='reshaped_proj_z',
                    is_training=is_training,
                )
                final_layer = reshaped_proj_z

            for i in range(stacked_num):
                if i > 0:
                    upsample_name = 'upsampling_%s' % i
                    with tf.variable_scope(upsample_name):
                        resized = tf.image.resize_nearest_neighbor(
                            images=final_layer,
                            size=[height_arr[i], width_arr[i]],
                            name=upsample_name,
                        )
                        self._print_layer(
                            name=upsample_name,
                            input_shape=final_layer.get_shape(),
                            output_shape=resized.get_shape(),
                        )

                    final_layer = resized

                for _ in range(2):

                    conv2d = self._layer_conv2d(
                        final_layer,
                        output_size=(
                            None,
                            height_arr[i],
                            width_arr[i],
                            channel,
                        ),
                        filter_size=filter_size,
                        stride_size=stride_size,
                        batch_norm_ok=False,
                        activation=conv_activation,
                        dropout=dropout,
                        name='conv2d_%s_%s' % (i, _),
                        is_training=is_training,
                    )
                    final_layer = conv2d

            decoded = self._layer_conv2d(
                final_layer,
                output_size=(
                    None,
                    height_arr[i],
                    width_arr[i],
                    self.OUTPUT_CHANNEL,
                ),
                filter_size=filter_size,
                stride_size=stride_size,
                batch_norm_ok=False,
                activation=None,
                # activation=tf.nn.tanh,
                # activation=tf.nn.sigmoid,
                dropout=dropout,
                name='conv2d_%s' % (i + 1),
                is_training=is_training,
            )
            final_layer = decoded

        return final_layer


    def _graph_generator(
        self,
        z,
        is_training=True,
        # filter_size=5,
        # stride_size=2,
        dropout=None,
        conv_activation=tf.nn.elu,
        reuse=tf.AUTO_REUSE,
        name='generator',
        ):

        self._print_scope(name)

        with tf.variable_scope(name, reuse=reuse):

            decoded = self._subgraph_decoder(
                z,
                is_training=is_training,
                conv_activation=conv_activation,
                dropout=dropout,
                reuse=reuse,
            )

        return decoded


    def _graph_discriminator(
        self,
        input_x,
        is_training=True,
        # filter_size=5,
        # stride_size=2,
        dropout=None,
        conv_activation=tf.nn.elu,
        reuse=tf.AUTO_REUSE,
        name='discriminator',
        return_all_layers=False,
        ):

        self._print_scope(name)

        with tf.variable_scope(name, reuse=reuse):

            encoded = self._subgraph_encoder(
                input_x,
                is_training=is_training,
                conv_activation=conv_activation,
                dropout=dropout,
                reuse=reuse,
            )
            decoded = self._subgraph_decoder(
                encoded,
                is_training=is_training,
                conv_activation=conv_activation,
                dropout=dropout,
                reuse=reuse,
            )

        return decoded

        if return_all_layers:
            return None
            # return (
            #     conv2d_0,
            #     conv2d_1,
            #     conv2d_2,
            #     conv2d_3,
            #     conv2d_4,
            #     conv2d_4_1x1,
            #     reshaped_proj_y,
            #     y,
            #     activated_y,
            # )
        else:
            return decoded

    def _graph_sampler(
            self,
            z,
            is_training=False,
            # filter_size=5,
            # stride_size=2,
            dropout=None,
            name='sampler',
            generator_name='generator',
            ):

        self._print_scope(name)

        with tf.variable_scope(name, reuse=True):
            sample_G = self._graph_generator(
                z,
                is_training=is_training,
                # filter_size=filter_size,
                # stride_size=stride_size,
                reuse=True,
                dropout=dropout,
                name=generator_name,
                )
            sample_G = tf.stop_gradient(sample_G)

        return sample_G

    def _graph_discriminator_featuremap(
        self,
        input_x,
        is_training=False,
        # filter_size=5,
        # stride_size=2,
        dropout=None,
        name='discriminator_featuremap',
        discriminator_name='discriminator',
        return_layer_idx=3,
        ):
        """
        layers:
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            reshaped_proj_y,
            y,
        """
        self._print_scope(name)

        with tf.variable_scope(name, reuse=True):
            layers = self._graph_discriminator(
                input_x,
                is_training=is_training,
                # filter_size=filter_size,
                # stride_size=stride_size,
                dropout=dropout,
                reuse=True,
                name=discriminator_name,
                return_all_layers=True,
            )
            layers = [tf.stop_gradient(layer) for layer in layers]

        return layers[return_layer_idx]

    def _graph_anomaly_z_distributor(
        self,
        input_z,
        is_training=True,
        name='anomaly_z_distributor',
        ):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # anomaly_z = self._layer_linear(
            #     input_z,
            #     is_training=is_training,
            #     output_size=input_z.get_shape()[-1],
            #     name=name,
            #     return_weight=False,
            # )
            anomaly_z = self._layer_dense(
                input_z,
                is_training=is_training,
                output_size=input_z.get_shape()[-1],
                name=name,
            )
        return anomaly_z


    def build_BEGAN(
        self,
        batch_norm_ok=False,
        conv_activation=tf.nn.elu,
        dropout=None,
        lr_decaying=True,
        learning_rate=.0002,
        adam_beta1=.5,
        adam_beta2=.8,
        ):
        """Short Description.

        TBD

        Attributes
        ----------
        TBD


        Example
        -------
        >>> "TBD"
        >>> train_model_spec = model_fn('train', train_input_spec, hyper_param,
                                    reuse=False)
        >>> eval_model_spec = model_fn('evaluate', eval_input_spec, hyper_param,
                                   reuse=True)

        """
        #is_training = (mode == 'train')
        reuse_ok =  tf.AUTO_REUSE  # (mode != 'train')
        # class_num = self.CLASS_NUM
        # # batch_size = self.BATCH_SIZE
        #
        # X_batch = self._x_batch_tensor
        # Z_batch = self._z_batch_tensor

        print('\n[  began_MODEL  ] ' + '=' * 47)

        # Objective Functions ================================================

        with tf.variable_scope('began_model', reuse=reuse_ok):

            # Input_X_began = tf.placeholder(
            #     self._x_input_tensor.dtype,
            #     self._x_input_tensor.get_shape(),
            #     name='input_x_began',
            # )
            Input_X_began = tf.placeholder(
                self.input_x_dtype,
                self.input_x_shape,
                name='input_x_began',
            )
            Input_Z_began = tf.placeholder(
                self._z_input_tensor.dtype,
                self._z_input_tensor.get_shape(),
                name='input_z_began',
            )
            Bool_is_training = tf.placeholder(
                tf.bool,
                (None),  #[None, 1],
                name='bool_is_training',
            )
            Input_lambda = tf.placeholder(
                tf.float32,
                (None),  #[None, 1],
                name='input_lambda',
            )
            Input_gamma = tf.placeholder(
                tf.float32,
                (None),  #[None, 1],
                name='input_gamma',
            )
            Start_learning_rate_tensor = tf.placeholder(
                tf.float32,
                (None),
                name='start_learning_rate_tensor',
            )
            G = self._graph_generator(
                z=Input_Z_began,  # z=Z_batch,
                # filter_size=5,
                # stride_size=2,
                conv_activation=self.CONV_ACTIVATION,
                dropout=dropout,
                is_training=Bool_is_training,
                name='generator',
            )
            D_real_y = self._graph_discriminator(
            #D_real_sigmoid_y = self._graph_discriminator(
                input_x=Input_X_began,  # input_x=X_batch,
                # filter_size=5,
                # stride_size=2,
                conv_activation=self.CONV_ACTIVATION,
                dropout=dropout,
                is_training=Bool_is_training,
                reuse=tf.AUTO_REUSE,
                name='discriminator',
                return_all_layers=False,
            )
            # S = self._graph_sampler(
            #     z=Input_Z_began,  # z=Z_batch,
            #     filter_size=5,
            #     stride_size=2,
            #     generator_name='generator',
            # )
            D_fake_y = self._graph_discriminator(
            #D_fake_sigmoid_y = self._graph_discriminator(
                input_x=G,
                # filter_size=5,
                # stride_size=2,
                conv_activation=self.CONV_ACTIVATION,
                dropout=dropout,
                is_training=Bool_is_training,
                reuse=True,
                name='discriminator',
                return_all_layers=False,
            )
            #G_to_img_rgb = (G + 1.) / 2. * 255.

            summary_Z = tf.summary.histogram('z', Input_Z_began)
            summary_G_img = tf.summary.image('G', G)
            summary_G_img_batch = tf.summary.image('G_to_img_batch', G)
            #summary_G_img = tf.summary.image('G_to_img', G_to_img)
            summary_D_real = tf.summary.histogram('D_real', D_real_y)
            summary_D_fake = tf.summary.histogram('D_fake', D_fake_y)


            with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):

                def pixel_autoencoder_loss(out, inp):
                            '''
                            The autoencoder loss used is the L1 norm
                            (note that this is based on the pixel-wise
                            distribution of losses that the authors assert
                            approximates the Normal distribution)
                            args:
                                out:  discriminator output
                                inp:  discriminator input
                            returns:
                                L1 norm of pixel-wise loss
                            '''
                            eta = 1  # paper uses L1 norm
                            diff = tf.abs(out - inp)
                            if eta == 1:
                                return tf.reduce_mean(diff)
                            else:
                                return tf.reduce_mean(tf.pow(diff, eta))

                pxl_loss_real = pixel_autoencoder_loss(D_real_y, Input_X_began)
                pxl_loss_fake = pixel_autoencoder_loss(D_fake_y, G)
                K = tf.get_variable(
                                name='K',
                                shape=[],
                                initializer=tf.constant_initializer(0.),
                                trainable=False,
                            )
                with tf.name_scope('loss_G_scope'):
                    loss_G = pxl_loss_fake

                with tf.name_scope('loss_D_scope'):
                    loss_D = pxl_loss_real - (K * pxl_loss_fake)

                with tf.name_scope('k_update'):
                    balance = (Input_gamma * pxl_loss_real) - pxl_loss_fake
                    K_next = K + (Input_lambda * balance)
                    update_K = tf.assign(
                        K,
                        tf.clip_by_value(K_next, 0., 1.),
                    )
                with tf.name_scope('convergence_measure'):
                    convergence_measure = pxl_loss_real + np.abs(balance)

            summary_loss_G = tf.summary.scalar('loss_G', loss_G)
            summary_loss_D = tf.summary.scalar('loss_D', loss_D)
            summary_K = tf.summary.scalar('K', K)
            summary_balance = tf.summary.scalar('balance', balance)
            summary_convergence_measure = tf.summary.scalar(
                'Convergence_measure',
                convergence_measure,
            )

        # Summaries for training
        # summary_op = tf.summary.merge_all()
        summary_op_G = tf.summary.merge([
            summary_Z,
            summary_G_img,
            summary_loss_G,
            summary_K,
            summary_balance,
            summary_convergence_measure,
        ])
        summary_op_G_batch = tf.summary.merge(
            [
                summary_G_img_batch,
            ],
        )
        summary_op_D = tf.summary.merge([
            summary_Z,
            summary_D_real,
            summary_D_fake,
            summary_loss_D,
        ])
        # ====================================================================


        # Optimization =======================================================

        with tf.variable_scope('began_optimization'):

            # for _ in range(10):
            #     print(0.0005 * (0.6 ** _))
            global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step',
            )
            if lr_decaying:
                learning_rate_decay = tf.train.exponential_decay(
                    learning_rate=Start_learning_rate_tensor,
                    global_step=global_step,
                    decay_steps=self.DECAY_LR_STEP,
                    decay_rate=.96,
                    staircase=True,
                    # decay_steps=20,
                    # decay_rate=.0005,
                    # staircase=True,
                    name='learning_rate_decay',
                )
            else:
                learning_rate_decay = Start_learning_rate_tensor

            learning_rate_decay = tf.clip_by_value(
                learning_rate_decay,
                1e-5,
                1e-4,
            )

            summary_learning_rate_decay = tf.summary.scalar(
                'learning_rate',
                learning_rate_decay,
            )

            optimizer_G = tf.train.AdamOptimizer(
                learning_rate=learning_rate_decay,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=1e-08,
                name='optimizer_generator',
            )
            optimizer_D = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=1e-08,
                name='optimizer_discriminator',
            )

            train_op_G = optimizer_G.minimize(
                loss_G,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="began_model/generator",
                ),
                global_step=global_step,
            )
            train_op_D = optimizer_D.minimize(
                loss_D,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="began_model/discriminator",
                ),
                global_step=global_step,
            )

        # variable_init_op_began = tf.group(*[tf.global_variables_initializer(),
        #                               tf.tables_initializer()])
        variable_began = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='began_',
        )
        variable_init_op_began = tf.group(
            *[
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='began_model',
                    )
                ),
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='began_optimization',
                    )
                ),
            ],
        )

        # ====================================================================

        with tf.variable_scope("began_metrics", reuse=reuse_ok):
            metrics_train_began = {
                'Train_loss_G': tf.metrics.mean(loss_G),
                'Train_loss_D': tf.metrics.mean(loss_D),
                'Learing_rate': tf.metrics.mean(learning_rate_decay),

            }
            metrics_valid_began = {
                'Valid_loss_G': tf.metrics.mean(loss_G),
                'Valid_loss_D': tf.metrics.mean(loss_D),
            }

        # Group the update ops for the tf.metrics
        update_metrics_op_train_began = tf.group(
            *[op for _, op in metrics_train_began.values()]
        )
        update_metrics_op_valid_began = tf.group(
            *[op for _, op in metrics_valid_began.values()]
        )

        # Get the op to reset the local variables used in tf.metrics
        metrics_init_op_began = tf.variables_initializer(
            var_list=tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="began_metrics",
            ),
            name='metrics_init_op_began',
        )

        # Return
        self.variable_began = variable_began
        self.variable_init_op_began = variable_init_op_began

        self.train_op_G = train_op_G
        self.train_op_D = train_op_D

        self.metrics_train_began = metrics_train_began
        self.metrics_valid_began = metrics_valid_began
        self.update_metrics_op_train_began = update_metrics_op_train_began
        self.update_metrics_op_valid_began = update_metrics_op_valid_began
        self.metrics_init_op_began = metrics_init_op_began

        self.summary_op_G = summary_op_G
        self.summary_op_D = summary_op_D
        self.summary_op_G_batch = summary_op_G_batch
        self.summary_learning_rate_decay = summary_learning_rate_decay

        self.Input_X_began = Input_X_began
        self.Input_Z_began = Input_Z_began
        self.Bool_is_training = Bool_is_training
        self.Input_lambda = Input_lambda
        self.Input_gamma = Input_gamma
        self.Start_learning_rate_tensor = Start_learning_rate_tensor

        # global_step = tf.train.get_global_step()
        self.global_step = global_step
        self.G = G
        # self.S = S
        self.D_real = D_real_y
        self.D_fake = D_fake_y
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.K = K
        self.update_K = update_K
        self.convergence_measure = convergence_measure


    def train_BEGAN(
        self,
        #input_x=None,
        input_x_filenames=None,
        input_z=None,
        batch_size=64,
        drop_remainder=False,
        epoch_num=2,
        validation_ratio=.2,
        learning_rate=0.0005,
        lambda_val=.001,
        gamma_val=.7,
        model_save_dir='./model_save/began',
        pre_trained_path=None,
        verbose=False,
        writer=None,
        ):
        """Short Description.

        TBD

        Attributes
        ----------
        TBD


        Example
        -------
        >>> "TBD"
        >>> train_fn(x_train, y_train, train_model_spec,
                 3, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path=None,
                 verbose=True)

        >>> train_fn(x_train, y_train, train_model_spec,
                 10, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path='model_save')

        """
        metrics_train = self.metrics_train_began
        metrics_valid = self.metrics_valid_began

        input_x = input_x_filenames
        # --------------------------------------------------------------------

        self.LEARNING_RATE = learning_rate
        self.VALIDATION_RATIO = validation_ratio

        try:
            if not self.VALIDATION_RATIO or not (0 < self.VALIDATION_RATIO < 1):
                raise NoValidValueException(
                    'validation_ratio',
                    self.VALIDATION_RATIO,
                )
            else:
                pass
                # print("Training Parameter\n" +
                #       "'VALIDATION_RATIO=%s'" % self.VALIDATION_RATIO
                # )
        except KeyError:
            raise NoValidValueException(
                'validation_ratio',
                None,
            )
        except NoValidValueException as err:
            # print(err)
            self.VALIDATION_RATIO = 0

        if batch_size is None:
            batch_size_int = self.BATCH_SIZE
        else:
            batch_size_int = int(batch_size)


        parameter_dict = {
            'VALIDATION_RATIO': self.VALIDATION_RATIO,
            'DROPOUT': self.DROPOUT,
            'LEARNING_RATE': self.LEARNING_RATE,
            'LR_DECAYING': self.LR_DECAYING,
            'BATCH_SIZE': batch_size_int,
            'EPOCH_NUM': epoch_num,
            'LAMBDA': lambda_val,
            'GAMMA': gamma_val,
            'ADAM_BETA1': self.ADAM_BETA1,
            'ADAM_BETA2': self.ADAM_BETA2,
        }
        self._print_parameter(parameter_dict)


        self.input(
            batch_size=batch_size_int,
            input_x_dtype=self.input_x_dtype,
            input_z_dtype=self.input_z_dtype,
            input_x_shape=self.input_x_shape,
            input_z_shape=self.input_z_shape,
            is_training=True,
            drop_remainder=drop_remainder,
        )


        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver(
            var_list=self.variable_began,
            max_to_keep=2,  # will keep last 5 epochs as default
            keep_checkpoint_every_n_hours=1.,
            name='saver',
        )
        begin_at_epoch = 0

        with tf.Session() as sess:


            if pre_trained_path is None:
                if os.path.isdir(model_save_dir):
                    print(
                        "'%s' Exists: Remove the old one and create a new one.\n" %
                        model_save_dir
                    )
                    shutil.rmtree(model_save_dir, ignore_errors=False)
                os.makedirs(model_save_dir, exist_ok=False)

            # Reload weights from directory if specified
            elif pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )
                    saved_model = tf.train.latest_checkpoint(last_save_path)
                elif not os.path.isdir(pre_trained_path) or not saved_model:
                    print(
                        'Warning: ' +
                        'Pre-trained model not Found. ' +
                        'Trying to reset...'
                    )
                else:
                    begin_at_epoch = int(saved_model.split('-')[-1])
                    epoch_num = begin_at_epoch + epoch_num

                    last_saver.restore(sess, saved_model)
                    print('Pre-trained model loaded')
                    print("Restoring parameters from {}".format(saved_model))


            global_epoch = begin_at_epoch
            # For TensorBoard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'train_summaries',
                ),
                graph=sess.graph,
            )
            if self.VALIDATION_RATIO:
                valid_writer = tf.summary.FileWriter(
                    logdir=os.path.join(
                        model_save_dir,
                        'valid_summaries',
                    ),
                    graph=sess.graph,
                )

            batch_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'batch_summaries',
                ),
                graph=sess.graph,
            )


            # Initialize model variables
            sess.run(self.variable_init_op_began)

            for epoch in range(begin_at_epoch, epoch_num):

                # Load the training dataset into the pipeline
                # and initialize the metrics local variables
                sess.run(
                    self.data_init_op,
                    feed_dict={
                        #self._x_input_tensor: input_x,
                        self._x_input_filename_tensor: input_x,
                        self._z_input_tensor: input_z,
                    }
                )
                sess.run(self.metrics_init_op_began)

                if not verbose:
                    epoch_msg = "\rEpoch %d/%d " % (epoch + 1, epoch_num)
                    sys.stdout.flush()
                else:
                    epoch_msg = "Epoch %d/%d \n" % (epoch + 1, epoch_num)
                sys.stdout.write(epoch_msg)

                # BATCH : Optimized by each chunk
                batch_num = 0
                batch_len = int(np.ceil(len(input_x) / batch_size_int))

                if self.VALIDATION_RATIO:
                    valid_len = max(int(batch_len * self.VALIDATION_RATIO), 1)
                    train_len = batch_len - valid_len
                else:
                    train_len = batch_len

                global_epoch = epoch + 1
                # print(global_step)

                batch_remains_ok = True
                while batch_remains_ok and (batch_num <= batch_len):
                    try:
                        for batch in range(train_len):

                            X_batch, Z_batch = sess.run(self.next_batch)

                            #K_val = sess.run(self.K)
                            (_,
                             _,
                             _,
                             summary_str_G_train,
                             summary_str_D_train,
                             summary_learning_rate,
                             global_step) = sess.run(
                                [
                                    self.train_op_G,
                                    self.train_op_D,
                                    self.update_K,
                                    self.summary_op_G,
                                    self.summary_op_D,
                                    self.summary_learning_rate_decay,
                                    self.global_step,
                                ],
                                feed_dict={
                                    self.Input_X_began: X_batch,
                                    self.Input_Z_began: Z_batch,
                                    self.Bool_is_training: True,
                                    self.Input_lambda: lambda_val,
                                    self.Input_gamma: gamma_val,
                                    #self.K: min(max(K_val, 0), 1),
                                    self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                },
                            )
                            err_G = self.loss_G.eval(
                                {
                                    self.Input_Z_began: Z_batch,
                                    self.Bool_is_training: False,
                                }
                            )
                            err_D = self.loss_D.eval(
                                {
                                    self.Input_X_began: X_batch,
                                    self.Input_Z_began: Z_batch,
                                    self.Bool_is_training: False,
                                }
                            )

                            if batch % 10 == 1:
                                summary_G_batch = sess.run(
                                    self.summary_op_G_batch,
                                    feed_dict={
                                        self.Input_Z_began: Z_batch,
                                        self.Bool_is_training: False,
                                        self.Input_lambda: lambda_val,
                                        self.Input_gamma: gamma_val,
                                        #self.K: min(max(K_val, 0), 1),
                                    },
                                )
                                batch_writer.add_summary(
                                    summary_G_batch,
                                    global_step,
                                )

                            sess.run(
                                [
                                    self.update_metrics_op_train_began,
                                    #self.metrics_train_began,
                                ],
                                feed_dict={
                                    self.Input_X_began: X_batch,
                                    self.Input_Z_began: Z_batch,
                                    self.Bool_is_training: False,
                                    self.Input_lambda: lambda_val,
                                    self.Input_gamma: gamma_val,
                                    #self.K: min(max(K_val, 0), 1),
                                    self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                },
                            )
                            # -----------------------------------------------

                            # Write summaries for tensorboard
                            # writer.add_summary(summ, global_step_val)

                            batch_num += 1

                            if verbose:
                                batch_pct = int(20 * batch_num / train_len)
                                batch_bar = "[%s] " % (("#" * batch_pct) + ("-" * (20 - batch_pct)))
                                batch_msg = "\rBatch [%s/%s] " % (batch_num, train_len)
                                batch_err = 'G: %.5s D: %.5s ' % (err_G, err_D)

                                batch_msg = batch_msg + batch_bar + batch_err

                                sys.stdout.flush()
                                sys.stdout.write(batch_msg)

                            # -----------------------------------------------

                        if self.VALIDATION_RATIO:

                            valid_num = 0
                            for valid in range(valid_len):
                                (X_batch,
                                 Z_batch) = sess.run(self.next_batch)
                                (summary_str_G_valid,
                                 summary_str_D_valid,
                                 _) = sess.run(
                                    [
                                        self.summary_op_G,
                                        self.summary_op_D,
                                        self.update_metrics_op_valid_began,
                                        #self.metrics_valid_began,
                                    ],
                                    feed_dict={
                                        self.Input_X_began: X_batch,
                                        self.Input_Z_began: Z_batch,
                                        self.Bool_is_training: False,
                                        self.Input_lambda: lambda_val,
                                        self.Input_gamma: gamma_val,
                                        #self.K: min(max(K_val, 0), 1),
                                    },
                                )
                            # -----------------------------------------------

                                valid_num += 1
                                if verbose:
                                    valid_pct = int(20 * valid_num / valid_len)
                                    valid_bar = "[%s] " % (("#" * valid_pct) + ("-" * (20 - valid_pct)))
                                    valid_msg = "\rValid [%s/%s] " % (valid_num, valid_len)
                                    valid_err = 'G: %.5f D: %.5s ' % (err_G, err_D)

                                    valid_msg = valid_msg + valid_bar + valid_err

                                    sys.stdout.flush()
                                    sys.stdout.write(batch_msg)

                            # -----------------------------------------------

                    except tf.errors.OutOfRangeError:
                        batch_remains_ok = False

                        # result_msg = "finished.\n"
                        # sys.stdout.write(result_msg)
                        continue


                train_writer.add_summary(
                    summary_str_G_train,
                    epoch,
                )
                train_writer.add_summary(
                    summary_str_D_train,
                    epoch,
                )
                train_writer.add_summary(
                    summary_learning_rate,
                    epoch,
                )

                if self.VALIDATION_RATIO:
                    valid_writer.add_summary(
                        summary_str_G_valid,
                        epoch,
                    )

                    valid_writer.add_summary(
                        summary_str_D_valid,
                        epoch,
                    )

                # # Optimize Generator first.
                # gen_train_advantage_step = 50
                # if batch > gen_train_advantage_step:
                #     train_writer.add_summary(
                #         summary_str_D,
                #         epoch,
                #     )
                if verbose:

                    # Metrics
                    metrics_values_train = {k: value[0]
                        for k, value in metrics_train.items()}
                    metrics_values_valid = {k: value[0]
                        for k, value in metrics_valid.items()}
                    metrics_res_train, metrics_res_valid = sess.run([
                        metrics_values_train,
                        metrics_values_valid,
                    ])
                    metrics_res = {
                        **metrics_res_train,
                        **metrics_res_valid,
                    }
                    metrics_str = "\n".join(
                        "{metric_key}: {metric_value:.8f}".format(
                                metric_key=k,
                                metric_value=value,
                        ) for k, value in metrics_res.items()
                    )
                    # print(
                    #     "-- Metrics -- ",
                    #     metrics_str,
                    #     sep='\n',
                    # )
                    sys.stdout.write('\n' + metrics_str + '\n')

                else:

                    if epoch == epoch_num-1:

                        metrics_values_train = {k: value[0]
                            for k, value in metrics_train.items()}
                        metrics_values_valid = {k: value[0]
                            for k, value in metrics_valid.items()}
                        metrics_res_train, metrics_res_valid = sess.run([
                            metrics_values_train,
                            metrics_values_valid,
                        ])
                        metrics_res = {
                            **metrics_res_train,
                            **metrics_res_valid,
                        }
                        metrics_str = "  ".join(
                            "{metric_key}: {metric_value:.5f}".format(
                                    metric_key=k,
                                    metric_value=value,
                            ) for k, value in metrics_res.items()
                        )

                # Save weights
                if model_save_dir:
                    last_save_path = os.path.join(
                        model_save_dir,
                        'last_weights',
                        'after-epoch',
                        #'after-step',
                    )
                    last_saver.save(
                        sess,
                        last_save_path,
                        global_step=epoch,
                    )

        print('\nTraining has been Finished.')
        print(
            "Metrics " +  "=" * 20,
            metrics_str,
            sep='\n',
        )
        print(
            'Latest Model Saved >> %s-%s' %
            (last_save_path, epoch)
        )


    def build_AnoGAN(
        self,
        learning_rate=.0002,
        adam_beta1=.5,
        dropout=.3,
        ano_lambda_=.1,
        use_featuremap=True,
        ):

        reuse_ok =  tf.AUTO_REUSE  # (mode != 'train')
        # class_num = self.CLASS_NUM
        # # batch_size = self.BATCH_SIZE
        #
        # X_batch = self._x_batch_tensor
        # Z_batch = self._z_batch_tensor

        print('\n[anogan_model]: ' + '='*30)

        # Objective Functions ================================================

        with tf.variable_scope('anogan_model', reuse=reuse_ok):

            Input_X_anogan = tf.placeholder(
                self.input_x_dtype,
                self.input_x_shape,
                name='input_x_anogan',
            )
            # Z = tf.get_variable(
            #     name='anomaly_z',
            #     shape=(
            #             Input_X_anogan.get_shape()[0],
            #             self.Z_DIM,
            #     ),  # It must be fully difined & evenly divisible(for reshape).
            #     initializer=tf.random_uniform_initializer(
            #         minval=-1,
            #         maxval=1,
            #     ),
            # )
            Input_1_anogan = tf.placeholder(
                self._z_input_tensor.dtype,
                self._z_input_tensor.get_shape(),
                name='Input_1_anogan',
            )
            ano_Z = self._graph_anomaly_z_distributor(
                input_z=Input_1_anogan,
                name='anomaly_z_distributor',
            )

        with tf.variable_scope('began_model', reuse=True):
            ano_G = self._graph_sampler(
                z=ano_Z,
                filter_size=5,
                stride_size=2,
                dropout=dropout,
                generator_name='generator',
            )

        with tf.variable_scope('anogan_model', reuse=reuse_ok):
            with tf.name_scope('loss_residual'):
                loss_residual = tf.reduce_mean(
                    tf.abs(
                        tf.subtract(Input_X_anogan, ano_G)
                    )
                )

        if use_featuremap:

            with tf.variable_scope('began_model', reuse=True):
                feature_D_from_X = self._graph_discriminator_featuremap(
                    Input_X_anogan,
                    filter_size=5,
                    stride_size=2,
                    dropout=dropout,
                    discriminator_name='discriminator',
                    return_layer_idx=3,
                )
                feature_D_from_Z= self._graph_discriminator_featuremap(
                    ano_G,
                    filter_size=5,
                    stride_size=2,
                    dropout=dropout,
                    discriminator_name='discriminator',
                    return_layer_idx=3,
                )
                feature_D_from_X = tf.stop_gradient(feature_D_from_X)
                feature_D_from_Z = tf.stop_gradient(feature_D_from_Z)

            with tf.variable_scope('anogan_model', reuse=reuse_ok):
                with tf.name_scope('loss_discrimination'):
                    loss_discrimination = tf.reduce_mean(
                        tf.abs(
                            tf.subtract(
                                feature_D_from_X,
                                feature_D_from_Z,
                            )
                        )
                    )

        else:

            with tf.variable_scope('began_model', reuse=True):
                test_D = self._graph_discriminator(
                    input_x=ano_G,
                    filter_size=5,
                    stride_size=2,
                    dropout=dropout,
                    is_training=False,
                    name='discriminator',
                    return_all_layers=False,
                )
            with tf.variable_scope('anogan_model', reuse=reuse_ok):
                with tf.name_scope('loss_discrimination'):
                    loss_discrimination = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=test_D,
                            labels=tf.ones_like(test_D),
                        )
                    )

        with tf.variable_scope('anogan_model', reuse=reuse_ok):
            with tf.name_scope('anomaly_score'):
                anomaly_score = (
                    ((1 - ano_lambda_) * loss_residual) +
                     (ano_lambda_ * loss_discrimination)
                )

            summary_Z = tf.summary.histogram(
                'anomaly_z',
                ano_Z,
            )
            summary_loss_residual = tf.summary.scalar(
                'loss_residual',
                loss_residual,
            )
            summary_loss_discrimination = tf.summary.scalar(
                'loss_discrimination',
                loss_discrimination,
            )
            summary_anomaly_score = tf.summary.scalar(
                'anomaly_score',
                anomaly_score,
            )
        # ====================================================================


        # Optimization =======================================================

        with tf.name_scope('anogan_optimization'):

            optimizer_Z = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=adam_beta1,
                beta2=0.999,
                epsilon=1e-08,
                name='optimizer_anomaly_detector',
            )

            train_op_Z = optimizer_Z.minimize(
                anomaly_score,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="anogan_model",
                )
            )

            summary_op_Z = tf.summary.merge([
                summary_Z,
                summary_loss_residual,
                summary_loss_discrimination,
                summary_anomaly_score,
            ])

        # ====================================================================

        # variable_init_op_began = tf.group(*[tf.global_variables_initializer(),
        #                               tf.tables_initializer()])
        variable_anogan = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='anogan_',
        )
        variable_init_op_anogan = tf.group(
            *[
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='anogan_model',
                    )
                ),
                tf.variables_initializer(
                    var_list = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='anogan_optimization',
                    )
                ),
            ],
        )

        # ====================================================================

        with tf.variable_scope("metrics_anogan", reuse=reuse_ok):
            metrics_train_anogan = {
                'Train_loss_residual': tf.metrics.mean(loss_residual),
                'Train_loss_discrimination': tf.metrics.mean(
                    loss_discrimination
                ),
                'Train_anomaly_score': tf.metrics.mean(anomaly_score),
            }
            metrics_valid_anogan = {
                'Valid_loss_residual': tf.metrics.mean(loss_residual),
                'Valid_loss_discrimination': tf.metrics.mean(
                    loss_discrimination
                ),
                'Valid_anomaly_score': tf.metrics.mean(anomaly_score),
            }

        # Group the update ops for the tf.metrics
        update_metrics_op_train_anogan = tf.group(
            *[op for _, op in metrics_train_anogan.values()]
        )
        update_metrics_op_valid_anogan = tf.group(
            *[op for _, op in metrics_valid_anogan.values()]
        )

        # Get the op to reset the local variables used in tf.metrics
        metrics_init_op_anogan = tf.variables_initializer(
            var_list=tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="metrics_anogan",
            ),
            name='metrics_init_op_anogan',
        )

        # Return
        self.variable_anogan = variable_anogan
        self.variable_init_op_anogan = variable_init_op_anogan

        self.train_op_Z = train_op_Z

        self.metrics_train_anogan = metrics_train_anogan
        self.metrics_valid_anogan = metrics_valid_anogan
        self.update_metrics_op_train_anogan = update_metrics_op_train_anogan
        self.update_metrics_op_valid_anogan = update_metrics_op_valid_anogan
        self.metrics_init_op_anogan = metrics_init_op_anogan

        self.summary_op_Z = summary_op_Z

        self.ano_Z = ano_Z
        self.ano_G = ano_G
        self.loss_residual = loss_residual
        self.loss_discrimination = loss_discrimination
        self.anomaly_score = anomaly_score
        self.Input_X_anogan = Input_X_anogan
        self.Input_1_anogan = Input_1_anogan


    def train_AnoGAN(
        self,
        input_x=None,
        input_z=None,
        epoch_num=2,
        model_save_dir='./model_save_anogan',
        pre_trained_path=None,
        verbose=False,
        writer=None,
        ):
        """Short Description.

        TBD

        Attributes
        ----------
        TBD


        Example
        -------
        >>> "TBD"
        >>> train_fn(x_train, y_train, train_model_spec,
                 3, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path=None,
                 verbose=True)

        >>> train_fn(x_train, y_train, train_model_spec,
                 10, None, hyper_param,
                 model_save_dir='model_save', pre_trained_path='model_save')

        """

        metrics_train = self.metrics_train_anogan
        metrics_valid = self.metrics_valid_anogan
        # --------------------------------------------------------------------
        #global_step = tf.train.get_global_step()

        try:
            if not self.VALIDATION_RATIO or not 0 < self.VALIDATION_RATIO < 1:
                raise NoValidValueException(self.VALIDATION_RATIO)
            else:
                print("Training with " +
                      "'VALIDATION_RATIO=%s'" % self.VALIDATION_RATIO)
        except KeyError:
            raise NoValidValueException(None)
        except NoValidValueException as err:
            print(err)
            self.VALIDATION_RATIO = 0

        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver(
            var_list=self.variable_anogan,
            max_to_keep=2,  # will keep last 5 epochs as default
        )
        begin_at_epoch = 0
        os.makedirs(model_save_dir, exist_ok=True)

        with tf.Session() as sess:

            # For TensorBoard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'train_summaries',
                ),
                graph=sess.graph,
            )
            # Initialize model variables
            sess.run(self.variable_init_op_began)
            sess.run(self.variable_init_op_anogan)

            # Reload weights from directory if specified
            if pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )
                    saved_model = tf.train.latest_checkpoint(last_save_path)
                    begin_at_epoch = int(saved_model.split('-')[-1])
                    epoch_num = begin_at_epoch + epoch_num
                last_saver.restore(sess, saved_model)
                print('Pre-trained model loaded')

            if verbose:

                for epoch in range(begin_at_epoch, epoch_num):

                    # Load the training dataset into the pipeline
                    # and initialize the metrics local variables
                    sess.run(
                        self.data_init_op,
                        feed_dict={
                            self._x_input_tensor: input_x,
                            self._z_input_tensor: input_z,
                        }
                    )
                    sess.run(self.metrics_init_op_anogan)
                    epoch_msg = "Epoch %d/%d\n" % (epoch + 1, epoch_num)
                    sys.stdout.write(epoch_msg)

                    # BATCH : Optimized by each chunk
                    batch_num = 0
                    batch_len = int(np.ceil(len(input_x) / self.BATCH_SIZE))

                    if self.VALIDATION_RATIO:
                        valid_len = int(batch_len * self.VALIDATION_RATIO)
                        train_len = batch_len - valid_len
                    else:
                        train_len = batch_len

                    batch_remains_ok = True
                    while batch_remains_ok:
                        try:
                            for batch in range(train_len):

                                X_batch, Z_one_batch = sess.run(self.next_batch)
                                _, summary_str_Z = sess.run(
                                    [
                                        self.train_op_Z,
                                        self.summary_op_Z,
                                    ],
                                    feed_dict={
                                        self.Input_X_anogan: X_batch,
                                        self.Input_1_anogan: Z_one_batch,
                                    },
                                )
                                sess.run(
                                    self.update_metrics_op_train_anogan,
                                    feed_dict={
                                        self.Input_X_anogan: X_batch,
                                        self.Input_1_anogan: Z_one_batch,
                                    },
                                    )
                                # -----------------------------------------------

                                # Write summaries for tensorboard
                                # writer.add_summary(summ, global_step_val)

                                batch_num += 1
                                batch_pct = int(20 * batch_num / train_len)
                                batch_bar = "[%s] " % (("#" * batch_pct) + ("-" * (20 - batch_pct)))
                                batch_msg = "\rBatch [%s/%s] " % (batch_num, train_len)

                                batch_msg = batch_msg + batch_bar

                                sys.stdout.flush()
                                sys.stdout.write(batch_msg)
                                # -----------------------------------------------

                            if self.VALIDATION_RATIO:

                                for valid in range(valid_len):
                                    sess.run(
                                        self.update_metrics_op_valid_anogan,
                                        feed_dict={
                                            self.Input_X_anogan: X_batch,
                                            self.Input_1_anogan: Z_one_batch,
                                        },
                                        )

                                # -----------------------------------------------

                        except tf.errors.OutOfRangeError:
                            batch_remains_ok = False

                            result_msg = "finished.\n"
                            sys.stdout.write(result_msg)
                            continue

                    train_writer.add_summary(
                        summary_str_Z,
                        epoch,
                    )

                    metrics_values_train = {k: value[0]
                        for k, value in metrics_train.items()}
                    metrics_values_valid = {k: value[0]
                        for k, value in metrics_valid.items()}
                    metrics_res_train, metrics_res_valid = sess.run([
                        metrics_values_train,
                        metrics_values_valid,
                    ])
                    metrics_res = {
                        **metrics_res_train,
                        **metrics_res_valid,
                    }
                    metrics_str = "\n".join(
                        "{metric_key}: {metric_value:05.5f}".format(
                                metric_key=k,
                                metric_value=value,
                        ) for k, value in metrics_res.items()
                    )
                    print(
                        "-- Metrics -- ",
                        metrics_str,
                        sep='\n',
                    )

                    # Save weights
                    if model_save_dir:
                        last_save_path = os.path.join(
                            model_save_dir,
                            'last_weights',
                            'after-epoch',
                        )
                        last_saver.save(
                            sess,
                            last_save_path,
                            global_step=epoch + 1,
                        )
                        print('Model Saved: %s' % last_save_path)

            else:

                raise Exception('`verbose` option is deprecated. ' +
                                ' set `verbose=True`.')

            self.EPOCH_NUM = epoch_num
            print("Training Finished!")


    def evaluate_BEGAN(
        self,
        #input_x=None,
        input_z=None,
        pre_trained_path='./model_save/began',
        target_epoch=None,
        ):
        """Short Description.

        TBD

        Attributes
        ----------
        TBD


        Example
        -------
        >>> "TBD"
        >>> evaluate_fn(x_test, y_test, eval_model_spec,
                    hyper_param, pre_trained_path='./model_save')

        """
        dropout = None
        assert pre_trained_path is not None, "`pre_trained_path` is mandatory."
        #global_step = tf.train.get_global_step()

        if self.G is None:
            tf.reset_default_graph()
            self.build_began(
                learning_rate = self.LEARNING_RATE,
                adam_beta1 = self.ADAM_BETA1,
                dropout=self.DROPOUT,
            )

        with tf.Session() as sess:
            ## Initialize model variables
            sess.run(self.variable_init_op_began)

            # Reload weights from directory if specified
            if pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(pre_trained_path, 'last_weights')

                    if target_epoch:
                        saved_model = ''.join(
                            [
                                last_save_path + '/',
                                'after-epoch-',
                                str(target_epoch),
                            ],
                        )
                        last_saver = tf.train.import_meta_graph(
                            saved_model + '.meta'
                        )
                    else:
                        last_saver = tf.train.Saver()
                        saved_model = tf.train.latest_checkpoint(
                            last_save_path,
                        )
                    #begin_at_epoch = int(saved_model.split('-')[-1])
                    last_saver.restore(sess, saved_model)

            pred = sess.run(
                self.G,
                feed_dict={
                    self.Input_Z_began: input_z,
                    self.Bool_is_training: False,
                }
            )

            pred_img = (pred + 1.) / 2. * 255

        return pred_img
