import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import importlib
import workspaces.src.anogan.anogan_ops as ano_op
importlib.reload(ano_op)


class NoValidationException(Exception):
    def __init__(self, valid_value):
        self.valid_value = str(valid_value)
    def __str__(self):
        msg = "Training with NO-VALIDATION: 'VALIDATION_RATIO=%s'"
        return repr(msg % self.valid_value)

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
class AnoGAN(NeuralNetworkModel):
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
        input_x_dtype=tf.float32,
        input_z_dtype=tf.float32,
        input_x_shape=(None, 108, 108, 3),
        input_z_shape=(None, 100),
        use_gpu=True,
        # input_width=108,
        # input_height=108,
        # input_channel=3,
        # output_width=64,
        # output_height=64,
        # output_channel=None,
        # class_num=None,
        # z_dim=100,
        g_filter_dim=64,
        d_filter_dim=64,
        g_fc_dim=1024,
        d_fc_dim=1024,
        batch_size=64,
        dropout=0,
        lr_decaying=True,
        buffer_size=1000,
        learning_rate=.0002,
        adam_beta1=.5,
        adam_beta2=.5,
        validation_ratio=.2,
        discriminator_training_ratio=.5,
        ano_lambda_=.1,
        use_featuremap=True,
        decay_lr_step=100,
        ):

        super().__init__()
        self.USE_GPU = use_gpu

        self.input_x_dtype = input_x_dtype
        self.input_z_dtype = input_z_dtype

        self.input_x_shape = input_x_shape
        self.input_z_shape = input_z_shape

        _, self.Z_DIM = self.input_z_shape        # Random Variable Z Dimension
        self.CLASS_NUM = None

        assert g_filter_dim == d_filter_dim
        assert g_fc_dim == d_fc_dim
        self.G_FILTER_DIM = g_filter_dim
        self.D_FILTER_DIM = d_filter_dim
        self.G_FC_DIM = g_fc_dim
        self.D_FC_DIM = d_fc_dim

        self.BATCH_SIZE = batch_size              # Training Batch Size
        self.DROPOUT = dropout
        self.BUFFER_SIZE = buffer_size            # For tf.Dataset.suffle(buffer_size)
        self.LEARNING_RATE = learning_rate        # Learning rate (Fixed for now)
        self.ADAM_BETA1 = adam_beta1
        self.ADAM_BETA2 = adam_beta2
        self.VALIDATION_RATIO = validation_ratio  # Validation Ratio
        self.DISCRIMINATOR_TRAINING_RATIO = discriminator_training_ratio
        self.EPOCH_NUM = 0                        # Cumulative Epoch Number
        self.LR_DECAYING = lr_decaying
        self.DECAY_LR_STEP = decay_lr_step


        self.LEARNING_RATE_ANO = learning_rate        # Learning rate (Fixed for now)
        self.ADAM_BETA1_ANO = adam_beta1
        self.ADAM_BETA2_ANO = adam_beta2
        self.ANO_LAMBDA_ = ano_lambda_
        self.USE_FEATUREMAP = use_featuremap
        self.LR_DECAYING_ANO = lr_decaying
        self.DECAY_LR_STEP_ANO = decay_lr_step

        self._x_input_tensor = None
        self._z_input_tensor = None
        self._x_batch_tensor = None
        self._z_batch_tensor = None
        self.data_init_op = None
        self.data_init_op_eval = None

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

        #--------------------------sohee_edit_version

        self.g_bn0 = ano_op.batch_norm(name = 'g_bn0')
        self.g_bn1 = ano_op.batch_norm(name = 'g_bn1')
        self.g_bn2 = ano_op.batch_norm(name = 'g_bn2')
        self.g_bn3 = ano_op.batch_norm(name = 'g_bn3')

        self.d_bn0 = ano_op.batch_norm(name = 'd_bn0')
        self.d_bn1 = ano_op.batch_norm(name = 'd_bn1')
        self.d_bn2 = ano_op.batch_norm(name = 'd_bn2')
        self.d_bn3 = ano_op.batch_norm(name = 'd_bn3')

        #--------------------------------------------

        self.input(
            input_x_dtype=self.input_x_dtype,
            input_z_dtype=self.input_z_dtype,
            input_x_shape=self.input_x_shape,
            input_z_shape=self.input_z_shape,
            is_training=True,
            drop_remainder=False,
        )

        self.build_DCGAN(
            lr_decaying=True,
            learning_rate = self.LEARNING_RATE,
            decay_lr_step = self.DECAY_LR_STEP,
            adam_beta1 = self.ADAM_BETA1,
            adam_beta2 = self.ADAM_BETA2,
            dropout=self.DROPOUT,
        )


        self.build_AnoGAN(
            lr_decaying=True,
            learning_rate = self.LEARNING_RATE_ANO,
            decay_lr_step = self.DECAY_LR_STEP_ANO,
            adam_beta1 = self.ADAM_BETA1_ANO,
            adam_beta2 = self.ADAM_BETA2_ANO,
            dropout=self.DROPOUT,
            ano_lambda_=self.ANO_LAMBDA_,
            use_featuremap=self.USE_FEATUREMAP,
        )


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

        parameter_str = "\n".join(
            f"| {prmt_key:18s}: {str(prmt_value):>8s} |"
            #if isinstance(prmt_value, (int, float, complex))
            #else f"{prmt_key:15s}\t: {prmt_value:>8s}"
            for prmt_key, prmt_value in parameter_dict.items()
        )
        print(
            "=" * 7 + " Given Parameters " +  "=" * 7,
            parameter_str,
            "=" * 32,
            sep='\n',
        )

    def _img_parse_fn(
        self,
        filename,
        data_z,
        return_image_only=False,
        ):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [64, 64])
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
        )

        if return_image_only:
            return image_scaled
        else:
            return image_scaled, data_z


    def input(
        self,
        input_x_dtype=tf.float32,
        input_z_dtype=tf.float32,
        input_x_shape=(None, 8, 8, 3),
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
            buffer_size = self.BUFFER_SIZE if is_training else 1
            #
            # assert self.INPUT_WIDTH == self.INPUT_HEIGHT
            #

            X_t = tf.placeholder(self.input_x_dtype, self.input_x_shape,
                                 name='x_tensor_interface')
            # X_t= tf.placeholder(tf.string, (None,),
            #                       name='x_file_interface')
            Z_t = tf.placeholder(self.input_z_dtype, self.input_z_shape,
                                 name='z_tensor_interface')

            dataset = tf.data.Dataset.from_tensor_slices((X_t, Z_t))
            #dataset = dataset.map(self._img_parse_fn)
            dataset = dataset.shuffle(
                buffer_size=buffer_size,
            )  # reshuffle_each_iteration=True as default.
            if is_training:
                dataset = dataset.batch(
                            self.BATCH_SIZE,
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


        self._x_input_tensor = X_t
        #self._x_input_filename_tensor = X_file_t
        self._z_input_tensor = Z_t
        self._x_batch_tensor = X_batch
        self._z_batch_tensor = Z_batch
        self.next_batch = next_batch
        self.data_init_op = data_init_op


    def _graph_generator(
        self,
        z,
        is_training=True,
        dropout=.3,
        reuse=tf.AUTO_REUSE,
        name='generator',
        ):

        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):

            # DeConvolutional Layer Size
            height_gen_x, width_gen_x = self.input_x_shape[1], self.input_x_shape[2] #(8,8)
            height_2, width_2 = int(height_gen_x/2), int(width_gen_x/2)
            #input_shape/2 #(4,4)
            height_1, width_1 = int(height_2/2), int(width_2/2)
            #input_shape/4 #(2,2)
            height_0, width_0 = int(height_1/2), int(width_1/2)
            #input_shape/8 #(1,1)

            # channel_3 = self.G_FILTER_DIM * (2 ** 0)  # 64 G_FILTER_DIM*1
            channel_2 = self.G_FILTER_DIM * (2 ** 1)  # 128 G_FILTER_DIM*2
            channel_1 = self.G_FILTER_DIM * (2 ** 2)  # 256 G_FILTER_DIM*4
            channel_0 = self.G_FILTER_DIM * (2 ** 3)  # 512 G_FILTER_DIM*8


            # projection_size = height_0 * width_0 * self.G_FILTER_DIM * 8
            projection_size = height_0 * width_0 * channel_0

            ##### conv 영역 ------------------
            proj_z = ano_op.linear(
                        z,
                        projection_size,
                        'proj_z',
                        with_w = False,
            )

            reshaped_proj_z = tf.reshape(
                                proj_z,
                                [-1, height_0, width_0, channel_0],
            )
            deconv2d_ = tf.nn.relu(
                            self.g_bn0(
                                reshaped_proj_z,
                            )
            )

            ##### deconv2d_0 영역 -------------
            deconv2d_0_pre  = ano_op.deconv2d(
                                deconv2d_,
                                output_shape=(None, height_1, width_1, channel_1),
                                name='deconv2d_0',
                                with_w=False,
                        )
            deconv2d_0 = tf.nn.relu(
                            self.g_bn1(
                                deconv2d_0_pre,
                            )
            )

            ##### deconv2d_1 영역 -------------
            deconv2d_1_pre = ano_op.deconv2d(
                    deconv2d_0,
                    [None, height_2, width_2, channel_2],
                    name='deconv2d_1',
                    with_w=False,
            )
            deconv2d_1 = tf.nn.relu(
                            self.g_bn2(
                                deconv2d_1_pre,
                            )
            )

            ##### generated_x 영역 -------------
            generated_x_pre = ano_op.deconv2d(
                        deconv2d_1,
                        self.input_x_shape, name='generated_x',
                        with_w=False,
            )


            generated_x = tf.nn.tanh(
                            generated_x_pre,
            )

            ##### print 영역 ------------------

            self._print_layer(
                name=proj_z.name.split('/')[-1],
                input_shape=z.get_shape(),
                output_shape=proj_z.get_shape(),
            )
            self._print_layer(
                name=reshaped_proj_z.name.split('/')[-1],
                input_shape=proj_z.get_shape(),
                output_shape=reshaped_proj_z.get_shape(),
            )
            self._print_layer(
                name=deconv2d_.name.split('/')[-1],
                input_shape=reshaped_proj_z.get_shape(),
                output_shape=deconv2d_.get_shape(),
            )
            self._print_layer(
                name=deconv2d_0_pre.name.split('/')[2],
                input_shape=deconv2d_.get_shape(),
                output_shape=deconv2d_0_pre.get_shape(),
            )
            self._print_layer(
                name=deconv2d_1_pre.name.split('/')[2],
                input_shape=deconv2d_0.get_shape(),
                output_shape=deconv2d_1_pre.get_shape(),
            )
            self._print_layer(
                name=generated_x_pre.name.split('/')[2],
                input_shape=deconv2d_1.get_shape(),
                output_shape=generated_x_pre.get_shape(),
            )

        return generated_x


    def _graph_discriminator(
        self,
        input_x,
        is_training=True,
        dropout=.3,
        reuse=tf.AUTO_REUSE,
        name='discriminator',
        return_all_layers=False,
        ):

        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):

            # Convolutional Layer Size

            channel_0 = self.D_FILTER_DIM * (2 ** 0)  # 64 D_FILTER_DIM*1
            channel_1 = self.D_FILTER_DIM * (2 ** 1)  # 128 D_FILTER_DIM*2
            channel_2 = self.D_FILTER_DIM * (2 ** 2)  # 512 D_FILTER_DIM*4
            channel_3 = self.D_FILTER_DIM * (2 ** 3)  # 1024 D_FILTER_DIM*8

            ##### conv 영역 ------------------

            conv2d_0 = ano_op.lrelu(
                ano_op.conv2d(
                    input_x,
                    channel_1,
                    name ='conv2d_0',
                )
            )
            conv2d_1 = ano_op.lrelu(
                self.d_bn0(
                    ano_op.conv2d(
                        conv2d_0,
                        channel_2,
                        name = 'conv2d_1',
                    )
                )
            )

            conv2d_2 = ano_op.lrelu(
                self.d_bn1(
                    ano_op.conv2d(
                        conv2d_1,
                        channel_3,
                        name = 'conv2d_2',
                    )
                )
            )
            reshaped_proj_y=tf.reshape(
                    conv2d_2,
                    [-1, channel_3],
            )
            y = ano_op.linear(
                reshaped_proj_y,
                1,
                'reshaped_Y_lin',
            )
            activated_y = tf.nn.sigmoid(
                y,
                name='discriminator_sigmoid',
            )

            ##### print 영역 ------------------
            print(
                input_x.name.split('/')[-1],
                ': ',
                input_x.get_shape(),
                '\n',

                conv2d_0.name.split('/')[1],
                'conv2d_0 : ',
                conv2d_0.get_shape(),
                '\n',

                conv2d_1.name.split('/')[1],
                'conv2d_1 : ',
                conv2d_1.get_shape(),
                '\n',

                conv2d_2.name.split('/')[1],
                'conv2d_2 : ',
                conv2d_2.get_shape(),
                '\n',

                reshaped_proj_y.name.split('/')[-1],
                ': ',
                reshaped_proj_y.get_shape(),
                '\n',

                y.name.split('/')[-1],
                y.get_shape(),
                '\n',

                activated_y.name.split('/')[-1],
                activated_y.get_shape(),
            )
            ##### return 영역 ------------------

            if return_all_layers:
                return (
                    conv2d_0,
                    conv2d_1,
                    conv2d_2,
                    reshaped_proj_y,
                    y,
                    activated_y,
                )
            else:
                return activated_y, y


    def _graph_sampler(
        self,
        z,
        is_training=False,
        dropout=.3,
        name='sampler',
        generator_name='generator',
        ):

        with tf.name_scope(name):
            print(name + ' ' + '-'*20)
            sample_G = self._graph_generator(
                z,
                is_training=is_training,
                reuse=True,
                dropout=dropout,
                name=generator_name,
            )
        return sample_G


    def _graph_discriminator_featuremap(
        self,
        input_x,
        is_training=False,
        dropout=.3,
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
        with tf.name_scope(name):
            print(name + ' ' + '-'*20)
            layers = self._graph_discriminator(
                input_x,
                is_training=is_training,
                dropout=dropout,
                reuse=True,
                name=discriminator_name,
                return_all_layers=True,
            )
            #layers = [tf.stop_gradient(layer) for layer in layers]

        return layers[return_layer_idx]
#------------------------------------------

    def _graph_anomaly_z_distributor(
        self,
        input_z,
        is_training=True,
        name='anomaly_z_distributor',
        ):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            anomaly_z = ano_op.linear(
                    input_z,
                    input_z.get_shape()[-1],
                    scope='anomaly_z',
                    #stddev=0.02,
                    #bias_start=0.0,
                    with_w=False,
            )
            return anomaly_z


    def build_DCGAN(
        self,
        lr_decaying=True,
        learning_rate=.0002,
        adam_beta1=.5,
        adam_beta2=.5,
        dropout=.3,
        decay_lr_step=200,
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

        print('\n[dcgan_model]: ' + '='*30)

        # Objective Functions ================================================

        with tf.variable_scope('dcgan_model', reuse=reuse_ok):

            # Input_X_dcgan = tf.placeholder(
            #     self._x_input_tensor.dtype,
            #     self._x_input_tensor.get_shape(),
            #     name='input_x_dcgan',
            # )
            Input_X_dcgan = tf.placeholder(
                self.input_x_dtype,
                self.input_x_shape,
                name='input_x_dcgan',
            )
            Input_Z_dcgan = tf.placeholder(
                self._z_input_tensor.dtype,
                self._z_input_tensor.get_shape(),
                name='input_z_dcgan',
            )
            Bool_is_training = tf.placeholder(
                tf.bool,
                #[None, 1],
                name='Bool_is_training',
            )
            Start_learning_rate_tensor = tf.placeholder(
                tf.float32,
                (None),
                name='start_learning_rate_tensor',
            )
            Decay_lr_step_tensor = tf.placeholder(
                tf.int32,
                (None),
                name='Decay_lr_step_tensor',
            )
            G = self._graph_generator(
                z=Input_Z_dcgan,  # z=Z_batch,
                dropout=dropout,
                is_training=Bool_is_training,
                name='generator',
            )
            D_real_sigmoid_y, D_real_y = self._graph_discriminator(
                input_x=Input_X_dcgan,  # input_x=X_batch,
                dropout=dropout,
                is_training=Bool_is_training,
                reuse=tf.AUTO_REUSE,
                name='discriminator',
                return_all_layers=False,
            )
            D_fake_sigmoid_y, D_fake_y = self._graph_discriminator(
                input_x=G,
                dropout=dropout,
                is_training=Bool_is_training,
                reuse=True,
                name='discriminator',
                return_all_layers=False,
            )

            summary_Z = tf.summary.histogram('z', Input_Z_dcgan)
            summary_G = tf.summary.histogram('G', G)
            summary_D_real = tf.summary.histogram('D_real', D_real_sigmoid_y)
            summary_D_fake = tf.summary.histogram('D_fake', D_fake_sigmoid_y)

            with tf.name_scope('loss_G_scope'):
                loss_G = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_fake_y,
                        labels=tf.ones_like(
                            D_fake_sigmoid_y,
                            name='D_fake_as_correct'
                        ),
                        name='generator_correct_score',
                    )
                )

            with tf.name_scope('loss_D_scope'):
                loss_D_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_real_y,
                        labels=tf.ones_like(
                            D_real_sigmoid_y,
                            name='D_real_as_correct'
                        ),
                        name='discriminator_correct_score',
                    )
                )
                loss_D_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_fake_y,
                        labels=tf.zeros_like(
                            D_fake_sigmoid_y,
                            name='D_real_as_correct'
                        ),
                        name='discriminator_fault_score',
                    )
                )
                loss_D = tf.add(
                    loss_D_real,
                    loss_D_fake,
                    name='discriminator_true_score',
                )

            summary_loss_G = tf.summary.scalar('loss_G', loss_G)
            summary_loss_D_real = tf.summary.scalar('loss_D_real', loss_D_real)
            summary_loss_D_fake = tf.summary.scalar('loss_D_fake', loss_D_fake)
            summary_loss_D = tf.summary.scalar('loss_D', loss_D)

        # Summaries for training
        summary_op_G = tf.summary.merge([
            summary_Z,
            summary_G,
            summary_loss_G,
        ])
        summary_op_D = tf.summary.merge([
            summary_Z,
            summary_D_real,
            summary_D_fake,
            summary_loss_D_real,
            summary_loss_D_fake,
            summary_loss_D,
        ])
        # ====================================================================


        # Optimization =======================================================

        with tf.variable_scope('dcgan_optimization'):

            global_step_G = tf.Variable(
                0,
                trainable=False,
                name='global_step_G',
            )
            global_step_D = tf.Variable(
                0,
                trainable=False,
                name='global_step_D',
            )

            global_step_G = global_step_G.assign(global_step_D)

            if lr_decaying:
                learning_rate_decay = tf.train.exponential_decay(
                    learning_rate=Start_learning_rate_tensor,
                    global_step=global_step_G,
                    # decay_steps=100000,
                    # decay_rate=.96,
                    # staircase=True,
                    decay_steps=Decay_lr_step_tensor,
                    decay_rate=.95,
                    staircase=True,
                    name='learning_rate_decay',
                )
            else:
                learning_rate_decay = Start_learning_rate_tensor

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
                learning_rate=learning_rate_decay,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=1e-08,
                name='optimizer_discriminator',
            )

            train_op_G = optimizer_G.minimize(
                loss_G,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="dcgan_model/generator",
                ),
                global_step=global_step_G,
            )
            train_op_D = optimizer_D.minimize(
                loss_D,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="dcgan_model/discriminator",
                ),
                global_step=global_step_D,
            )

        # variable_init_op_dcgan = tf.group(*[tf.global_variables_initializer(),
        #                               tf.tables_initializer()])
        variable_dcgan = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='dcgan_model',
        )

        var_init_op_dcgan = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='dcgan_model',
            )
        )
        opt_init_op_dcgan = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='dcgan_optimization',
            )
        )
        variable_init_op_dcgan = tf.group(
            *[var_init_op_dcgan, opt_init_op_dcgan]
        )

        # ====================================================================

        with tf.variable_scope("dcgan_metrics", reuse=reuse_ok):
            metrics_train_dcgan = {
                'Train_loss_G': tf.metrics.mean(loss_G),
                'Train_loss_D': tf.metrics.mean(loss_D),
                'Learning_rate': tf.metrics.mean(learning_rate_decay),
            }
            metrics_valid_dcgan = {
                'Valid_loss_G': tf.metrics.mean(loss_G),
                'Valid_loss_D': tf.metrics.mean(loss_D),
            }

        # Group the update ops for the tf.metrics
        update_metrics_op_train_dcgan = tf.group(
            *[op for _, op in metrics_train_dcgan.values()]
        )
        update_metrics_op_valid_dcgan = tf.group(
            *[op for _, op in metrics_valid_dcgan.values()]
        )

        # Get the op to reset the local variables used in tf.metrics
        metrics_init_op_dcgan = tf.variables_initializer(
            var_list=tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="dcgan_metrics",
            ),
            name='metrics_init_op_dcgan',
        )

        # Return
        self.variable_dcgan = variable_dcgan
        self.var_init_op_dcgan = var_init_op_dcgan
        self.opt_init_op_dcgan = opt_init_op_dcgan
        self.variable_init_op_dcgan = variable_init_op_dcgan

        self.train_op_G = train_op_G
        self.train_op_D = train_op_D

        self.metrics_train_dcgan = metrics_train_dcgan
        self.metrics_valid_dcgan = metrics_valid_dcgan
        self.update_metrics_op_train_dcgan = update_metrics_op_train_dcgan
        self.update_metrics_op_valid_dcgan = update_metrics_op_valid_dcgan
        self.metrics_init_op_dcgan = metrics_init_op_dcgan

        self.summary_op_G = summary_op_G
        self.summary_op_D = summary_op_D
        self.summary_learning_rate_decay = summary_learning_rate_decay

        self.Input_X_dcgan = Input_X_dcgan
        self.Input_Z_dcgan = Input_Z_dcgan
        self.Bool_is_training = Bool_is_training
        self.Start_learning_rate_tensor = Start_learning_rate_tensor
        self.Decay_lr_step_tensor = Decay_lr_step_tensor

        self.global_step_G = global_step_G
        self.global_step_D = global_step_D
        self.G = G
        # self.S = S
        self.D_real = D_real_sigmoid_y
        self.D_fake = D_fake_sigmoid_y
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.loss_D_fake = loss_D_fake
        self.loss_D_real = loss_D_real


    def train_DCGAN(
        self,
        input_x=None,
        #input_x_filenames=None,
        input_z=None,
        adam_beta1=.5,
        adam_beta2=.5,
        batch_size=128,
        learning_rate=0.001,
        decay_lr_step=200,
        # lambda_val=.001,
        # gamma_val=.7,
        epoch_num=25,
        validation_ratio=.0,
        gen_train_advantage_ratio=.5,
        gen_train_n_times=2,
        model_save_dir='./model_save_dcgan',
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
        metrics_train = self.metrics_train_dcgan
        metrics_valid = self.metrics_valid_dcgan
        # --------------------------------------------------------------------
        #global_step = tf.train.get_global_step()
        self.LEARNING_RATE = learning_rate
        self.VALIDATION_RATIO = validation_ratio
        self.ADAM_BETA1 = adam_beta1
        self.ADAM_BETA2 = adam_beta2
        self.BATCH_SIZE = batch_size
        self.DECAY_LR_STEP = decay_lr_step


        try:
            if not self.VALIDATION_RATIO or not (0 < self.VALIDATION_RATIO < 1):
                raise NoValidationException(self.VALIDATION_RATIO)
            else:
                print("Training with " +
                      "'VALIDATION_RATIO=%s'" % self.VALIDATION_RATIO)
        except KeyError:
            raise NoValidationException(None)
        except NoValidationException as err:
            print(err)
            self.VALIDATION_RATIO = 0

        try:
            if not gen_train_advantage_ratio or not 0 < gen_train_advantage_ratio < 1:
                raise NoValidationException(gen_train_advantage_ratio)
            else:
                print("Training with " +
                      "'GENERATOR Training Advantage Ratio=%s'" % gen_train_advantage_ratio)
        except KeyError:
            raise NoValidationException(None)
        except NoValidationException as err:
            print(err)
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
            'DECAY_LR_STEP': self.DECAY_LR_STEP,
            'BATCH_SIZE': batch_size_int,
            'EPOCH_NUM': epoch_num,
            # 'LAMBDA': lambda_val,
            # 'GAMMA': gamma_val,
            'ADAM_BETA1': self.ADAM_BETA1,
            'ADAM_BETA2': self.ADAM_BETA2,
        }
        self._print_parameter(parameter_dict)

        self.input(
            input_x_dtype=self.input_x_dtype,
            input_z_dtype=self.input_z_dtype,
            input_x_shape=self.input_x_shape,
            input_z_shape=self.input_z_shape,
            is_training=True,
        )


        if pre_trained_path is None:
            if os.path.isdir(model_save_dir):
                shutil.rmtree(model_save_dir)
                os.makedirs(model_save_dir, exist_ok=True)

        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver(
            var_list=self.variable_dcgan,
            max_to_keep=2,  # will keep last 5 epochs as default
        )
        begin_at_epoch = 0

        with tf.Session() as sess:

            # For TensorBoard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'train_summaries',
                ),
                graph=sess.graph,
            )

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

            # Initialize model variables
            sess.run(self.variable_init_op_dcgan)

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
                    sess.run(self.metrics_init_op_dcgan)
                    epoch_msg = "Epoch %d/%d\n" % (epoch + 1, epoch_num)
                    sys.stdout.write(epoch_msg)

                    # BATCH : Optimized by each chunk
                    batch_num = 0
                    batch_len = int(np.ceil(len(input_x) / batch_size_int))

                    if self.VALIDATION_RATIO:
                        valid_len = int(batch_len * self.VALIDATION_RATIO)
                        train_len = batch_len - valid_len
                    else:
                        train_len = batch_len

                    gen_adv_step = int(
                        train_len * gen_train_advantage_ratio
                    )

                    batch_remains_ok = True
                    while batch_remains_ok and (batch_num <= batch_len):
                        try:
                            for batch in range(train_len):

                                X_batch, Z_batch = sess.run(self.next_batch)

                                (_,
                                 summary_str_G_train,
                                 summary_learning_rate) = sess.run(
                                    [
                                        self.train_op_G,
                                        #self.update_K,
                                        self.summary_op_G,
                                        self.summary_learning_rate_decay,
                                    ],
                                    feed_dict={
                                        self.Input_Z_dcgan: Z_batch,
                                        self.Bool_is_training: True,
                                        # self.Input_lambda: lambda_val,
                                        # self.Input_gamma: gamma_val,
                                        self.Start_learning_rate_tensor:  self.LEARNING_RATE,
                                        self.Decay_lr_step_tensor: self.DECAY_LR_STEP,
                                    },
                                )
                                err_G = self.loss_G.eval(
                                    {
                                        self.Input_Z_dcgan: Z_batch,
                                        self.Bool_is_training: False,
                                    }
                                )

                                # Optimize Generator first.
                                if batch >= gen_adv_step:
                                    (_, summary_str_D,
                                     global_step_D_val) = sess.run(
                                        [
                                            self.train_op_D,
                                            self.summary_op_D,
                                            self.global_step_D,
                                        ],
                                        feed_dict={
                                            self.Input_X_dcgan: X_batch,
                                            self.Input_Z_dcgan: Z_batch,
                                            self.Bool_is_training: True,
                                            self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                            self.Decay_lr_step_tensor: self.DECAY_LR_STEP,
                                        },
                                    )
                                    err_D_fake = self.loss_D_fake.eval(
                                        {
                                            self.Input_Z_dcgan: Z_batch,
                                            self.Bool_is_training: False,
                                        }
                                    )
                                    err_D_real = self.loss_D_real.eval(
                                        {
                                            self.Input_X_dcgan: X_batch,
                                            self.Bool_is_training: False,
                                        }
                                    )
                                else:
                                    err_D_fake, err_D_real = None, None

                                if ((gen_train_n_times)
                                    and (gen_train_n_times-2 >= 0)):
                                    for n_time in range(gen_train_n_times-2):
                                        _, summary_str_G = sess.run(
                                            [
                                                self.train_op_G,
                                                self.summary_op_G,
                                            ],
                                            feed_dict={
                                                self.Input_Z_dcgan: Z_batch,
                                                self.Bool_is_training: True,
                                                self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                                self.Decay_lr_step_tensor: self.DECAY_LR_STEP,
                                            },
                                        )

                                err_G = self.loss_G.eval(
                                    {
                                        self.Input_Z_dcgan: Z_batch,
                                        self.Bool_is_training: False,
                                    }
                                )

                                sess.run(
                                    [
                                        self.update_metrics_op_train_dcgan,
                                        #self.metrics_train_dcgan,
                                    ],
                                    feed_dict={
                                        self.Input_X_dcgan: X_batch,
                                        self.Input_Z_dcgan: Z_batch,
                                        self.Bool_is_training: False,
                                        self.Start_learning_rate_tensor: self.LEARNING_RATE,
                                        self.Decay_lr_step_tensor: self.DECAY_LR_STEP,
                                    },
                                )
                                # -----------------------------------------------

                                # Write summaries for tensorboard

                                batch_num += 1
                                batch_pct = int(20 * batch_num / train_len)
                                batch_bar = "[%s] " % (("#" * batch_pct) + ("-" * (20 - batch_pct)))
                                batch_msg = "\rBatch [%s/%s] " % (batch_num, train_len)
                                batch_err = 'G: %.5f D_fake: %.5s D_real: %.5s LR: %.7s' % (err_G, err_D_fake, err_D_real, summary_learning_rate)

                                batch_msg = batch_msg + batch_bar + batch_err

                                sys.stdout.flush()
                                sys.stdout.write(batch_msg)

                            if self.VALIDATION_RATIO:

                                for valid in range(valid_len):
                                    X_batch, Z_batch = sess.run(self.next_batch)
                                    sess.run(
                                        [
                                            self.update_metrics_op_valid_dcgan,
                                            #self.metrics_valid_dcgan,
                                        ],
                                        feed_dict={
                                            self.Input_X_dcgan: X_batch,
                                            self.Input_Z_dcgan: Z_batch,
                                            self.Bool_is_training: False,
                                        },
                                    )

                                # -----------------------------------------------

                        except tf.errors.OutOfRangeError:
                            batch_remains_ok = False

                            result_msg = "finished.\n"
                            sys.stdout.write(result_msg)
                            continue

                    train_writer.add_summary(
                        summary_str_G,
                        epoch,
                    )

                    train_writer.add_summary(
                        summary_str_D,
                        epoch,
                    )

                    train_writer.add_summary(
                        summary_learning_rate,
                        epoch,
                    )

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



    def build_AnoGAN(
        self,
        lr_decaying=True,
        decay_lr_step=100,
        learning_rate=.0002,
        adam_beta1=.5,
        adam_beta2=.5,
        dropout=0,
        ano_lambda_=.1,
        use_featuremap=True,
        use_anoZ_tile=False,
        ):


        reuse_ok =  tf.AUTO_REUSE
        print('\n[anogan_model]: ' + '='*30)

        # Objective Functions ================================================

        with tf.variable_scope('anogan_model', reuse=reuse_ok):

            Input_X_anogan = tf.placeholder(
                self.input_x_dtype,
                self.input_x_shape,
                name='input_x_anogan',
            )
            Input_1_anogan = tf.placeholder(
                self._z_input_tensor.dtype,
                self._z_input_tensor.get_shape(),
                name='Input_1_anogan',
            )
            Input_X_discr = tf.placeholder(
                self.input_x_dtype,
                self.input_x_shape,
                name='Input_X_discr',
            )
            Start_learning_rate_tensor_ano = tf.placeholder(
                tf.float32,
                (None),
                name='start_learning_rate_tensor_ano',
            )
            Decay_lr_step_tensor_ano = tf.placeholder(
                tf.int32,
                (None),
                name='Decay_lr_step_tensor_ano',
            )

            with tf.variable_scope('anomaly_z_distributor'):

                if use_anoZ_tile:

                    ano_Z = tf.get_variable(
                        name='ano_Z',
                        shape=[1, self.Z_DIM],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(0, 1., dtype=tf.float32),
                        trainable=True,
                    )

                    ano_Z_tile = tf.tile(
                        input=ano_Z,
                        multiples=[tf.shape(Input_X_anogan)[0], 1],
                        name='ano_Z_tile',
                    )
                    ano_Z_final = ano_Z_tile

                else:
                    ano_Z_batched = tf.get_variable(
                        name='ano_Z_batched',
                        shape=[self.BATCH_SIZE, self.Z_DIM],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(0, 1., dtype=tf.float32),
                        trainable=True,
                    )
                    ano_Z_final = ano_Z_batched




            #--------------------------------
        with tf.variable_scope('dcgan_model', reuse=True):

            fine_G = self._graph_sampler(
                z=Input_1_anogan,
                is_training=tf.constant([False], dtype=tf.bool),
                dropout=dropout,
                name = 'sampler',
                generator_name='generator',
            )

            ano_G = self._graph_sampler(
                z=ano_Z_final,
                is_training=tf.constant([False], dtype=tf.bool),
                dropout=dropout,
                name = 'sampler',
                generator_name='generator',
            )

            with tf.name_scope('evaluate_most_similar_pattern'):

                max_iteration = tf.shape(Input_X_anogan)[0]
                num_i = 0

                outputs = tf.TensorArray(
                    size=max_iteration,
                    dynamic_size=True,
                    dtype=tf.int32,
                )

                input_ta = tf.TensorArray(
                    size=max_iteration,
                    dtype=tf.float32,
                )
                input_ta = input_ta.unstack(Input_X_anogan)


                def cond_fn(num_i, max_iteration, output_ta_t):

                    return num_i < max_iteration


                def body_fn(num_i, max_iteration, output_ta_t):

                    xt = input_ta.read(num_i)

                    new_output = tf.argmin(
                        tf.reduce_mean(
                            tf.abs(
                                tf.subtract(
                                    xt,
                                    ano_G,
                                ),
                            ),
                            axis=[1, 2, 3],
                        ),
                        axis=0,
                        output_type=tf.int32,
                    )
                    output_ta_t = output_ta_t.write(num_i, new_output)

                    return num_i + 1, max_iteration, output_ta_t


                _, _, final_output = tf.while_loop(
                    cond_fn,
                    body_fn,
                    loop_vars=[num_i, max_iteration, outputs],
                )
                most_similar_pattern_idx = final_output.stack()

                selected_ano_G = tf.gather(
                    params=ano_G,
                    indices=most_similar_pattern_idx,
                    axis=0,
                )

        with tf.variable_scope('anogan_model', reuse=reuse_ok):
            with tf.name_scope('loss_residual_origin'):
                loss_residual_origin = tf.abs(
                    tf.subtract(
                        Input_X_anogan,
                        ano_G,
                    )
                )

            with tf.name_scope('loss_residual'):
                loss_residual = tf.reduce_mean(
                    loss_residual_origin,
                )

            with tf.name_scope('loss_residual_selected'):
                loss_residual_selected = tf.abs(
                    tf.subtract(
                        Input_X_anogan,
                        selected_ano_G,
                    )
                )

        if use_featuremap:

            with tf.variable_scope('dcgan_model', reuse=True):
                feature_D_from_X = self._graph_discriminator_featuremap(
                    Input_X_anogan,
                    dropout=dropout,
                    is_training=tf.constant([False], dtype=tf.bool),
                    discriminator_name='discriminator',
                    return_layer_idx=2,
                )
                feature_D_from_Z= self._graph_discriminator_featuremap(
                    ano_G,
                    dropout=dropout,
                    is_training=tf.constant([False], dtype=tf.bool),
                    discriminator_name='discriminator',
                    return_layer_idx=2,
                )
                feature_D_from_Z_selected = self._graph_discriminator_featuremap(
                    selected_ano_G,
                    dropout=dropout,
                    is_training=tf.constant([False], dtype=tf.bool),
                    discriminator_name='discriminator',
                    return_layer_idx=2,
                )

            with tf.variable_scope('anogan_model', reuse=reuse_ok):
                with tf.name_scope('loss_discrimination_origin'):
                    loss_discrimination_origin = tf.abs(
                        tf.subtract(
                            feature_D_from_X,
                            feature_D_from_Z,
                        )
                    )

                with tf.name_scope('loss_discrimination_selected'):
                    loss_discrimination_selected = tf.abs(
                        tf.subtract(
                            feature_D_from_X,
                            feature_D_from_Z_selected,
                        )
                    )

                with tf.name_scope('loss_discrimination'):
                    loss_discrimination = tf.reduce_mean(
                        loss_discrimination_origin,
                    )



        else:
            pass

        with tf.variable_scope('anogan_model', reuse=reuse_ok, auxiliary_name_scope=False):
            with tf.name_scope('anomaly_score'):
                anomaly_score = (
                    ((1 - ano_lambda_) * loss_residual) +
                     (ano_lambda_ * loss_discrimination)
                )
            summary_Z = tf.summary.histogram(
                'anomaly_z',
                ano_Z_final,
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
            summary_G = tf.summary.histogram(
                'ano_G',
                ano_G,
            )
            summary_anomaly_score = tf.summary.scalar(
                'anomaly_score',
                anomaly_score,
            )

        # Optimization =======================================================

        with tf.variable_scope('anogan_optimization'):
            global_step_ano = tf.Variable(
                0,
                trainable=False,
                name='global_step_ano',
            )
            if lr_decaying:
                learning_rate_decay = tf.train.exponential_decay(
                    learning_rate=Start_learning_rate_tensor_ano,
                    global_step=global_step_ano,
                    # decay_steps=100000,
                    # decay_rate=.96,
                    # staircase=True,
                    decay_steps=Decay_lr_step_tensor_ano,
                    decay_rate=.95,
                    staircase=True,
                    name='learning_rate_decay',
                )
            else:
                learning_rate_decay = Start_learning_rate_tensor_ano

            summary_learning_rate_decay_ano = tf.summary.scalar(
                'learning_rate',
                learning_rate_decay,
            )
            optimizer_Z = tf.train.AdamOptimizer(
                learning_rate=learning_rate_decay,
                # learning_rate=learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=1e-08,
                name='optimizer_anomaly_detector',
            )
            variable_anogan = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='anogan_model',
            )
            z_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='anogan_model/anomaly_z_distributor',
            ),
            train_op_Z = optimizer_Z.minimize(
                anomaly_score,
                var_list=z_vars,
                global_step = global_step_ano,
            )
            summary_op_Z = tf.summary.merge([
                summary_Z,
                summary_loss_residual,
                summary_loss_discrimination,
                summary_anomaly_score,
            ])

        # ====================================================================

        var_init_op_anogan = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='anogan_model',
            )
        )
        opt_init_op_anogan = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='anogan_optimization',
            )
        )
        variable_init_op_anogan = tf.group(
            *[var_init_op_anogan, opt_init_op_anogan],
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
        self.var_init_op_anogan = var_init_op_anogan
        self.opt_init_op_anogan = opt_init_op_anogan
        self.variable_init_op_anogan = variable_init_op_anogan

        self.train_op_Z = train_op_Z

        self.metrics_train_anogan = metrics_train_anogan
        self.metrics_valid_anogan = metrics_valid_anogan
        self.update_metrics_op_train_anogan = update_metrics_op_train_anogan
        self.update_metrics_op_valid_anogan = update_metrics_op_valid_anogan
        self.metrics_init_op_anogan = metrics_init_op_anogan

        self.summary_op_Z = summary_op_Z
        self.summary_learning_rate_decay_ano = summary_learning_rate_decay_ano


        self.ano_Z_final = ano_Z_final
        self.ano_G = ano_G
        self.selected_ano_G = selected_ano_G

        self.loss_residual = loss_residual
        self.loss_residual_origin = loss_residual_origin
        self.loss_residual_selected = loss_residual_selected

        self.loss_discrimination = loss_discrimination
        self.loss_discrimination_origin = loss_discrimination_origin
        self.loss_discrimination_selected = loss_discrimination_selected

        self.anomaly_score = anomaly_score
        self.Input_X_anogan = Input_X_anogan
        self.Input_1_anogan = Input_1_anogan
        self.Start_learning_rate_tensor_ano = Start_learning_rate_tensor_ano
        self.Decay_lr_step_tensor_ano = Decay_lr_step_tensor_ano
        self.global_step_ano = global_step_ano


    def train_AnoGAN(
        self,
        input_x=None,
        input_z=None,
        epoch_num=2,
        batch_size=128,
        validation_ratio=.0,
        learning_rate=.0002,
        decay_lr_step=200,
        adam_beta1=.5,
        adam_beta2=.5,
        ano_lambda=.1,
        dcgan_model_save_dir='./model_save_dcgan_sohee',
        model_save_dir='./model_save_anogan_sohee',
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

        if batch_size is None:
            batch_size_int = self.BATCH_SIZE
        else:
            batch_size_int = int(batch_size)

        # --------------------------------------------------------------------
        #global_step = tf.train.get_global_step()
        self.LEARNING_RATE_ANO = learning_rate
        self.VALIDATION_RATIO = validation_ratio
        self.ADAM_BETA1_ANO = adam_beta1
        self.ADAM_BETA2_ANO = adam_beta2
        self.ANO_LAMBDA_ = ano_lambda
        self.DECAY_LR_STEP_ANO = decay_lr_step

        try:
            if not self.VALIDATION_RATIO or not 0 < self.VALIDATION_RATIO < 1:
                raise NoValidationException(self.VALIDATION_RATIO)
            else:
                print("Training with " +
                      "'VALIDATION_RATIO=%s'" % self.VALIDATION_RATIO)
        except KeyError:
            raise NoValidationException(None)
        except NoValidationException as err:
            print(err)
            self.VALIDATION_RATIO = 0

        parameter_dict = {
            'VALIDATION_RATIO': self.VALIDATION_RATIO,
            'DROPOUT': self.DROPOUT,
            'EPOCH_NUM': epoch_num,
            'BATCH_SIZE': self.BATCH_SIZE,
            'LEARNING_RATE': self.LEARNING_RATE_ANO,
            'LR_DECAYING': self.LR_DECAYING_ANO,
            'DECAY_LR_STEP': self.DECAY_LR_STEP_ANO,
            'ADAM_BETA1': self.ADAM_BETA1_ANO,
            'ADAM_BETA2': self.ADAM_BETA2_ANO,
            'ANO_LAMBDA_': self.ANO_LAMBDA_,
        }
        self._print_parameter(parameter_dict)


        self.input(
            input_x_dtype=self.input_x_dtype,
            input_z_dtype=self.input_z_dtype,
            input_x_shape=self.input_x_shape,
            input_z_shape=self.input_z_shape,
            is_training=True,
        )

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
            batch_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'batch_summaries',
                ),
                graph=sess.graph,
            )
            # Initialize model variables
            sess.run(self.var_init_op_dcgan)
            sess.run(self.variable_init_op_anogan)

            # Reload weights from directory if specified
            dcgan_last_save_path = os.path.join(
                dcgan_model_save_dir,
                'last_weights'
            )

            dcgan_last_saver = tf.train.Saver(
                self.variable_dcgan,
                name='dcgan_saver',
            )
            dcgan_saved_model = tf.train.latest_checkpoint(
                dcgan_last_save_path,
            )
            dcgan_last_saver.restore(sess, dcgan_saved_model)
            print("Pre-trained 'DCGAN' model has been loaded.")

            if pre_trained_path is not None:

                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )
                    saved_model = tf.train.latest_checkpoint(
                        last_save_path
                    )
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

                                (_, summary_str_Z,
                                _learning_rate) = sess.run(
                                    [
                                        self.train_op_Z,
                                        self.summary_op_Z,
                                        self.summary_learning_rate_decay_ano,

                                    ],
                                    feed_dict={
                                        self.Input_X_anogan: X_batch,
                                        self.Input_1_anogan: Z_one_batch,
                                        self.Start_learning_rate_tensor_ano:  self.LEARNING_RATE_ANO,
                                        self.Decay_lr_step_tensor_ano: self.DECAY_LR_STEP_ANO,
                                    },
                                )
                                sess.run(
                                    self.update_metrics_op_train_anogan,
                                    feed_dict={
                                        self.Input_X_anogan: X_batch,
                                        self.Input_1_anogan: Z_one_batch,
                                        self.Start_learning_rate_tensor_ano:  self.LEARNING_RATE_ANO,
                                        self.Decay_lr_step_tensor_ano: self.DECAY_LR_STEP_ANO,
                                    },
                                )

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


    def evaluate_DCGAN(
        self,
        #input_x=None,
        input_z=None,
        pre_trained_path='./model_save',
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
            self.build_DCGAN(
                learning_rate = self.LEARNING_RATE,
                adam_beta1 = self.ADAM_BETA1,
                dropout=self.DROPOUT,
            )

        with tf.Session() as sess:
            ## Initialize model variables
            sess.run(self.variable_init_op_dcgan)

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
                        last_saver = tf.train.Saver(
                            self.variable_dcgan
                        )
                        saved_model = tf.train.latest_checkpoint(
                            last_save_path,
                        )
                    #begin_at_epoch = int(saved_model.split('-')[-1])
                    last_saver.restore(sess, saved_model)

            pred = sess.run(
                self.G,
                feed_dict={
                    self.Input_Z_dcgan: input_z,
                    self.Bool_is_training: False,
                }
            )

            # pred_img = (pred + 1.) / 2. * 255
            # Input_X_dcgan = tf.placeholder(
            #     self._x_input_tensor.dtype,
            #     self._x_input_tensor.get_shape(),
            #     name='input_x_dcgan',
            # )
            # Input_Z_dcgan = tf.placeholder(
            #     self._z_input_tensor.dtype,
            #     self._z_input_tensor.get_shape(),
            #     name='input_z_dcgan',
            # )
            #
            # # dcgan_generator = tf.get_collection(
            # #     tf.GraphKeys.GLOBAL_VARIABLES,
            # #     #scope='dcgan_model',
            # # )
            # G = tf.get_collection(
            #     'generator',
            #     scope='dcgan_model',
            # )
            # #pred = sess.run(G, feed_dict={Input_Z_dcgan: input_z})
            # pred = sess.run(G, input_z)

        #tf.get_default_graph().get_tensor_by_name('input')
        return pred

    def evaluate_AnoGAN(
        self,
        input_x=None,
        #input_x_filenames=None,
        input_z=None,
        pre_trained_path_dcgan='./model_save/dcgan',
        pre_trained_path_anogan='./model_save/anogan',
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
        assert pre_trained_path_anogan is not None, "`pre_trained_path` is mandatory."
        #global_step = tf.train.get_global_step()

        if self.ano_G is None:
            tf.reset_default_graph()
            self.build_AnoGAN(
                learning_rate = self.LEARNING_RATE,
                adam_beta1 = self.ADAM_BETA1,
                dropout=self.DROPOUT,
                ano_lambda_=self.ANO_LAMBDA_,
                use_featuremap=self.USE_FEATUREMAP,
            )
        with tf.Session() as sess:
            ## Initialize model variables
            sess.run(self.variable_init_op_dcgan)

            # Reload weights from directory if specified
            assert os.path.isdir(pre_trained_path_dcgan)
            last_save_path_dcgan = os.path.join(
                pre_trained_path_dcgan,
                'last_weights'
            )
            if target_epoch:
                saved_model_dcgan = ''.join(
                    [
                        last_save_path_dcgan + '/',
                        'after-epoch-',
                        str(target_epoch),
                    ],
                )
                last_saver_dcgan = tf.train.import_meta_graph(
                    saved_model_dcgan + '.meta'
                )
            else:
                last_saver_dcgan = tf.train.Saver(
                    self.variable_dcgan
                )
                saved_model_dcgan = tf.train.latest_checkpoint(
                    last_save_path_dcgan,
                )
                last_saver_dcgan.restore(sess, saved_model_dcgan)


            if pre_trained_path_anogan is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path_anogan):
                    last_save_path_anogan = os.path.join(pre_trained_path_anogan, 'last_weights')

                    if target_epoch:
                        saved_model_anogan = ''.join(
                            [
                                last_save_path_anogan + '/',
                                'after-epoch-',
                                str(target_epoch),
                            ],
                        )
                        last_saver_anogan = tf.train.import_meta_graph(
                            saved_model_anogan + '.meta'
                        )
                    else:
                        last_saver_anogan = tf.train.Saver(
                            self.variable_anogan
                        )
                        saved_model_anogan = tf.train.latest_checkpoint(
                            last_save_path_anogan,
                        )
                        last_saver_anogan.restore(sess, saved_model_anogan)

            res_list = []
            for _ in range(len(input_x) // self.BATCH_SIZE):
                loss_residual_origin =sess.run(
                            self.loss_residual_origin,
                            feed_dict={
                            self.Input_X_anogan: input_x,
                            }
                )

                loss_discrimination_origin =sess.run(
                            self.loss_discrimination_origin,
                            feed_dict={
                            self.Input_X_anogan: input_x,
                            }
                )
                loss_residual_selected =sess.run(
                            self.loss_residual_selected,
                            feed_dict={
                            self.Input_X_anogan: input_x,
                            }
                )

                loss_discrimination_selected =sess.run(
                            self.loss_discrimination_selected,
                            feed_dict={
                            self.Input_X_anogan: input_x,
                            }
                )
                ano_G, selected_ano_G = sess.run(
                        [
                            self.ano_G,
                            self.selected_ano_G,
                        ],
                        feed_dict={
                            self.Input_X_anogan:input_x,
                            }
                )

                pred = sess.run(
                    self.anomaly_score,
                    feed_dict={
                        self.Input_X_anogan: input_x,
                        }
                )
                res_list += [pred]

            return (
                res_list,
                loss_residual_origin,
                loss_discrimination_origin,
                loss_residual_selected,
                loss_discrimination_selected,
                ano_G,
                selected_ano_G,
            )

    def loss_plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
        ax.plot(self.train_loss_history, label='train')
        if self.VALIDATION_RATIO:
            ax.plot(self.train_valid_history, label='valid')
        #ax.set_xticks(np.arange(self.))
        return fig
