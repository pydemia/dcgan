"""Test `AnoGAN` Interactively.
"""

# %% Pre-load ----------------------------------------------------------------

import tensorflow as tf
import os
import shutil
print(os.getcwd())
if os.getcwd().split('/')[-1] != 'dcgan':
    os.chdir('../git/dcgan')

# !python download.py celebA

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.io import imread_collection, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# %% Data Preprocessing ------------------------------------------------------

# from glob import glob
# import shutil
#
# # sample_list = [
# #     1000,
# #     3000,
# #     5000,
# #     10000,
# #     50000,
# #     100000,
# # ]
# #
# # origin_list = glob('./data/celebA/*.jpg')
# # for _ in sample_list:
# #     alist = origin_list[:_]
# #
# #     newdir_name = './data/celebA_%s' % _
# #     if os.path.isdir(newdir_name):
# #         shutil.rmtree(newdir_name)
# #     os.makedirs(newdir_name, exist_ok=True)
# #     [shutil.copy2(file, './data/celebA_%s/' % _) for file in alist]
#
#
# origin_path = './data/celebA'
# origin_list = glob(origin_path + '/*.jpg')
# origin_data = imread_collection(
#     load_pattern=origin_list,
# )
#
# def center_crop(image, size=(128, 128)):
#     xcrop, ycrop = size
#     ysize, xsize, chan = image.shape
#     xoff = (xsize - xcrop) // 2
#     yoff = (ysize - ycrop) // 2
#     cropped = image[yoff:-yoff,xoff:-xoff]
#     return cropped
#
# cropped = map(lambda img: center_crop(img, size=(64, 64)), origin_data)
#
# resized = np.array(
#     list(map(lambda img: resize(
#                 img,
#                 (64, 64),
#                 preserve_range=False,
#                 anti_aliasing=True,
#                 mode='reflect',
#             ),
#             cropped
#         )
#     ),
#     dtype=np.float32,
# )
#
#
# sample_list = [
#     1000,
#     3000,
#     5000,
#     10000,
#     50000,
#     100000,
# ]
#
# for _ in sample_list:
#     alist = resized[:_]
#
#     newdir_name = './data/celebA_cropped_%s' % _
#     if os.path.isdir(newdir_name):
#         shutil.rmtree(newdir_name)
#     os.makedirs(newdir_name, exist_ok=True)
#     [imsave(
#         newdir_name + '/cropped_%s.png' % str(i).zfill(len(str(_))),
#         img,
#     ) for i, img in enumerate(alist)]

# %% -------------------------------------------------------------------------

# %%
# data_x = imread_collection(
#     #load_pattern='./data/celebA/*.jpg',
#     #load_pattern='./data/celebA_1000/*.jpg',
#     #load_pattern='./data/celebA_3000/*.jpg',
#     #load_pattern='./data/celebA_5000/*.jpg',
#     #load_pattern='./data/celebA_10000/*.jpg',
#     #load_pattern='./data/celebA_50000/*.jpg',
#     #load_pattern='./data/celebA_100000/*.jpg',
#     load_pattern='./data/celebA_cropped_100000/*.png',
# )
# data_x = (np.array(data_x) - .5) * 2.
#
# plt.imshow(data_x[0])
# print('`data_x` is ready:', data_x.shape, data_x.dtype)
#
#
# data_x[0].shape
# data_x[0].mean(axis=0).astype(np.float32)
# data_x[0].min(axis=0).astype(np.float32)
# data_x[0].max(axis=0).astype(np.float32)
# data_x[0].std(axis=0).astype(np.float32)


from glob import glob


#data_x_filenames = glob('/Users/soheehwang/Downloads/DCGAN-tensorflow-master/samples/*.png')
# data_x = glob('./data/celebA/*.jpg')
data_x = glob('./data/data_crop_256_jpg/*.jpg')
# plt.imshow(data_x[0])
print('`data_x` is ready:', len(data_x))


# data_z = np.random.normal(
#     loc=0.,
#     scale=1.,
#     size=(len(data_x), 100),
# )
data_z = np.random.uniform(
    low=0.,
    high=1.,
    size=(len(data_x), 64),
).astype(np.float32)

print('`data_z` is ready:', data_z.shape, data_z.dtype)

data_onevector = np.ones_like(data_z)

(train_x, test_x,
 train_z, test_z,
 train_onevector, test_onevector) = train_test_split(
    data_x,
    data_z,
    data_onevector,
    test_size=.2,
)


# %% AnoGAN: Build -----------------------------------------------------------

import importlib
from src.began import began as BeGAN
importlib.reload(BeGAN)
BEGAN = BeGAN.BEGAN

tf.reset_default_graph()
began = BEGAN(
    input_x_ext='jpg',
    input_x_dtype=tf.float32,
    input_z_dtype=tf.float32,
    input_x_shape=(None, 256, 256, 3),
    input_z_shape=(None, 64),
    use_gpu=True,
    #input_width=64,
    #input_height=64,
    #input_channel=3,
    # output_width=None,
    # output_height=None,
    # output_channel=None,
    # class_num=1,
    filter_dim=64,  # 128
    # g_filter_dim=64,
    # d_filter_dim=64,
    g_fc_dim=1024,
    d_fc_dim=1024,
    batch_size=64,
    batch_norm_ok=False,
    conv_activation=tf.nn.elu,
    dropout=None,
    lr_decaying=True,
    decay_lr_step=100_000,
    buffer_size=1000,
    learning_rate=0.0005,
    adam_beta1=.5,
    adam_beta2=.9,
    validation_ratio=.2,
    ano_lambda_=.1,
)

# %% AnoGAN: Train DCGAN -----------------------------------------------------

shutil.rmtree('./model_save/began_origin', ignore_errors=True)

# %%

began.train_BEGAN(
    #input_x=train_x,
    input_x_filenames=train_x,
    input_z=train_z,
    batch_size=16,
    epoch_num=5,
    validation_ratio=.05,
    learning_rate=.0001,
    lambda_val=.001,
    gamma_val=.7,
    model_save_dir='./model_save/began_origin',
    #pre_trained_path='./model_save/began_origin',
    pre_trained_path=None,
    verbose=True,
    #writer=None,
)


# %% AnoGAN: Train DCGAN -----------------------------------------------------

began.train_BEGAN(
    #input_x=train_x,
    input_x_filenames=train_x,
    input_z=train_z,
    batch_size=32,
    epoch_num=50,
    validation_ratio=.05,
    learning_rate=.0003,
    lambda_val=.001,
    gamma_val=.7,
    model_save_dir='./model_save/began_origin',
    pre_trained_path='./model_save/began_origin',
    #pre_trained_path=None,
    verbose=True,
    #writer=None,
)


# %% AnoGAN: Evaluate DCGAN --------------------------------------------------

gen_x = began.evaluate_BEGAN(
    input_z=test_z[:1],
    pre_trained_path='model_save_dcgan_origin',
    target_epoch=5,
)
test_z[0]
gen_x.shape
gen_x[0].min()
gen_x[0].max()
img = gen_x[0]
img

pred_img = (img + 1.) / 2. * 255
plt.imshow(gen_x[0])
type(gen_x)

gen_x


# %% AnoGAN: Train AnoGAN ----------------------------------------------------

began.train_BEGAN(
    input_x=train_x,
    input_z=train_onevector,
    epoch_num=1,
    model_save_dir='./model_save_anogan_origin',
    pre_trained_path=None,
    verbose=True,
    writer=None,
)


# %% AnoGAN: Evaluate AnoGAN -------------------------------------------------

gen_x = began.evaluate_BEGAN(
    input_z=test_z[:1],
    pre_trained_path='model_save_dcgan_origin',
    target_epoch=5,
)

# %% Test Code: tf.data.Dataset ----------------------------------------------

if __name__ == '__main__':
    tf.reset_default_graph()
    X_t = tf.placeholder(tf.int16, (None, 2),
                         name='x_tensor_interface')
    Z_t = tf.placeholder(tf.int16,  (None, 1),
                         name='z_tensor_interface')

    dataset = tf.data.Dataset.from_tensor_slices((X_t, Z_t))
    dataset = dataset.shuffle(buffer_size=1000)  # reshuffle_each_iteration=True as default.
    dataset = dataset.batch(2)
    dataset = dataset.flat_map(
        lambda data_x, data_z: tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensors(data_x),
                tf.data.Dataset.from_tensors(data_z),
            )
        ).repeat(3)
    )


    data_op = dataset.make_initializable_iterator()
    data_init_op = data_op.initializer
    X_batch, Z_batch = data_op.get_next()

    bias_x0 = tf.convert_to_tensor(np.array([1, 2]), dtype=tf.int16)
    bias_z0 = tf.convert_to_tensor(np.array([7]), dtype=tf.int16)

    bias_x1 = tf.convert_to_tensor(np.array([10, 11]), dtype=tf.int16)
    bias_z1 = tf.convert_to_tensor(np.array([50]), dtype=tf.int16)
    add1 = tf.nn.bias_add(X_batch, bias_x0)
    add2 = tf.nn.bias_add(Z_batch, bias_z0)
    add3 = tf.nn.bias_add(X_batch, bias_x1)
    add4 = tf.nn.bias_add(Z_batch, bias_z1)

    a = np.array([
        [100, 100],
        [200, 200],
        [300, 300],
        [400, 400],
        [500, 500],
    ])

    b = np.array([
        [600],
        [700],
        [800],
        [900],
        [600],
    ])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_op.run()

        for epoch in range(10):
            print('[EPOCH]', epoch+1, '======================')
            sess.run(data_init_op, feed_dict={X_t: a, Z_t: b})
            batch_remains_ok = True
            batch_num = 0
            while batch_remains_ok and (batch_num+1 < 3):
                try:
                    for batch_num in range(3): # batch_size=2, num=3
                        print('[BATCH]', batch_num+1, '--------------')
                        res1 = sess.run(add1)
                        # res2 = sess.run(add2)
                        # res3, res4 = sess.run([add3, add4])
                        res5 = sess.run(add1)
                        # res6 = sess.run(add2)

                        print('res1', res1, '\n')
                        # print(res2)
                        # print(res3, '\n')
                        print('res5', res5, '\n')
                        # print(res6, '\n')

                except tf.errors.OutOfRangeError:
                    batch_remains_ok = False
                    continue



if __name__ == '__main__':
    tf.reset_default_graph()
    X_t = tf.placeholder(tf.int16, (None, 2),
                         name='x_tensor_interface')
    Z_t = tf.placeholder(tf.int16,  (None, 1),
                         name='z_tensor_interface')

    dataset = tf.data.Dataset.from_tensor_slices((X_t, Z_t))
    dataset = dataset.shuffle(buffer_size=1000)  # reshuffle_each_iteration=True as default.
    dataset = dataset.batch(2)
    dataset = dataset.flat_map(
        lambda data_x, data_z: tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensors(data_x),
                tf.data.Dataset.from_tensors(data_z),
            )
        ).repeat(12)
    )


    data_op = dataset.make_initializable_iterator()
    data_init_op = data_op.initializer
    X_batch, Z_batch = data_op.get_next()

    bias_x0 = tf.convert_to_tensor(np.array([1, 2]), dtype=tf.int16)
    bias_z0 = tf.convert_to_tensor(np.array([7]), dtype=tf.int16)

    bias_x1 = tf.convert_to_tensor(np.array([10, 11]), dtype=tf.int16)
    bias_z1 = tf.convert_to_tensor(np.array([50]), dtype=tf.int16)
    add1 = tf.nn.bias_add(X_batch, bias_x0)
    add2 = tf.nn.bias_add(Z_batch, bias_z0)
    add3 = tf.nn.bias_add(X_batch, bias_x1)
    add4 = tf.nn.bias_add(Z_batch, bias_z1)

    a = np.array([
        [100, 100],
        [200, 200],
        [300, 300],
        [400, 400],
        [500, 500],
    ])

    b = np.array([
        [600],
        [700],
        [800],
        [900],
        [600],
    ])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_op.run()

        for epoch in range(10):
            print('[EPOCH]', epoch+1, '======================')
            xx, zz = sess.run(data_init_op, feed_dict={X_t: a, Z_t: b})

            print(xx)
            print(zz)


# %% Test Code: tf.train.Saver load ------------------------------------------

if __name__ == '__main__':

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = tf.train.import_meta_graph('model_save_dcgan_origin/last_weights/after-epoch-2.meta')
        saved_var = 'model_save_dcgan_origin/last_weights/after-epoch-2'
        model.restore(sess, saved_var)

# %% Test Code: reduce_mean --------------------------------------------------

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    tf.reset_default_graph()

    def _layer_linear(
        input_x,
        output_size,
        is_training=True,
        stddev=0.02,
        bias_start=0.,
        dropout=.3,
        return_weight=False,
        reuse=tf.AUTO_REUSE,
        name='linear',
        ):

        with tf.variable_scope(name, reuse=reuse):

            input_shape = input_x.get_shape().as_list()

            weight = tf.get_variable(
                'weight',
                shape=(input_shape[1], output_size),
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    mean=1.,
                    stddev=stddev
                ),
            )
            bias = tf.get_variable(
                'bias',
                shape=(output_size,),
                dtype=tf.float32,
                initializer=tf.constant_initializer(bias_start),
            )

            lineared = tf.nn.bias_add(
                input_x @ weight,
                bias,
                data_format='NHWC',
                name='linear_function',
            )

            # (Optional) Dropout Layer
            # if dropout:
            #
            #     keep_prob = tf.cond(
            #         is_training,
            #         lambda x: tf.Constant((1. - dropout)),
            #         lambda x: tf.Constant(1.),
            #         name='choose_prob_if_training'
            #     )
            #     # TBD: tf.layers.dropout()
            #     lineared = tf.nn.dropout(
            #         lineared,
            #         keep_prob=keep_prob,
            #         name='dropout',
            #     )

        print(name, lineared.get_shape())

        if return_weight:
          return lineared, weight, bias
        else:
          return lineared

    a = np.arange(10 * 64 * 64 * 3).reshape(-1, 64, 64, 3).astype(np.float32)

    a_tensor = tf.convert_to_tensor(a, dtype=tf.float32)
    a_ones_tensor = tf.ones_like(a_tensor, dtype=tf.float32)
    a_ones_subset_tensor = tf.ones_like(a_tensor[:,:], dtype=tf.float32)
    a_sigmoid_tensor = tf.nn.sigmoid(a_tensor)
    a_dense_tensor = tf.layers.dense(a_tensor, 1)
    a_dense_sigmoid_tensor = tf.nn.sigmoid(a_dense_tensor)
    a_reshape_tensor = tf.reshape(
        a_tensor,
        shape=(-1, 64 * 64 * 3),
    )
    a_linear_tensor = _layer_linear(
        a_reshape_tensor,
        output_size=1,
        return_weight=False,
    )
    a_linear_sigmoid_tensor = tf.nn.sigmoid(a_linear_tensor)
    a_linear_sigmoid_ones_tensor = tf.ones_like(a_linear_sigmoid_tensor, dtype=tf.float32)

    var_init_op = tf.global_variables_initializer()
    with tf.Session() as sess:

        print('a', a.shape)
        print('a_tensor', a_tensor.get_shape())
        print('a_ones_tensor', a_ones_tensor.get_shape())
        print('a_ones_subset_tensor', a_ones_subset_tensor.get_shape())
        print('a_sigmoid_tensor', a_sigmoid_tensor.get_shape())
        print('a_dense_tensor', a_dense_tensor.get_shape())
        print('a_dense_sigmoid_tensor', a_dense_sigmoid_tensor.get_shape())
        print('a_reshape_tensor', a_reshape_tensor.get_shape())
        print('a_linear_tensor', a_linear_tensor.get_shape())
        print('a_linear_sigmoid_tensor', a_linear_sigmoid_tensor.get_shape())
        print('a_linear_sigmoid_ones_tensor', a_linear_sigmoid_ones_tensor.get_shape())

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=a_linear_sigmoid_tensor,
            labels=a_linear_sigmoid_ones_tensor,
        )
        loss_mean = tf.reduce_mean(
            loss
        )
        print(loss.get_shape())
        print(loss_mean.get_shape())
        sess.run(var_init_op)
        print(sess.run(loss))
        print(sess.run(loss_mean))
