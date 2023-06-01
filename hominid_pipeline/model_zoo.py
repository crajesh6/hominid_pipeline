import tensorflow as tf
import tensorflow.keras as keras
from hominid_pipeline import layers
from tensorflow.keras.regularizers import l1, l2, l1_l2


class AttentionPooling(keras.layers.Layer):

    def __init__(self, pool_size, *args, **kwargs):
        super(AttentionPooling, self).__init__(*args, **kwargs)
        self.pool_size = pool_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
        })
        return config

    def build(self, input_shape):
        self.dense = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1, activation=None, use_bias=False)

    def call(self, inputs):
        N, L, F = inputs.shape
        inputs = tf.keras.layers.Cropping1D((0, L % self.pool_size))(inputs)
        inputs = tf.reshape(inputs, (-1, L//self.pool_size, self.pool_size, F))

        raw_weights = self.dense(inputs)
        att_weights = tf.nn.softmax(raw_weights, axis=-2)

        return tf.math.reduce_sum(inputs * att_weights, axis=-2)


# let's establish our base model:
def base_model(
    conv1_activation,
    conv1_batchnorm,
    conv1_channel_weight,
    conv1_dropout,
    conv1_filters,
    conv1_kernel_size,
    conv1_pool_type,
    conv1_attention_pool_size,
    conv1_max_pool,
    conv1_type,
    dense_activation,
    dense_batchnorm,
    dense_dropout,
    dense_units,
    input_shape,
    mha_d_model,
    mha_dropout,
    mha_head_type,
    mha_heads,
    mha_layernorm,
    output_activation,
    output_shape,
    softconv_filters=128,
    ):

    diag = l2(1e-6)
    offdiag = l2(1e-3)

    keras.backend.clear_session()

    inputs = keras.layers.Input(shape=input_shape, name='input')

    # additive conv layer
    if conv1_type == 'standard':
        nn = keras.layers.Conv1D(filters=conv1_filters, kernel_size=conv1_kernel_size, padding='same', use_bias=True, name='conv1')(inputs)
    else:
        nn = layers.PairwiseConv1D(
                        conv1_filters,
                        kernel_size=conv1_kernel_size,
                        padding='same',
                        kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
                        use_bias=True)(inputs)
    if conv1_batchnorm:
        nn = keras.layers.BatchNormalization(name='conv1_bn')(nn)
    nn = keras.layers.Activation(conv1_activation, name='conv1_activation')(nn)

    if conv1_channel_weight == 'se':
            ratio = 4
            b, _, c = nn.shape
            x = tf.keras.layers.GlobalAveragePooling1D(name='se_pool')(nn)
            x = tf.keras.layers.Dense(c // ratio, activation="silu", use_bias=False, name='se_silu')(x)
            x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False, name='se_sigmoid')(x)
            x = tf.keras.layers.Reshape((1, c), input_shape=(x.shape), name='se_reshape')(x)
            nn = nn * x

    elif conv1_channel_weight == 'softconv':
            nn = tf.keras.layers.Conv1D(filters=softconv_filters, kernel_size=1, padding='same', use_bias=True, name='softconv_conv')(nn) # change this!
            nn = keras.layers.Activation('relu', name='softconv_activation')(nn)
    if conv1_pool_type == 'attention':
        nn = AttentionPooling(conv1_attention_pool_size)(nn)
    else:
        if (conv1_max_pool != 0):
            nn = keras.layers.MaxPooling1D(conv1_max_pool, name='conv1_maxpool')(nn)
    nn = keras.layers.Dropout(conv1_dropout, name='conv1_dropout')(nn)

    # multi-head attention layer
    # TODO: change this to pooled or task specific
    if mha_head_type == 'task_specific':
        outputs = []
        for task in range(output_shape):
            if mha_layernorm:
                nn = keras.layers.LayerNormalization(name=f'mha_layernorm_{task}')(nn)
            nn2, att = layers.MultiHeadAttention(num_heads=1, d_model=96)(nn, nn, nn)
            nn2 = keras.layers.Dropout(0.1)(nn2)
            # mlp layers
            nn2 = keras.layers.Flatten()(nn2)
            for units, dropout in zip([256, 256], [0.4, 0.4]):
                nn2 = keras.layers.Dense(units)(nn2)
                if dense_batchnorm:
                    nn2 = keras.layers.BatchNormalization()(nn2)
                nn2 = keras.layers.Activation("relu")(nn2)
                nn2 = keras.layers.Dropout(dropout)(nn2)

                # add the outputs to the list of outputs
            output = keras.layers.Dense(1, activation='linear')(nn2)
            outputs += [output]
        # recombine the final output layer
        outputs = tf.concat(outputs, axis=-1)
    else:
        if mha_layernorm:
            nn = keras.layers.LayerNormalization(name='mha_layernorm')(nn)
        nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
        nn = keras.layers.Dropout(mha_dropout, name='mha_dropout')(nn)
        # mlp layers
        dense_count = 0
        nn = keras.layers.Flatten(name='flatten')(nn)
        for units, dropout in zip(dense_units, dense_dropout):
            nn = keras.layers.Dense(units, name=f'dense_{dense_count}')(nn)
            if dense_batchnorm:
                nn = keras.layers.BatchNormalization(name=f'bn_{dense_count}')(nn)
            nn = keras.layers.Activation(dense_activation, name=f'dense_activation_{dense_count}')(nn)
            nn = keras.layers.Dropout(dropout, name=f'dense_dropout_{dense_count}')(nn)
            dense_count += 1

        # output layer
        if output_activation=='linear':
            outputs = keras.layers.Dense(output_shape, activation='linear', name='output')(nn)
        else:
            logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
            outputs = keras.layers.Activation(output_activation, name='output')(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs)




def PairwiseConvAtt(
    input_shape,
    output_shape,
    num_filters=128,
    kernel_size=15,
    diag=l2(1e-6),
    offdiag=l2(1e-3),
    conv_activation='relu',
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10,
    mha_heads=4,
    mha_d_model=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batch_norm=True,
    dense_activation='relu',
    output_activation='linear',
    se_block=False
    ):

    inputs = keras.layers.Input(shape=input_shape)

    # pairwise conv layer
    nn = layers.PairwiseConv1D(num_filters,
                               kernel_size=kernel_size,
                               padding='same',
                               kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
                               use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)

    if se_block:
            ratio = 4
            b, _, c = nn.shape
            x = tf.keras.layers.GlobalAveragePooling1D()(nn)
            x = tf.keras.layers.Dense(c // ratio, activation="silu", use_bias=False)(x)
            x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
            x = tf.keras.layers.Reshape((1, c), input_shape=(x.shape))(x)
            nn = nn * x
    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation=='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def PairwiseConvSEAtt(
    input_shape,
    output_shape,
    num_filters=128,
    kernel_size=15,
    diag=l2(1e-6),
    offdiag=l2(1e-3),
    conv_activation='relu',
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10,
    mha_heads=4,
    mha_d_model=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batch_norm=True,
    dense_activation='relu',
    output_activation='linear',

    ):

    inputs = keras.layers.Input(shape=input_shape)

    # pairwise conv layer
    nn = layers.PairwiseConv1D(num_filters,
                               kernel_size=kernel_size,
                               padding='same',
                               kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
                               use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)

    # SE block
    ratio = 4
    b, _, c = nn.shape
    x = tf.keras.layers.GlobalAveragePooling1D()(nn)
    x = tf.keras.layers.Dense(c // ratio, activation="silu", use_bias=False)(x)
    x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
    x = tf.keras.layers.Reshape((1, c), input_shape=(x.shape))(x)
    nn = nn * x

    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation=='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs)




def AdditiveConvAtt(
    input_shape,
    output_shape,
    num_filters=128,
    kernel_size=15,
    conv_activation='relu',
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10,
    mha_heads=4,
    mha_d_model=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batchnorm=True,
    dense_activation='relu',
    output_activation='linear',
    se_block=False
    ):

    inputs = keras.layers.Input(shape=input_shape)

    # additive conv layer
    nn = nn = keras.layers.Conv1D(filters=128, kernel_size=15, padding='same', use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)

    if se_block:
            ratio = 4
            b, _, c = nn.shape
            x = tf.keras.layers.GlobalAveragePooling1D()(nn)
            x = tf.keras.layers.Dense(c // ratio, activation="silu", use_bias=False)(x)
            x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
            x = tf.keras.layers.Reshape((1, c), input_shape=(x.shape))(x)
            nn = nn * x
    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batchnorm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation=='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def AdditiveConvSEAtt(
    input_shape,
    output_shape,
    num_filters=128,
    kernel_size=15,
    conv_activation='relu',
    conv_dropout=0.2,
    conv_batchnorm=None,
    max_pool=10,
    mha_heads=4,
    mha_d_model=96,
    mha_dropout=0.1,
    mha_layernorm=False,
    dense_units=[256, 256],
    dense_dropout=[0.4, 0.4],
    dense_batchnorm=True,
    dense_activation='relu',
    output_activation='linear',
    ):

    inputs = keras.layers.Input(shape=input_shape)

    # additive conv layer
    nn = nn = keras.layers.Conv1D(filters=128, kernel_size=15, padding='same', use_bias=True)(inputs)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv_activation, name='conv_activation')(nn)

    # SE block
    ratio = 4
    b, _, c = nn.shape
    x = tf.keras.layers.GlobalAveragePooling1D()(nn)
    x = tf.keras.layers.Dense(c // ratio, activation="silu", use_bias=False)(x)
    x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
    x = tf.keras.layers.Reshape((1, c), input_shape=(x.shape))(x)
    nn = nn * x

    nn = keras.layers.MaxPooling1D(max_pool)(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn)

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
    nn = keras.layers.Dropout(mha_dropout)(nn)

    # mlp layers
    nn = keras.layers.Flatten()(nn)
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units)(nn)
        if dense_batchnorm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(dense_activation)(nn)
        nn = keras.layers.Dropout(dropout)(nn)

    # output layer
    if output_activation=='linear':
        outputs = keras.layers.Dense(output_shape, activation='linear')(nn)
    else:
        logits = keras.layers.Dense(output_shape, activation='linear', name='logits')(nn)
        outputs = keras.layers.Activation(output_activation)(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def AdditiveConvAttBase(
    input_shape,
    num_tasks,
    filters=128,
    kernel_size=15,
    conv_batchnorm=False,
    conv_activation='exponential',
    conv_dropout=0.2,
    downsample_factor=3,
    mha_heads=4,
    mha_d_model=128,
    mha_dropout=0.1,
    mha_layernorm=False,
    bottleneck=128,
    decode_filters=64,
    decode_kernel_size=7,
    decode_batchnorm=True,
    decode_activation='relu',
    decode_dropout=0.4,
    num_resid=4,
    task_filters=32,
    task_kernel_size=7,
    task_dropout=0.2,
    task_activation='softplus',

    ):

    inputs = keras.layers.Input(shape=input_shape)

    # zero-pad to ensure L can downsample exactly with 2^downsample
    max_pool = 2**downsample_factor
    remainder = tf.math.mod(input_shape[0], max_pool)
    inputs_padded = tf.keras.layers.ZeroPadding1D((0, max_pool-remainder.numpy()))(inputs)

    # convolutional layer
    nn = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=True)(inputs_padded)
    if conv_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn_connect = keras.layers.Activation(conv_activation, name='conv_activation')(nn)
    nn = keras.layers.Dropout(conv_dropout)(nn_connect)
    nn = keras.layers.MaxPool1D(max_pool, padding='same')(nn) #10

    # multi-head attention layer
    if mha_layernorm:
        nn = keras.layers.LayerNormalization()(nn)
    nn1, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(nn, nn, nn)
    nn1 = keras.layers.Dropout(mha_dropout)(nn1)

    # expand back to base-resolution
    for i in range(downsample_factor):
        nn = keras.layers.Conv1DTranspose(filters=decode_filters, kernel_size=decode_kernel_size, strides=2, padding='same')(nn)
        if decode_batchnorm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(decode_activation)(nn)
        nn = keras.layers.Dropout(decode_dropout)(nn)

    nn = keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same')(nn)
    if decode_batchnorm:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(decode_activation)(nn)
    nn = keras.layers.Dropout(decode_dropout)(nn)
    nn = keras.layers.add([nn, nn_connect])

    nn2 = residual_block(nn, 3, activation=dense_activation, dilated=num_resid)
    nn = keras.layers.add([nn, nn2])

    outputs = []
    for i in range(num_tasks):
        nn2 = keras.layers.Conv1D(filters=output_filters, kernel_size=output_kernel_size, padding='same')(nn)
        nn2 = keras.layers.Activation(decode_activation)(nn2)
        nn2 = keras.layers.Dropout(task_dropout)(nn2)
        nn2 = keras.layers.Dense(1, activation=task_activation)(nn2)
        outputs.append(nn2)
    outputs = tf.concat(outputs, axis=2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)





def residual_block(input_layer, filter_size, activation='relu', dilated=5):

    factor = []
    base = 2
    for i in range(dilated):
        factor.append(base**i)

    num_filters = input_layer.shape.as_list()[-1]

    nn = keras.layers.Conv1D(filters=num_filters,
                                    kernel_size=filter_size,
                                    activation=None,
                                    use_bias=False,
                                    padding='same',
                                    dilation_rate=1,
                                    )(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    for f in factor:
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = keras.layers.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        strides=1,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        dilation_rate=f,
                                        )(nn)
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)
