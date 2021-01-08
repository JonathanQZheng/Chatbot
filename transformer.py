import tensorflow as tf
import matplotlib.pyplot as plt

def scaled_dot_product_attention(query, key, value, mask):
    # Calculate the attention weights.
    matmul_qk = tf.matmul(query, key, transpose_b = True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax normalize on last axis
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenate heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

# mask out padding tokens
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # dimension: (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

print(create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])))
# mask out future data
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model)
        # sin is applied to the even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        # cos is applied to the odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# sample_pos_encoding = PositionalEncoding(50, 512)

# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    # Create a tensor input
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    # Create a padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    # Create the attention layer
    attention = MultiHeadAttention(d_model, num_heads, name="attention") ({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    # Apply dropout to attention
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    # Apply layer noramlization to attention
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    # Apply dense layer to attention
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    # Apply dense layer
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    # Apply dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    # Apply normalization to outputs and attention
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

# sample_encoder_layer = encoder_layer(units=512, d_model=128, num_heads=4, dropout=0.3, name="sample_encoder_layer")

# tf.keras.utils.plot_model(sample_encoder_layer, to_file='encoder_layer1.png', show_shapes=True)

def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    # Create a tensor input
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    # Create a padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    # Create embeddings
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # Multiply embeddings by the square root of the dimension of the model
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # Apply the positional encoding to the embeddings
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    # Apply dropout to the embeddings
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    # For the number of layers, encode the inputs
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

# sample_encoder = encoder(vocab_size=8192, num_layers=2, units=512, d_model=128, num_heads=4, dropout=0.3, name="sample_encoder")

# tf.keras.utils.plot_model(sample_encoder, to_file='encoder1.png', show_shapes=True)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    # Create a tensor input
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    # Create the encoded outputs that are used as inputs in the decoder layer
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    # Create the look ahead mask
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    # Create the padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    # First layer of attention
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    # Normalize the attention and inputs
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    # Second layer of attention
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    # Apply dropout
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    # Normalize the attention and the inputs (attention 1)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)
    # Dense layer 1
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    # Dense layer 2
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    # Apply dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    # Normalize the output and inputs (attention 2)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],outputs=outputs, name=name)

# sample_decoder_layer = decoder_layer(units=512,d_model=128,num_heads=4,dropout=0.3,name="sample_decoder_layer")

# tf.keras.utils.plot_model(sample_decoder_layer, to_file='decoder_layer1.png', show_shapes=True)

def decoder(vocab_size,num_layers,units,d_model,num_heads,dropout,name='decoder'):
    # Create a tensor input
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    # Create the encoded outputs that are used as inputs to the decoder
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    # Create a look ahead mask
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    # Create a padding mask
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    # Create the embeddings
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # Apply the embeddings to the positional encoding embeddings
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    # Apply dropout to the embeddings
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    # For the number of layers, decode the outputs
    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],outputs=outputs,name=name)

# sample_decoder = decoder(vocab_size=8192,num_layers=2,units=512,d_model=128,num_heads=4,dropout=0.3,name="sample_decoder")

# tf.keras.utils.plot_model(sample_decoder, to_file='decoder1.png', show_shapes=True)

def transformer(vocab_size,num_layers,units,d_model,num_heads,dropout,name="transformer"):
    # Create a tensor input
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    # Decode the inputs
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
    # Create the encoded padding mask
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask,output_shape=(1, None, None),name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='dec_padding_mask')(inputs)
    # put the inputs through the encoder layer
    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])
    # put the encoded outputs through the decoder layer
    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    # put the decoded output through a dense linear layer
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

# sample_transformer = transformer(vocab_size=8192,num_layers=4,units=512,d_model=128,num_heads=4,dropout=0.3,name="sample_transformer")

# tf.keras.utils.plot_model(sample_transformer, to_file='transformer1.png', show_shapes=True)

# tf.keras.backend.clear_session()
