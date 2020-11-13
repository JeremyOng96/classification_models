from classification_models import get_submodules_from_kwargs
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

class SelfAttention2D(keras.layers.Layer):
    def __init__(self, depth_k, depth_v, num_heads, relative, **kwargs):
        """
        Applies attention augmentation on a convolutional layer output.

        Args:
        depth_k : Depth of k (int)
        depth_v : Depth of v (int)
        num_heads : Num attention heads (int)
        relative: Whether to use relative embeddings (bool)

        Returns:
        Output of tensor shape
        [Batch, Height, Width, Depth_v]
        """
        super(SelfAttention2D, self).__init__(**kwargs)

        # Performs checking for MHA assumptions
        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                              'Given depth_k = %d, num_heads = %d' % (
                              depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                              'Given depth_v = %d, num_heads = %d' % (
                                  depth_v, num_heads))


        # Initialize necessary variables
        self.dk = depth_k
        self.dv = depth_v
        self.nh = num_heads
        self.relative = relative
        self.dkh = self.dk // self.nh
        self.dvh = self.dv // self.nh

        # Initialize the necessary layers
        self.conv_kqv = layers.Conv2D(filters = 2*self.dk + self.dv,kernel_size = 1,padding = "same") # Convolutional layer to produce KQV matrix
        self.conv_project = layers.Conv2D(filters = self.dv,kernel_size=1,padding ="same") # Convolutional layer of size 1 to project attention layer to size of filter
        self.bn = layers.BatchNormalization()
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        self._shape = input_shape
        self.B, self.H, self.W, self.d = input_shape

        if self.relative:
            self.rel_embeddings_w = self.add_weight('rel_embeddings_w',shape=(2 * self.W - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)
            self.rel_embeddings_h = self.add_weight('rel_embeddings_h',shape=(2 * self.H - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)


    def call(self,inputs,**kwargs):
        # Input = [1,16,16,64] 
        # dk = 24, dv = 24
        flatten_hw = lambda x,d: tf.reshape(x, [-1, self.nh, self.H*self.W,d])
        # Compute q, k, v matrix 

        kqv = self.conv_kqv(inputs) # [1,16,16,72]
        k, q, v = tf.split(kqv,[self.dk,self.dk,self.dv],axis = -1) # [1,16,16,24] for k q and v
        # Rescale the value of q
        q *= (self.dkh ** -0.5)

        # Splits a tensor with shape [batch, num_heads, height, width, channels] 
        # to a tensor with shape [batch,num_heads,height,width,channels/num_heads]

        q = self.split_heads_2d(q,self.nh)
        k = self.split_heads_2d(k,self.nh)
        v = self.split_heads_2d(v,self.nh)
        # [B,Nh,HW,HW]
        logits = tf.matmul(flatten_hw(q,self.dkh),flatten_hw(k,self.dkh),transpose_b= True)

        if self.relative:
            rel_logits_h, rel_logits_w = self.relative_logits(q,self.H,self.W,self.nh)

            logits += rel_logits_h
            logits += rel_logits_w


        weights = self.softmax(logits)
        attn_out = tf.matmul(weights, flatten_hw(v,self.dvh))
        attn_out = tf.reshape(attn_out,[-1,self.nh,self.H,self.W,self.dvh])
        attn_out = self.combine_heads_2d(attn_out)

        # Project heads

        out = self.conv_project(attn_out)

        return out

    def shape_list(self,x):
        """
        Returns a list of dimensions

        Arguments:
        x : A keras tensor    
        """

        static = x.get_shape().as_list()
        shape = tf.shape(x)
        ret = []
        for i, static_dim in enumerate(static):
            dim = static_dim or shape[i]
            ret.append(dim)

        return ret


    def split_heads_2d(self,inputs,Nh):
        """ Split channels into multiple heads """
        B, H, W, d = self.shape_list(inputs)
        ret_shape = [B,H,W,Nh,d//Nh]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0,3,1,2,4])

    def combine_heads_2d(self, inputs):
        """ Combine heads (inverse of split_heads_2d)."""
        transposed = tf.transpose(inputs,[0,2,3,1,4])
        Nh, channels = self.shape_list(transposed)[-2:]
        ret_shape = self.shape_list(transposed)[:-2] + [Nh * channels]
        return tf.reshape(transposed,ret_shape)

    def rel_to_abs(self,x):
        """ Converts tensor from relative to absolute indexing. """
        # [B, Nh, L, 2L-1]
        B, Nh, L, _ = self.shape_list(x)
        # Pad to shift from relative to absolute indexing
        col_pad = tf.zeros((B,Nh,L,1))
        x = tf.concat([x,col_pad],axis = 3)
        flat_x = tf.reshape(x, [B,Nh,L*2*L])
        flat_pad = tf.zeros((B,Nh,L-1))
        flat_x_padded = tf.concat([flat_x,flat_pad],axis = 2)
        # Reshape and slice out the padded elements
        final_x = tf.reshape(flat_x_padded, [B,Nh,L+1,2*L-1])
        final_x = final_x[:,:,:L,L-1:]
        return final_x

    def relative_logits_1d(self,q,rel_k,H,W,Nh,transpose_mask):
        """ Compute relative logits along H or W """

        rel_logits = tf.einsum("bhxyd,md->bhxym",q,rel_k)
        # Collapse height and heads
        rel_logits = tf.reshape(rel_logits, [-1,Nh*H,W,2 * W-1])
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it and tile height times
        rel_logits = tf.reshape(rel_logits, [-1, Nh,H,W,W])
        rel_logits = tf.expand_dims(rel_logits, axis = 3)
        rel_logits = tf.tile(rel_logits,[1,1,1,H,1,1])
        # Reshape for adding to the logits
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1,Nh,H*W,H*W])
        return rel_logits

    def relative_logits(self,q,H,W,Nh):
        """ Compute relative logits """

        rel_logits_w = self.relative_logits_1d(q,self.rel_embeddings_w,H,W,Nh,[0,1,2,4,3,5])
        rel_logits_h = self.relative_logits_1d(q,self.rel_embeddings_h,W,H,Nh,[0,1,4,2,5,3])

        # [B, Nh, HW, HW]
        return rel_logits_h, rel_logits_w

    def get_config(self):
        config = {
            "dk" : self.dk,
            "dv" : self.dv,
            "nh" : self.nh,
            "filters" : self.filters,
            "kernel_size" : self.kernel_size,
            "relative" : self.relative,
            "downsample" : self.downsample,
            "dkh" : self.dkh,
            "dvh" : self.dvh
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def AugmentedConv2d(  filters,
                      kernel_size,
                      Rk = 0.25,
                      Rv = 0.25,
                      Nh = 8,
                      relative = True):
    
    def layer(input_tensor):
        ei = lambda x : int(np.ceil(x/Nh)*Nh)
        dk = ei(filters*Rk)
        dv = ei(filters*Rv)
        conv_out = layers.Conv2D(filters = filters-dv,kernel_size = kernel_size, padding = "same")(input_tensor)
        attn_out = SelfAttention2D(dk,dv,Nh,relative)(input_tensor)
        out = layers.Concatenate()([conv_out,attn_out])

        return out
    
    return layer

def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).

    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.

    """

    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = layers.Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = layers.Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = layers.Concatenate(axis=slice_axis)(blocks)
        return x

    return layer


def expand_dims(x, channels_axis):
    if channels_axis == 3:
        return x[:, None, None, :]
    elif channels_axis == 1:
        return x[:, :, None, None]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(channels_axis))


def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = layers.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('sigmoid')(x)

        # apply attention
        x = layers.Multiply()([input_tensor, x])

        return x

    return layer
