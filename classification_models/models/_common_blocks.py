from classification_models import get_submodules_from_kwargs
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import initializers


def cbam_block(ratio=16, **kwargs):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	def layer(input_tensor):
		output = channel_attention(input_tensor, ratio)
		output = spatial_attention(output)
    
		return output
	return layer

def channel_attention(input_tensor, ratio=16):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_tensor.shape[channel_axis]
	
	shared_layer_one = layers.Dense(channel//ratio,activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')    
	shared_layer_two = layers.Dense(channel,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')	
    
    # Use average pooling layers
	avg_pool = layers.GlobalAveragePooling2D()(input_tensor)    
	avg_pool = layers.Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
    # Use max pooling layers
	max_pool = layers.GlobalMaxPooling2D()(input_tensor)
	max_pool = layers.Reshape((1,1,channel))(max_pool) # The reshaping is used for python broadcasting
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = layers.Add()([avg_pool,max_pool])
	cbam_feature = layers.Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
	
	return layers.Multiply()([input_feature,cbam_feature]) # Output of F'

def spatial_attention(input_tensor):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_tensor.shape[1]
		cbam_feature = layers.Permute((2,3,1))(input_tensor)
	else:
		channel = input_tensor.shape[-1]
		cbam_feature = input_tensor
	
	avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='glorot_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
		
	return layers.Multiply()([input_tensor,cbam_feature]) # Output of F''


class SelfAttention(keras.layers.Layer):
    def __init__(self,
                 stage=None,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        
        self.convk_name = 'decoder_stage{}_k'.format(stage)
        self.convq_name = 'decoder_stage{}_q'.format(stage)
        self.convv_name = 'decoder_stage{}_v'.format(stage)
               
        self.softmax = layers.Activation('softmax')
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        
        self._shape = input_shape
        _, self.h, self.w, self.filters = input_shape
        
        self.conv_k = layers.Conv2D(self.filters//8, 1, use_bias=False, kernel_initializer='he_normal',name=self.convk_name)
        self.conv_q = layers.Conv2D(self.filters//8, 1, use_bias=False, kernel_initializer='he_normal',name=self.convq_name)
        self.conv_v = layers.Conv2D(self.filters, 1, use_bias=False, kernel_initializer='he_normal',name=self.convv_name)
        
        self.built = True

    def call(self, input_tensor):

        self.k = self.conv_k(input_tensor)
        self.q = self.conv_q(input_tensor)
        self.v = self.conv_v(input_tensor)
        self.k = K.reshape(self.k,(-1,self.h*self.w,self.filters//8)) # [B,HW,f]
        self.q = tf.transpose(K.reshape(self.q, (-1, self.h*self.w, self.filters // 8)), (0, 2, 1))
        self.logits = K.batch_dot(self.k, self.q)
        self.xd = self.softmax(self.logits)
        self.v = K.reshape(self.v, (-1, self.h*self.w, self.filters))
        self.attn = K.batch_dot(self.xd, self.v) # [B,Hw,f]
        self.attn = K.reshape(self.attn, (-1, self.h, self.w, self.filters))

        self.out = self.gamma*self.attn + input_tensor
        return self.out
    
    def get_config(self):
        config = {
            "gamma" : self.gamma,
            "k" : self.k,
            "q" : self.q,
            "v" : self.v,
            "attn" : self.attn,
            "w": self.xd,
            "out" : self.out
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class AttentionAugmented2D(keras.layers.Layer):
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
        super(AttentionAugmented2D, self).__init__(**kwargs)

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

    def build(self, input_shape):
        self._shape = input_shape
        self.B, self.H, self.W, self.d = input_shape

        if self.relative:
            self.rel_embeddings_w = self.add_weight('rel_embeddings_w',shape=(2 * self.W - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)
            self.rel_embeddings_h = self.add_weight('rel_embeddings_h',shape=(2 * self.H - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)


    def call(self,inputs,**kwargs):
        # Input is the KQV matrix
        # dk = 24, dv = 24
        flatten_hw = lambda x,d: tf.reshape(x, [-1, self.nh, self.H*self.W,d])
        
        # Compute q, k, v matrix 
        k, q, v = tf.split(inputs,[self.dk,self.dk,self.dv],axis = -1) # [1,16,16,24] for k q and v
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

        weights = tf.nn.softmax(logits)
        attn_out = tf.matmul(weights, flatten_hw(v,self.dvh))
        attn_out = tf.reshape(attn_out,[-1,self.nh,self.H,self.W,self.dvh])
        attn_out = self.combine_heads_2d(attn_out) # Output shape = [B,H,W,dv]

        return attn_out

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
    
		
def AugmentedConv2D(  f_out,
                      kernel_size,
                      Rk = 0.25,
                      Rv = 0.25,
                      Nh = 8,
                      relative = True,
                      stage = None,
                      block = None):
    
    kqv_name = f'stage{stage}_unit{block}_kqv'
    normal_name = f'stage{stage}_unit{block}_normal'
    projection_name = f'stage{stage}_unit{block}_projection'
    
    def layer(input_tensor):
        ei = lambda x : int(np.ceil(x/Nh)*Nh)
        dk = ei(f_out*Rk)
        dv = ei(f_out*Rv)
        
        # Normal convolution
        conv_out = layers.Conv2D(filters = f_out-dv, kernel_size = kernel_size, padding = "same",name=normal_name)(input_tensor)
        
        # Convolution for the KQV matrix
        kqv = layers.Conv2D(filters = 2*dk + dv,kernel_size = 1,padding = "same",kernel_initializer="he_normal",name=kqv_name)(input_tensor)
        # Calculate the Multi Headed Attention and concatenates all the heads
        attn_out = AttentionAugmented2D(dk,dv,Nh,relative)(kqv)
        # Project the result of MHA 
        attn_out = layers.Conv2D(filters = dv,kernel_size=1,padding ="same", kernel_initializer="he_normal",name=projection_name)(attn_out)
           
        out = layers.Concatenate()([conv_out,attn_out])
       	out = layers.BatchNormalization()(out)
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
