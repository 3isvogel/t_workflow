from typing import Callable, Iterable

def _pipe_model_apply(functions: Iterable[Callable], l):
    processed = l
    for step in functions:
        processed = step(processed)
    return processed

def pipe(functions: Iterable[Callable]) -> Callable:
    """Returns a lambda that iteratively apply the specified functions to its input

    Args:
        layers (Iterable[Callable]): list of functions to apply

    Returns:
        Callable: A lambda that passes its arguments through the list of functions
    """
    return lambda x: _pipe_model_apply(functions, x)

# Please install tensorflow
import tensorflow as tf

class SelfAttention(tf.keras.Model):
    def __init__(self, score_units: int, dimension = 1):
        """Create Bahdanau attention layer

        Args:
            score_units (int): Number of units in the score computation
            dimension (int, optional): Dimension of the attention map: 1 for unidimensional, 2 for bidimensional
        Raises:
            ValuError: If `dimension` is neither 1 or 2
        """
        super(SelfAttention, self).__init__()
        if dimension not in [1,2]:
            raise ValueError('Dimensions must be either 1 or 2')
        self.score_units = score_units
        self.dimension = dimension

    def build(self, input_shape):
        self.W1 = tf.keras.layers.Dense(self.score_units)
        self.W2 = tf.keras.layers.Dense(self.score_units)
        self.V = tf.keras.layers.Dense(input_shape[0][1] if self.dimension == 2 else 1)
        super(SelfAttention, self).build(input_shape)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    
# Pleas install keras
import keras
from keras.layers import Conv2D, Dropout

def BaseConv2D(filters: int,
               kernel_size: tuple[int,int] = (4,4),
               padding: str = 'same',
               dropout: float = 0.25):
    """Concatenate a 2D convolution layer with a dropout layer

    Args:
        filters (int): 2D convolution kernels to use
        kernel_size (tuple[int,int], optional): Size of the convolution kernel. Defaults to (4,4).
        padding (str, optional): Padding to use, see Keras `Conv2D` for more informations. Defaults to 'same'.
        dropout (float, optional): Dropout portion. Defaults to 0.25.

    Returns:
        _type_: _description_
    """
    return pipe([Conv2D(filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              activation = 'relu'),
                 Dropout(dropout)])