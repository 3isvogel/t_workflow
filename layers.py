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

try:
    import tensorflow as tf
except Exception as e:
    raise ValueError(f'Tensorflow not installed: {e}')

class SelfAttention(tf.keras.Model):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    

try:
    import keras
except Exception as e:
    raise ValueError(f'Keras not installed: {e}')
from keras.layers import Conv2D, Dropout

def BaseConv2D(filters: int,
               kernel_size: tuple[int,int] = (4,4),
               padding: str = 'same',
               dropout: float = 0.25):
    return pipe([Conv2D(filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              activation = 'relu'),
                 Dropout(dropout)])