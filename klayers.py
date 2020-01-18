import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.callbacks as kc
import tensorflow.keras.backend as kb
import tensorflow.keras.regularizers as kr
import tensorflow.keras.optimizers as ko

def Attention(name='Attention'):
    """ Attention Layer for the inputs with arbitrary shapes with a last feature dim, 
        like an RGB image with shape of `M x N x 3` or `K x 3`.
    """
    def func(inputs):
        dims = len(inputs.shape)
        per = list(range(2, dims))+[1]
        a = kl.Permute(per, name='%s_permute0'%name)(inputs)
        probs = kl.Dense(inputs.shape[1], activation='softmax', name='%s_fc'%name)(a)
        per = [dims-1] + list(range(1, dims-1))
        probs = kl.Permute(per, name='%s_permute1'%name)(probs)
        outputs = kl.Multiply(name='%s_out'%name)([inputs, probs])
        return outputs, probs
    return func

def MLP(shapes, drop_rate=None, activation='relu', name='MLP'):
    """ MLP layer with Dense and Dropout layers.
    """
    if type(shapes) is int: shapes = [shapes]
    def func(x):
        for i,n in enumerate(shapes):
            x = kl.Dense(n, activation=activation, name='{}_fc{}'.format(name,i))(x)
            if drop_rate:
                x = kl.Dropout(rate=drop_rate, name='{}_dropout{}'.format(name,i))(x)
        return x
    return func

def Classifier(shapes, drop_rate=0, activation='relu' , name='Classifier'):
    """ Classifier layer contains a MLP layer followed by a Dense with unitis of number of classes.
    """
    if type(shapes) is int: shapes = [shapes]
    def func(x):
        x = MLP(shapes[:-1], drop_rate, activation=activation, name='%s_MLP'%name)(x)
        act = 'sigmoid' if shapes[-1]==1 else 'softmax'
        x = kl.Dense(shapes[-1], activation=act, name='{}_output'.format(name))(x)
        return x
    return func 

def Reduce(op, axis, keepdims=False, name='Reduce'):
    """ Reduce layer.

    Args:
        op: 'min', 'max', 'mean', 'sum' or 'prod'.
        axis: the axis to do the reduce operation.
        keepdims: Optional. Defaults to False.
    Returns:
        A function of Keras Layers.
    """
    reduce_func = {'min': tf.reduce_min, 
                    'max': tf.reduce_max, 
                    'mean': tf.reduce_mean,
                    'sum': tf.reduce_sum,
                    'prod': tf.reduce_prod,
                    }
    assert op in reduce_func, 'Invalid OP name: %s'%op
    def func(inputs):
        return kl.Lambda(lambda x:reduce_func[op](x, axis=axis, keepdims=keepdims), name=name)(inputs)
    return func

if __name__ == "__main__":
    inputs = kl.Input(shape=(10,5,4,3))
    x = MLP([8,6], drop_rate=0.5)(inputs)
    x, p = Attention()(x)
    print(x.shape, p.shape)
    x = Reduce('sum', axis=-1)(x)
    print(x.shape)
    outputs = Classifier(3)(x)
    model = km.Model(inputs, outputs)
    # model.summary()