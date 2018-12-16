import nn
import tensorflow as tf


def test_value():
    value_function_params = {
        'hidden_layer_sizes': (128, 128),
    }
    value_function = nn.ValueFunction(
        name='value_function', **value_function_params)

    observation_ph = tf.placeholder(tf.float32, shape=(None, 5))

    try:
        print(value_function(observation_ph))
    except:
        print('value_function(observation_ph) falied')
    try:
        print(value_function((observation_ph, )))
    except:
        print('value_function((observation_ph, )) failed')


def test_q():
    q_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    q_function = nn.QFunction(
        name='value_function', **q_function_params)

    observation_ph = tf.placeholder(tf.float32, shape=(None, 5))
    action_ph = tf.placeholder(tf.float32, shape=(None, 3))

    try:
        print(q_function(observation_ph, action_ph))
    except:
        print('q_function(observation_ph) falied')
    try:
        print(q_function((observation_ph, action_ph)))
    except:
        print('q_function((observation_ph, )) failed')


class NestedFunc(object):
    def test_nested_func(self):
        self.b = 0
        def nest_func():
            self.b = 1
        nest_func()
        print(self.b)


def test_nested_func():
    nested_func = NestedFunc()
    nested_func.test_nested_func()


if __name__ == '__main__':
    # test_value()
    # test_q()
    test_nested_func()
