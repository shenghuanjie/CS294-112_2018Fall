import tensorflow as tf
import numpy as np

'''
a = tf.placeholder(shape=[None, 1], dtype=tf.int32)
with tf.Session() as sess:
    b = sess.run([a], feed_dict={a: np.ones((10, 1))})
    print(b)
    print(b[0])
    print(sess.run(a, feed_dict={a: np.ones((10, 1))}))


a = np.array([1, 2, 3])
print(a.shape)
print(np.atleast_2d(a).shape)
'''
'''
a = []
a.append(np.tile(10, 10))
a.append(np.tile(1, 10))
print(np.array(a).flatten())

bs_p = tf.constant(np.arange(1, 10))
mean_bs_p, var_bs_p = tf.nn.moments(bs_p, axes=[0])
norma_bs_p = tf.nn.batch_normalization(bs_p, mean_bs_p, var_bs_p, None, None, 1e-10)
'''

'''

def modify_args(args, **kwargs):
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args
'''


def test_reward_to_go(gamma=0.1):
    rp = np.arange(1, 10, 1)
    len_rp = len(rp)
    discounts = np.geomspace(1, gamma ** len_rp, num=len_rp)
    q_n = (np.flip(np.cumsum(np.flip(rp * discounts, 0)), 0))
    print(q_n)


if __name__ == '__main__':

    test_reward_to_go()

    '''
    
    b = 10
    a = 0.0087213987213897219387128937129837
    print('a:{0:d}, b:{1:.2g}'.format(b, a).replace('.', '-'))
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')

    args = parser.parse_args(['addd'])
    print(args)

    modify_args(args, exp_name='test')
    print(args)
    '''
