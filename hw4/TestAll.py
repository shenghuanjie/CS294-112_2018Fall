import tensorflow as tf
import numpy as np


def test_yield():
    for i in range(3):
        yield 1 + i, 2 + i


def test_dim():
    import numpy as np
    a = np.array([1, 2, 3])
    b = np.atleast_2d(a)
    print(b[0].shape)


class TestProperty(object):

    def __init__(self, a):
        self.a = a

    @property
    def test(self):
        return self.a


def test_tf_slice():
    t = tf.constant([[1, 1, 1], [2, 2, 2],
                     [3, 3, 3], [4, 4, 4]])
    t_slice = tf.slice(t, [0, 0], [4, 1])
    print(t_slice)
    with tf.Session() as sess:
        print(sess.run(t_slice))


def test_tf_stack():
    cost = []
    cost.append([[1], [2], [3], [4]])
    cost.append([[5], [6], [7], [8]])
    cost.append([[9], [10], [11], [12]])
    tf_cost = tf.stack(cost)
    print(tf_cost)
    with tf.Session() as sess:
        print(sess.run(tf_cost))


def test_random_uniform():
    n = 5
    mins = [-1, 1, 3]
    maxes = [1, 2, 4]
    test_i = tf.random_uniform((2, n, 3), mins, maxes)
    print(test_i[0])
    with tf.Session() as sess:
        print(sess.run(test_i))


def test_tf_tile():
    a = tf.constant([1, 2, 3])
    a = tf.reshape(a, (1, tf.size(a)))
    b = tf.tile(a, [5, 1])
    print(b)
    with tf.Session() as sess:
        print(sess.run(b))


def test_build_mlp():
    import utils
    input_layer = tf.placeholder(tf.float32, (None, 10))
    new_layer = utils.build_mlp(input_layer,
                                5,
                                "test",
                                n_layers=2,
                                hidden_dim=500,
                                activation=tf.nn.relu,
                                output_activation=None,
                                reuse=False)
    print(new_layer)
    print('---------------------')
    reuse_layer = utils.build_mlp(input_layer,
                                  5,
                                  "test",
                                  n_layers=2,
                                  hidden_dim=500,
                                  activation=tf.nn.relu,
                                  output_activation=None,
                                  reuse=True)
    print(reuse_layer)
    print(new_layer == reuse_layer)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        org, reuse = sess.run([new_layer, reuse_layer],
                              feed_dict={input_layer: np.atleast_2d(np.arange(0, 10))})
        print('org: ', org)
        print('reuse: ', reuse)


def test_tf_zeros():
    a = tf.zeros(5)
    print(a)


def test_tf_concat():
    norm_state = tf.constant([[1, 2, 3], [4, 5, 6]])
    norm_action = tf.constant([[7, 8], [9, 10]])
    norm_state_action = tf.concat([norm_state, norm_action], 1)
    print(norm_state_action)
    with tf.Session() as sess:
        print(sess.run(norm_state_action))


def test_tf_multiply():
    norm_state = tf.constant([[1, 2, 3], [4, 5, 6]])
    with tf.Session() as sess:
        print(sess.run(norm_state * [1, 2, 3]))


def test_sort():
    a = [1, 2, 3, 4, 5]
    b = [5, 3, 2, 1, 4]
    print([x for _, x in sorted(zip(b, a), reverse=True)])


def test_truncnorm():
    from scipy.stats import truncnorm
    samples = []
    for i in range(3):
        dist = truncnorm(-1, 1, loc=0.5)
        samples.append(dist.rvs(10))

    print(np.array(samples).shape)
    print(np.array(samples))


def sortXonY(self, x, y, reverse=False):
        return [ix for _, ix in sorted(zip(y, x), reverse=reverse)]


def _cross_entropy_action_selection(self, state_ph):

    N = 100 # batch sample size
    Ne = N / 10 # elite number
    state_ph = tf.tile(state_ph, (N, 1))

    mu = (self._action_space_low + self._action_space_high) / 2
    sigma2 = (self._action_space_high - self._action_space_low) ** 2 / 12
    epsilon = np.finfo(np.float32).eps
    t = 0

    all_first_actions = []

    while t < self._num_random_action_selection and np.all(sigma2 > epsilon):

        if t + N > self._num_random_action_selection:
            N = self._num_random_action_selection - t

        current_state = state_ph
        # Obtain N samples from current sampling distribution
        sample_actions = self.truncnorm_action(N, mu, sigma2)
        all_first_actions.append(sample_actions)
        # Evaluate objective function at sampled points
        S = 0
        for i_step in range(self._horizon):
            next_state_pred = self._dynamics_func(current_state, sample_actions, reuse=True)
            S += self._cost_fn(current_state, sample_actions, next_state_pred)
            current_state = next_state_pred
            sample_actions = self.truncnorm_action(N, mu, sigma2)
        # Sort X by objective function values in descending order
        X = tf.gather(X, tf.contrib.framework.argsort(S))
        # Update parameters of sampling distribution
        mu, sigma2 = tf.nn.moments(X[1:Ne], axes=[1])
        t += N

    return mu

def truncnorm_action(self, num, mu, sigma2):
    sample_actions = []
    assert(len(mu) == self._action_dim)
    assert (len(sigma2) == self._action_dim)
    for lower, upper, mean, variance in zip(self._action_space_low, self._action_space_high, mu, sigma2):
        dist = truncnorm((lower - mu) / sigma2, (upper - mu) / sigma2, loc=mean, scale=variance)
        sample_actions.append(dist.rvs(size=num))
    return np.array(sample_actions).T


def test_clip_by_value():
    a = tf.constant([[1, 2], [3, 4], [5, 6]])
    b = tf.clip_by_value(a, [-1, 4], [2, 5])
    with tf.Session() as sess:
        print(sess.run(b))


def test_slice():
    a = tf.constant([[1, 2], [3, 4], [5, 6]])
    b = tf.slice(a, begin=[0, 0], size=[1, 2])
    with tf.Session() as sess:
        print(sess.run(b))


def test_reshape():
    a = tf.constant([[[1, 2, 3], [3, 4, 5], [5, 6, 7]], [[7, 8, 9], [9, 10, 11], [11, 12, 13]]])
    print(a.get_shape())
    b = tf.reshape(a, shape=(-1, 3))
    with tf.Session() as sess:
        print(sess.run(b))


def test_tf_stack2():
    a = []
    a.append(tf.constant([1, 2, 3]))
    a.append(tf.constant([1, 2, 3]))
    b = tf.concat(a, 0)
    with tf.Session() as sess:
        print(sess.run(b))


def test_tf_stack3():
    a = []
    a.append(tf.constant([[1, 2, 3], [4, 5, 6]]))
    a.append(tf.constant([[1, 2, 3], [4, 5, 6]]))
    b = tf.concat(a, 0)
    with tf.Session() as sess:
        print(sess.run(b))


if __name__ == '__main__':
    """
    put your test function here
    """
    """
    x = test_yield()
    while True:
        a, b = next(x, [None, None])
        if a is None:
            break
        else:
            print(a, b)
    """
    # test_dim()
    # test = TestProperty(1)
    # print(test.test())
    # test_tf_slice()
    # test_tf_stack()
    # test_random_uniform()
    # test_tf_tile()
    # test_build_mlp()
    # test_tf_zeros()
    # test_tf_concat()
    # test_tf_multiply()
    # test_sort()
    # test_truncnorm()
    # test_clip_by_value()
    # test_slice()
    # test_reshape()
    test_tf_stack3()
