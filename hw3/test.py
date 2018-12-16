import tensorflow as tf


def test_mask():
    a = tf.constant([0, 2])
    b = tf.constant([[1, 2, 3], [4, 5, 6]])
    mask = tf.one_hot(a, 3)
    print(a, mask)
    b_mask = tf.boolean_mask(b, tf.one_hot(a, 3))

    with tf.Session() as sess:
        print(sess.run(b_mask))


if __name__ == '__main__':
    test_mask()
