"""

Predict house's price using tensorflow (1.14.0).

Functions:
    np.asarray - transform input to array
"""

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from plots import Plots

if __name__ == '__main__':
    p = Plots()
    # Hyper params
    learning_rate = 0.01
    training_epochs = 2000
    display_step = 200

    # Train data set: x: house length, y: house price
    train_x = np.asarray(
        [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_y = np.asarray(
        [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    # Test data set
    test_x = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    # Number of train samples
    n_samples = train_x.shape[0]

    # Placeholders x (input)  y (output)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # Weight and bias
    w = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")

    # Building linear model: y = w*x + b
    linear_model = w * x + b

    # Mean squared error
    cost = tf.reduce_sum(tf.square(linear_model - y)) / (2 * n_samples)

    # Optimizer with Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Init variables
    init = tf.global_variables_initializer()

    # Open session
    with tf.Session() as sess:
        # Init variables
        sess.run(init)

        # Training model
        # number of trainings (used to find w and b)
        for epoch in range(training_epochs):

            # Optimizer with Gradient Descent
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})

            # Display each epoch
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={x: train_x, y: train_y})
                print("Epoch:{0:6} \t Cost/Error:{1:10.4} \t W:{2:6.4} \t b:{3:6.4}".format(epoch + 1, c, sess.run(w),
                                                                                            sess.run(b)))

        # Print final model params
        print("\nOptimization finished!")
        training_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
        print("Train cost:", training_cost, " - w final:", sess.run(w), " - b final:", sess.run(b))

        p.plot_train(train_x, train_y, sess.run(w), sess.run(b))

        # Testing model
        testing_cost = sess.run(tf.reduce_sum(tf.square(linear_model - y)) / (2 * test_x.shape[0]),
                                feed_dict={x: test_x, y: test_y})

        print("Test cost:", testing_cost)
        print("Absolute difference train cost and test cost:", abs(training_cost - testing_cost))

        p.plot_test(train_x, test_x, test_y, sess.run(w), sess.run(b))

        # Example of predict y for x = 4.6
        print("\nA house with 4.6 lenght will cost: $%.2f." % sess.run(linear_model, feed_dict={x: 4.6}))

    sess.close()
