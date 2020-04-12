import matplotlib.pyplot as plt


class Plots(object):

    def plot_train(self, train_x, train_y, sess_w, sess_b):
        plt.plot(train_x, train_y, 'ro', label='Train data')
        plt.plot(train_x, sess_w * train_x + sess_b, label='Linear Regression')
        plt.legend()
        plt.show()

    def plot_test(self, train_x, test_x, test_y, sess_w, sess_b):
        plt.plot(test_x, test_y, 'bo', label='Test data')
        plt.plot(train_x, sess_w * train_x + sess_b, label='Linear Regression')
        plt.legend()
        plt.show()
