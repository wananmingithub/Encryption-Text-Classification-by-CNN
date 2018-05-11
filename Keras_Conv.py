# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf


class CharCNN(object):
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification paper.
    """

    def __init__(self, num_classes=12, filter_sizes=(7, 7, 3, 3, 3, 3), num_filters_per_size=256,
                 l2_reg_lambda=0.0, sequence_max_length=500, num_quantized_chars=26):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        x = tf.reshape(self.input_x, [-1, 26, 500])
        x = tf.transpose(x, [0, 2, 1])
        print(x)
        x = tf.layers.conv1d(x, 256, 7, activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv1d(x, 256, 7, activation=tf.nn.relu)
        print(x)
        x = tf.layers.max_pooling1d(x, 3, 3)
        print(x)
        x = tf.layers.conv1d(x, 128, 7, activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv1d(x, 128, 7, activation=tf.nn.relu)
        print(x)
        x = tf.layers.max_pooling1d(x, 3, 3)
        print(x)

        x = tf.layers.conv1d(x, 64, 3, activation=tf.nn.relu)
        print(x)
        x = tf.layers.conv1d(x, 64, 3, activation=tf.nn.relu)
        print(x)
        x = tf.layers.max_pooling1d(x, 3, 3)
        print(x)

        x = tf.contrib.layers.flatten(x)
        print(x)
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.layers.dense(x, 1024, activation=tf.nn.sigmoid)
        print(x)
        scores = tf.layers.dense(x, 12, activation=tf.nn.sigmoid)
        print(scores)

        predictions = tf.argmax(scores, 1, name="predictions")

        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
