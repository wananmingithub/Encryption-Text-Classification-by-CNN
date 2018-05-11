# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf


class CharCNN(object):
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification paper.
    参数 filter_shape: (number of filters, num input feature maps, 
                              filter height, filter width) 
        num of filters：是卷积核的个数，有多少个卷积核，那么本层的out feature maps的个数 
        也将生成多少个。num input feature maps：输入特征图的个数。 
        然后接着filter height, filter width是卷积核的宽高，比如5*5,9*9…… 
        filter_shape是列表，因此我们可以用filter_shape[0]获取卷积核个数 
        第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
        具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
        要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    """
    def __init__(self, num_classes=12, filter_sizes=(7, 7, 3, 3, 3, 3), num_filters_per_size=256,
                 l2_reg_lambda=0.0, sequence_max_length=500, num_quantized_chars=26):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # ================ Layer 1 ================
        with tf.name_scope("conv-maxpool-1"):
            filter_shape = [num_quantized_chars, filter_sizes[0], 1, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool1")
        print("conv-maxpool-1",pooled)
        # ================ Layer 2 ================
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [1, filter_sizes[1], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool2")
        print("conv-maxpool-2",pooled)
        # ================ Layer 3 ================
        with tf.name_scope("conv-3"):
            filter_shape = [1, filter_sizes[2], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print("conv-3",h)
        # ================ Layer 4 ================
        with tf.name_scope("conv-4"):
            filter_shape = [1, filter_sizes[3], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print("conv-4",h)
        # ================ Layer 5 ================
        with tf.name_scope("conv-5"):
            filter_shape = [1, filter_sizes[4], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print("conv-5", h)
        # ================ Layer 6 ================
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [1, filter_sizes[5], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool6")
        print("conv-maxpool-6", pooled)
        # ================ Layer 7 ================
        num_features_total = 14 * num_filters_per_size
        h_pool_flat = tf.reshape(pooled, [-1, num_features_total])

        # Add dropout
        with tf.name_scope("dropout-1"):
            drop1 = tf.nn.dropout(h_pool_flat, 0.5)
        print("dropout-1", drop1)

        # Fully connected layer 1
        with tf.name_scope("fc-1"):
            W = tf.Variable(tf.truncated_normal([num_features_total, 1024], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[num_features_total, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")
        print("fc-1", fc_1_output)
        # ================ Layer 8 ================
        # Add dropout
        with tf.name_scope("dropout-2"):
            drop2 = tf.nn.dropout(fc_1_output, 0.5)

        # Fully connected layer 2
        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[1024, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

        # ================ Layer 9 ================
        # Fully connected layer 3
        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.truncated_normal([1024, num_classes], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            predictions = tf.argmax(scores, 1, name="predictions")
        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
