# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf
import numpy as np

test_filename = 'xtrain_obfuscated.txt'

def load_yelp(alphabet):
    examples = []
    labels = []
    with open('xtrain_obfuscated.txt') as f:
        for text in f:
                text_end_extracted = extract_end(list(text.lower()))
                padded = pad_sentence(text_end_extracted)
                text_int8_repr = string_to_int8_conversion(padded, alphabet)
                examples.append(text_int8_repr)
                
    with open('ytrain.txt') as l:
        for text in l:
            if int(text) == 0:
                labels.append([1,0,0,0,0,0,0,0,0,0,0,0])
            elif int(text) == 1:
                labels.append([0,1,0,0,0,0,0,0,0,0,0,0])
            elif int(text) == 2:
                labels.append([0,0,1,0,0,0,0,0,0,0,0,0])
            elif int(text) == 3:
                labels.append([0,0,0,1,0,0,0,0,0,0,0,0])
            elif int(text) == 4:
                labels.append([0,0,0,0,1,0,0,0,0,0,0,0])
            elif int(text) == 5:
                labels.append([0,0,0,0,0,1,0,0,0,0,0,0])
            elif int(text) == 6:
                labels.append([0,0,0,0,0,0,1,0,0,0,0,0])
            elif int(text) == 7:
                labels.append([0,0,0,0,0,0,0,1,0,0,0,0])
            elif int(text) == 8:
                labels.append([0,0,0,0,0,0,0,0,1,0,0,0])
            elif int(text) == 9:
                labels.append([0,0,0,0,0,0,0,0,0,1,0,0])
            elif int(text) == 10:
                labels.append([0,0,0,0,0,0,0,0,0,0,1,0])
            elif int(text) == 11:
                labels.append([0,0,0,0,0,0,0,0,0,0,0,1])
            else:
                break                     
    return examples, labels


def extract_end(char_seq):
    if len(char_seq) > 500:
        char_seq = char_seq[-500:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 500
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    if len(x_batch) == 0:
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), 0, 1])
    else:
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data():
    # TODO Add the new line character later for the yelp'cause it's a multi-line review
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    examples, labels = load_yelp(alphabet)
    # print(labels)
    x = np.array(examples, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    print("x_char_seq_ind=" + str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            batch = list(zip(x_batch, y_batch))
            yield batch
