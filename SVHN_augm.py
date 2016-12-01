# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:18:58 2016

@author: atolmid
Identifies numbers from the SVHN google dataset, using known bounding boxes.
Uses additional data created through data augmentation.
Starting code was taken from assignment 4 of the Udacity Deep Learning course
"""

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
#import process_data as pd

train_dataset = None
train_labels = None
valid_dataset = None
valid_labels = None
test_dataset = None
test_labels = None
            
pickle_train = 'train.pickle'
pickle_test = 'test.pickle'
image_size = 128
num_labels = 11
num_channels = 1 # grayscale
num_digits = 5

batch_size = 32
test_batch_size = 100
patch_size = 5
depth = 16
num_hidden = 500

model_name = "CNN_parameters_augmented_data-numhidden500_trainshape0_steps_double_dropout0.9_0.5_learning_decay_6xdata_batch32_imsize128-adam.ckpt"

graph = tf.Graph()

# The files where the original training data, and the ones acquired through data augmrntation are stored
train_pickles = ['train0.pickle', 'train1.pickle', 'train2.pickle', 'train3.pickle', 'train4.pickle', 'train5.pickle']

def get_pickle(stored_pickle, dataset):
    # load the training/validation dataset from the train pickle 
    with open(stored_pickle, 'rb') as f:
    #train, test1, _ = ds.getDigitStruct()
        save = pickle.load(f)
        if dataset == "train":
            train_dataset = save['train_dataset']
            train_labels = np.array(save['train_labels'])
            valid_dataset = save['valid_dataset']
            valid_labels = np.array(save['valid_labels'])
            print('Training set: ', train_dataset.shape, train_labels.shape)
            print('Validation set: ', valid_dataset.shape, valid_labels.shape)
            #delete save to free up memory
            del save
            return train_dataset, train_labels, valid_dataset, valid_labels
        else:
            test_dataset = save['test_dataset']
            test_labels = np.array(save['test_labels'])
            print('Test dataset: ', test_dataset.shape, test_labels.shape)
            #delete save to free up memory
            del save
        return test_dataset, test_labels
    
  
    
"""
Reformat into a TensorFlow-friendly shape:
convolutions need the image data formatted as a cube (width by height by #channels)
labels as float 1-hot encodings.
"""
def reformat(dataset, labels):
    labi = [None]*5
    dataset = dataset.reshape((-1, image_size, image_size, num_channels))
    for i in range(5):
        lab = labels[:,i]
        labi[i] = (np.arange(num_labels) == lab[:,None]).astype(np.float32)
    lab = np.array(labi)
    return dataset, lab

train_dataset, train_labels, valid_dataset, valid_labels = get_pickle(train_pickles[0], 'train')
test_dataset, test_labels = get_pickle('test.pickle', 'test')
# reformat the dataset/labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

            
            
# define how accuracy is measured
def accuracy(predictions, labels):
    # for each prediction, each digit predicted is compared to the respective label
    # accuracy is the percentage of succesful predictions (all 5 digits have been successfully identified)
    p = []
    l = []
    for i in range(num_digits):
        p.append(np.argmax(predictions[i], 1))
        l.append(np.argmax(labels[i], 1))
    p1 = np.array(p).T
    l1 = np.array(l).T
    ac = [np.sum(p1[i][b] == l1[i][b] for b in range(num_digits)) for i in range(p1.shape[0]) ]
    acc = 1.0 * ac.count(5)/np.array(ac).shape[0]
    return 100*acc 
    
def avg_accuracy(predictions, labels):
    # returns the mean accuracy of the batches included in predictions/labels
    return np.mean([accuracy(predictions[i].eval(), labels[i]) for i in range(len(predictions))])
        

# define the model
with graph.as_default():
    
    global_step = tf.Variable(0)  # count the number of steps taken.
    # training input data
    tf_train_dataset =  tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(num_digits, batch_size, num_labels))

    # split the validation data into batches of size test_batch_size
    val_data1 = [valid_dataset[x:x+test_batch_size, :, :, :] for x in range(0,(valid_dataset.shape[0]), test_batch_size)]
    tf_valid_dataset = [tf.constant(val_dat) for val_dat in val_data1]
    
    # list of label batches, of size test_batch_size each
    val_labels1 = [valid_labels[:, x:x+test_batch_size, :] for x in range(0,(valid_labels.shape[1]), test_batch_size)]
    
    # split the testing data into batches of size test_batch_size
    test_data1 = [test_dataset[x:x+test_batch_size, :, :, :] for x in range(0,(test_dataset.shape[0]), test_batch_size)]
    tf_test_dataset = [tf.constant(test_dat) for test_dat in test_data1]

    # list of label batches, of size test_batch_size each
    test_labels1 = [test_labels[:, x:x+test_batch_size, :] for x in range(0,(test_labels.shape[1]), test_batch_size)]
    
    #probability that a neuron's output is kept during dropout
    keep_prob = tf.placeholder(tf.float32)
    # variables
    # 4 convolution layers
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth],  stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    
    layer4_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    # 2 fully connected layers for each digit
    # dimentions in the truncated normal function are height of image fed to the layer*width of image fed to the layer*depth, 
    # number of nodes in the hidden layer
    # first digit
    layer5a_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1)) 
    layer5a_biases = tf.Variable(tf.constant(1.0, shape=([num_hidden])))
    layer6a_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer6a_biases = tf.Variable(tf.constant(1.0, shape=([num_labels])))
    # second digit
    layer5b_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1))
    layer5b_biases = tf.Variable(tf.constant(1.0, shape=([num_hidden])))
    layer6b_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer6b_biases = tf.Variable(tf.constant(1.0, shape=([num_labels])))
    # third digit
    layer5c_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1))
    layer5c_biases = tf.Variable(tf.constant(1.0, shape=([num_hidden])))
    layer6c_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer6c_biases = tf.Variable(tf.constant(1.0, shape=([num_labels])))
    # fourth digit 
    layer5d_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1))
    layer5d_biases = tf.Variable(tf.constant(1.0, shape=([num_hidden])))
    layer6d_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer6d_biases = tf.Variable(tf.constant(1.0, shape=([num_labels])))
    # fifth digit
    layer5e_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1))
    layer5e_biases = tf.Variable(tf.constant(1.0, shape=([num_hidden])))
    layer6e_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer6e_biases = tf.Variable(tf.constant(1.0, shape=([num_labels])))


    # the model used
    def model(data, keep_prob):
        # run a convolution with stride = 1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        # send the convolution output plus the bias through a relu function
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
        # second convolution layer, stride = 1
        conv2 = tf.nn.conv2d(hidden1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        # relu function
        hidden2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
        # third convolution layer
        conv3 = tf.nn.conv2d(hidden2, layer3_weights, [1, 1, 1, 1], padding='SAME')
        # third relu function
        hidden3 = tf.nn.relu(tf.nn.bias_add(conv3, layer3_biases))
        # fourth convolution layer
        conv4 = tf.nn.conv2d(hidden3, layer4_weights, [1, 1, 1, 1], padding='SAME')
        # fourth relu function
        hidden4 = tf.nn.relu(tf.nn.bias_add(conv4, layer4_biases))
        # add a max pooling function and dropout
        pool1 = tf.nn.dropout(tf.nn.max_pool(hidden4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME'), keep_prob)
        # add a max pooling function
        #pool1 = tf.nn.max_pool(hidden4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # reshape
        reshape = tf.reshape(pool1, [-1, (image_size/2) * (image_size/2) * depth])
        # fully connected layers (different for each digit)
        # first digit
        hidden5a = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer5a_weights)+layer5a_biases), keep_prob)
        logit_a = tf.matmul(hidden5a, layer6a_weights)+layer6a_biases
        # second digit
        hidden5b = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer5b_weights)+layer5b_biases), keep_prob)
        logit_b = tf.matmul(hidden5b, layer6b_weights)+layer6b_biases
        # third digit
        hidden5c = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer5c_weights)+layer5c_biases), keep_prob)
        logit_c = tf.matmul(hidden5c, layer6c_weights)+layer6c_biases
        # fourth digit
        hidden5d = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer5d_weights)+layer5d_biases), keep_prob)
        logit_d = tf.matmul(hidden5d, layer6d_weights)+layer6d_biases
        # fifth digit
        hidden5e = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer5e_weights)+layer5e_biases), keep_prob)
        logit_e = tf.matmul(hidden5e, layer6e_weights)+layer6e_biases
        
        #return the logits
        return [logit_a, logit_b, logit_c, logit_d, logit_e]
        
    # Training computation
    logits = model(tf_train_dataset, keep_prob= 1.0) #no dropout
    # loss is the mean softmax cross entropy for each of the digits 
    loss =tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits(logits[i], tf_train_labels[i, :, :]) for i in range(num_digits)])
    # optimizer is Adam optimizer, with learning rate 1e-4
    #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    #define a decaying learning rate
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1, 0.96)        
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # Predictions for the train, validation, and test data
    train_prediction = [tf.nn.softmax(logits[i]) for i in range(num_digits)]
    valid_prediction = [tf.reshape([tf.nn.softmax(model(valid, keep_prob= 1.0)[i]) for i in range(num_digits)], [num_digits, -1, num_labels]) for valid in tf_valid_dataset]
    test_prediction = [tf.reshape([tf.nn.softmax(model(test, keep_prob= 1.0)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])  for test in tf_test_dataset]

    # save model
    saver = tf.train.Saver()


#num_steps = train_dataset.shape[0]


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for i in range (0, len(train_pickles)):
        train_dataset, train_labels, valid_dataset, valid_labels = get_pickle(train_pickles[i], 'train')
        # reformat the dataset/labels
        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        # print the training/validation dataset shape
        print('Training dataset: ', train_dataset.shape, train_labels.shape)
        print('Validation dataset: ', valid_dataset.shape, valid_labels.shape)
  
        print('Pickle No ', i)
        # go through all the data once
        for step in range(train_dataset.shape[0]):
          global_step = tf.add(global_step,1)
        
          # calculate offset
          offset = (step * batch_size) % (train_labels.shape[1] - batch_size)
          # select batch data
          batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
          # select batch labels
          batch_labels = train_labels[:, offset:(offset + batch_size), :]
          # run optimizer, make  prediction, calculate loss
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        
          _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
          # every 5000 steps calculate and print validation accuracy
          if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(np.array(predictions), batch_labels))
          if (step % 5000 == 0):
            print('Validation accuracy: %.1f%%' % avg_accuracy(
              valid_prediction, val_labels1))
        # save model
        save_path = saver.save(session, model_name)
        print("Model saved in file: %s" % save_path)

    
    # print the test dataset shape
    print('Test dataset: ', test_dataset.shape, test_labels.shape)
    # calculate and print test accuracy
    print('Test accuracy: %.1f%%' % avg_accuracy(test_prediction, test_labels1))
    