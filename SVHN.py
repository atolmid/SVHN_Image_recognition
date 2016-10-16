# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:50:13 2016

@author: atolmid
"""

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
#from operator import mul

pickle_train = 'train.pickle'
pickle_test = 'test.pickle'
image_size = 128
num_labels = 11
num_channels = 1 # grayscale
num_digits = 5

batch_size = 64
patch_size = 5
depth = 16
num_hidden = 1024

graph = tf.Graph()

# load the training/validation dataset from the train pickle 
with open(pickle_train, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = np.array(save['train_labels'])
    valid_dataset = save['valid_dataset']
    valid_labels = np.array(save['valid_labels'])
    #delete save to free up memory
    del save
    #print(train_labels)
    print('Training set: ', train_dataset.shape, train_labels.shape)
    print('Validation set: ', valid_dataset.shape, valid_labels.shape)
 
# load the testing dataset from the test pickle   
with open(pickle_test, 'rb') as f:
    save = pickle.load(f)
    test_dataset = save['test_dataset']
    test_labels = np.array(save['test_labels'])
    del save
    print('Test dataset: ', test_dataset.shape, test_labels.shape)
    
    
"""
Reformat into a TensorFlow-friendly shape:
convolutions need the image data formatted as a cube (width by height by #channels)
labels as float 1-hot encodings.
"""

def reformat(dataset, labels):
    labi = [None]*5
    dataset = dataset.reshape((-1, image_size, image_size, num_channels))#.astype(np.float32)
    for i in range(5):
        lab = labels[:,i]
        #print('lab[',i,']  : ', lab)
        #print('lab :::: :', lab[:,None])
        labi[i] = (np.arange(num_labels) == lab[:,None]).astype(np.float32)
        #print('labi[',i,'] shape : ', labi[i])
        #for col in labels[:, i]:
        #    print(labels[:, i])
        #    print(np.arange(num_labels))
        #    print([(np.arange(num_labels) == k) for k in col])
        #    lab[i] = (np.arange(num_labels) == labels[:,i]).astype(np.float32)
    lab = np.array(labi)
    #print('lab shape :', lab.shape)
    return dataset, lab
    
train_dataset, train_labels = reformat(train_dataset, train_labels)
#print('Training label: ', train_labels[0])
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training dataset: ', train_dataset.shape, train_labels.shape)
print('Validation dataset: ', valid_dataset.shape, valid_labels.shape)
print('Test dataset: ', test_dataset.shape, test_labels.shape)

# definehow accuracy is measured
def accuracy(predictions, labels):
    #for b in range(predictions.shape[1]):
     #   print('predictions.shape[1] : ', predictions.shape[1])
     #   print('label : ', labels[:][b][:])
     #   ac1 = [(predictions[i][b][:]) == labels[i][b][:] for i in range(num_digits)]
     #   print('ac1 : ', ac1) 
    p = []
    l = []
    for i in range(num_digits):
        #print('predictions[', i , '] : ', predictions[i])
        #print('labels[', i , '] : ', labels[i])
        p.append(np.argmax(predictions[i], 1))
        l.append(np.argmax(labels[i], 1))
        #print('np.argmax(predictions[i], 1)[', i , '] : ', np.argmax(predictions[i], 1))
        #print('np.argmax(labels[i], 1)[', i , '] : ', np.argmax(labels[i], 1))
    p1 = np.array(p).T
    l1 = np.array(l).T
    #print('p: ', np.array(p).T)
    #print('l: ', np.array(l).T)
    """
    for i in range(batch_size):
        for b in range(num_digits):
            print('p1[',i,'][', b, '] : ', p1[i][b])
            print('l1[',i,'][', b, '] : ', l1[i][b])
    """
    ac = [np.sum(p1[i][b] == l1[i][b] for b in range(num_digits)) for i in range(batch_size) ]
    acc = 1.0 * ac.count(5)/np.array(ac).shape[0]
    #print('acc : ', acc)
    #print('nacc : ', ac)
    #print('acc : ', reduce(mul, acc))
    return 100*acc #100*reduce(mul, acc) #(100*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))/predictions.shape[0]
    


# define the model
with graph.as_default():
    # input data
    tf_train_dataset =  tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(num_digits, batch_size, num_labels))
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)
    # split validation dataset and labels into twop parts
    #data
    val_data1 = valid_dataset[0:(int(0.1*(valid_dataset.shape[0]))), :, :, :]
    tf_valid_dataset1 = tf.constant(val_data1)
    val_data2 = valid_dataset[(int(0.1*(valid_dataset.shape[0]))):, :, :, :]
    tf_valid_dataset2 = tf.constant(val_data2)
    #labels
    val_labels1 = valid_labels[:, 0:(int(0.1*(valid_labels.shape[1]))), :]
    tf_valid_labels1 = tf.constant(val_labels1)
    val_labels2 = valid_labels[:, (int(0.1*(valid_labels.shape[1]))):, :]
    tf_valid_labels2 = tf.constant(val_labels2)
    
    #split test dataset and labels into two parts
    #dataset
    test_data1 = test_dataset[0:(int(0.1*(test_dataset.shape[0]))), :, :, :]
    tf_test_dataset1 = tf.constant(test_data1)
    test_data2 = test_dataset[(int(0.1*(test_dataset.shape[0]))):, :, :, :]
    tf_test_dataset2 = tf.constant(test_data2)
    #labels
    test_labels1 = test_labels[:, 0:(int(0.1*(test_labels.shape[1]))), :]
    tf_test_labels1 = tf.constant(test_labels1)
    test_labels2 = test_labels[:, (int(0.1*(test_labels.shape[1]))):, :]
    tf_test_labels2 = tf.constant(test_labels2)
    
    # variables
    # 3 convolution layers
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth],  stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))
    #print('layer1_weights : ', layer1_weights, layer1_biases)
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    #print('layer2_weights : ', layer2_weights, layer2_biases)
    layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    layer4_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=([depth])))
    # 2 fully connected layers for each digit
    # first digit
    layer5a_weights = tf.Variable(tf.truncated_normal([(image_size/2) * (image_size/2) * depth, num_hidden], stddev=0.1)) 
    # dimentions in the truncated normal function are height of image fed to the layer*width of image fed to the layer*depth, 
    # number of nodes in the hidden layer
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
    def model(data):
        # run a convolution with stride = 1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        #print("conv 1: ", conv1)
        # send the convolution output plus the bias through a relu function
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
        #print("conv : ", hidden1)
        # second convolution layer, stride = 1
        conv2 = tf.nn.conv2d(hidden1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        #print("conv 2: ", conv2)
        # relu function
        hidden2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
        #print("conv : ", hidden2)
        # third convolution layer
        conv3 = tf.nn.conv2d(hidden2, layer3_weights, [1, 1, 1, 1], padding='SAME')
        #print("conv 3: ", conv3)
        # third relu function
        hidden3 = tf.nn.relu(tf.nn.bias_add(conv3, layer3_biases))
        #print("conv : ", hidden3)
        # fourth convolution layer
        conv4 = tf.nn.conv2d(hidden3, layer4_weights, [1, 1, 1, 1], padding='SAME')
        #print("conv 4: ", conv4)
        # fourth relu function
        hidden4 = tf.nn.relu(tf.nn.bias_add(conv4, layer4_biases))
        #print("conv : ", hidden4)
        # add a max pooling function
        pool1 = tf.nn.max_pool(hidden4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #print("pool1: ", pool1)
        #get data shape after pooling
        #shape = pool1.get_shape().as_list()
        #print("pool1 as list: ", shape)
        # reshape
        reshape = tf.reshape(pool1, [-1, (image_size/2) * (image_size/2) * depth])
        #print("reshape : ", reshape)
        # fully connected layers (different for each digit)
        # first digit
        hidden5a = tf.nn.relu(tf.matmul(reshape, layer5a_weights)+layer5a_biases)
        #print("hidden5a : ", hidden5a)
        logit_a = tf.matmul(hidden5a, layer6a_weights)+layer6a_biases
        #print("logit a: ", logit_a)
        # second digit
        hidden5b = tf.nn.relu(tf.matmul(reshape, layer5b_weights)+layer5b_biases)
        logit_b = tf.matmul(hidden5b, layer6b_weights)+layer6b_biases
        # third digit
        hidden5c = tf.nn.relu(tf.matmul(reshape, layer5c_weights)+layer5c_biases)
        logit_c = tf.matmul(hidden5c, layer6c_weights)+layer6c_biases
        # fourth digit
        hidden5d = tf.nn.relu(tf.matmul(reshape, layer5d_weights)+layer5d_biases)
        logit_d = tf.matmul(hidden5d, layer6d_weights)+layer6d_biases
        # fifth digit
        hidden5e = tf.nn.relu(tf.matmul(reshape, layer5e_weights)+layer5e_biases)
        logit_e = tf.matmul(hidden5e, layer6e_weights)+layer6e_biases
        
        #return the logits
        return [logit_a, logit_b, logit_c, logit_d, logit_e]
        
    # Training computation
    logits = model(tf_train_dataset)
    # loss is the mean softmax cross entropy for each of the digits 
    loss =tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits(logits[i], tf_train_labels[i, :, :]) for i in range(num_digits)])
    #for i in range(num_digits):
    #    losses.append(tf.nn.softmax_cross_entropy_with_logits(logits[i], tf_train_labels[i, :, :]))
    #loss = tf.reduce_mean(losses)
    #loss = tf.reduce_mean(
    #[tf.nn.softmax_cross_entropy_with_logits(logits[i], tf_train_labels[i]) for i in range(len(logits))])
    
    # optimizer is Adam optimizer, with learning rate 1e-4
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    # Predictions for the train, validation, and test data
    train_prediction = [tf.nn.softmax(logits[i]) for i in range(num_digits)]
    #valid_prediction = tf.reshape([tf.nn.softmax(model(tf_valid_dataset)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    valid_prediction1 = tf.reshape([tf.nn.softmax(model(tf_valid_dataset1)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    valid_prediction2 = tf.reshape([tf.nn.softmax(model(tf_valid_dataset2)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    #valid_predictions1 = valid_prediction[:, 0:int(0.5*valid_prediction.get_shape()[1]), :]
    #valid_predictions2 = valid_prediction[:, int(0.5*valid_prediction.get_shape()[1]):valid_prediction.get_shape()[1], :]
    #valid_labels1 = valid_labels[0:int(0.5*valid_labels.get_shape()[1])]
    #valid_labels2 = valid_labels[int(0.5*valid_labels.get_shape()[1]):valid_labels.get_shape()[1]]
    #print('valid valid_prediction', valid_predictions1)
    #print('valid _prediction', valid_prediction.shape)
    #test_prediction = tf.reshape([tf.nn.softmax(model(tf_test_dataset)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    test_prediction1 = tf.reshape([tf.nn.softmax(model(tf_test_dataset1)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    test_prediction2 = tf.reshape([tf.nn.softmax(model(tf_test_dataset2)[i]) for i in range(num_digits)], [num_digits, -1, num_labels])
    print('valid valid_prediction', valid_prediction1)
    #test_prediction = [tf.nn.softmax(model(tf_test_dataset)[i]) for i in range(num_digits)]
    

num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  #print('train_labels.shape :', train_labels.shape)
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[1] - batch_size)
    #print('offset :', train_dataset.shape)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    #print('batch data :', batch_data.shape)
    batch_labels = train_labels[:, offset:(offset + batch_size), :]
    #print('batch labels :', batch_labels.shape)
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    #print('loss: ', loss.shape)
    #print('train prediction :', train_prediction.shape)
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('predictions', np.array(predictions).shape)
      print('Minibatch accuracy: %.1f%%' % accuracy(np.array(predictions), batch_labels))
      print('valid data :', (val_data1).shape)
      #print('valid predictions', np.array(valid_prediction1).shape)
      print('valid labels', val_labels1.shape)
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction1.eval(), val_labels1))
      #print('Validation accuracy: %.1f%%' % np.mean(accuracy(
        #valid_prediction1.eval(), val_labels1), accuracy(
        #valid_prediction2.eval(), val_labels2)))
        #print('Validation accuracy: %.1f%%' % accuracy(
        #valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction1.eval(), test_labels1))
  #print('Test accuracy: %.1f%%' % accuracy(np.array(test_prediction), test_labels))
  #print('test predictions', np.array(test_prediction1).shape)
  print('test labels', test_labels1.shape)
    