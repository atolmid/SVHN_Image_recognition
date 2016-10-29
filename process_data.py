# -*- coding: utf-8 -*-
"""
@author: atolmid
# starting code was taken from assignment 1 of the Udacity Deep Learning course
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from PIL import Image
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from operator import itemgetter
import readDigitStruct as ds


train_url = 'http://ufldl.stanford.edu/housenumbers/'
test_url = 'http://ufldl.stanford.edu/housenumbers/'
extra_url = 'http://ufldl.stanford.edu/housenumbers/'

last_percent_reported = None
num_classes = 11
np.random.seed(133)
image_width = 128  # Pixel width. 
image_height = 128 # Pixel height.
pixel_depth = 255.0  # Number of levels per pixel
#set up the backend for matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"


def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(url, filename, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  print(statinfo)
  return filename


def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  return root
  

train_filename = maybe_download(train_url, 'train.tar.gz')
test_filename = maybe_download(test_url, 'test.tar.gz')
extra_filename = maybe_download(extra_url, 'extra.tar.gz')

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)


def load_picture(folder, im_data, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  # dataset that will contain the images
  # size is len(image_files) - 2, since two of the files are the matlab files
  # and the rest are the images, named with a sequential number
  dataset = np.ndarray(shape=(len(image_files)-2, image_width, image_height),
                         dtype=np.float32)
  num_images = len(image_files)-2
  #last 2 files are matlab files
  #we don't need them
  data_labels = []
  for i in range(1, num_images+1):
      try:
          # get the name of the image as a string
          name = str(i)+'.png'
          image_file = os.path.join(folder, name)
          # open the image and covert it to grayscale
          image = Image.open(image_file).convert('L')
          plt.imshow(image)
          #get the data for the specific image from the details dictionary
          # (the one containing the numbers and the bounding boxes for each number in each image)
          data = im_data[name]
          l_data = [data[im] for im in data]
          labels = [en for en in enumerate([im for im in data])]
          left = [en for en in sorted(enumerate(l_dat[0] for l_dat in l_data), key=itemgetter(1))]
          ind = [n[0] for n in left]
          labels = [labels[n][1] for n in ind]
          # find the bounding nox that contains all numbers in the image
          # left and top point
          start_x = min([l_dat[0] for l_dat in l_data])
          start_y = min([l_dat[1] for l_dat in l_data])
          # width
          # get a tuple containing the sublist with the max left element, and the element itself
          m1 = max(enumerate(l_dat[0] for l_dat in l_data), key=itemgetter(1))          
          # add the hight of the element with the maximum left to its left, 
          # to find the boundary point for the combined number bounding box
          # that is most to the right
          end_x = m1[1] + l_data[m1[0]][2]
          # height
          # get a tuple containing the sublist with the max top element, and the element itself
          m2 = max(enumerate(l_dat[1] for l_dat in l_data), key=itemgetter(1))
          # add the hight of the element with the maximum top to the top, 
          # to find the boundary point for the combined number bounding box
          # that is lowest
          end_y = m2[1] + l_data[m2[0]][3]
          # from the above calculations, define a bounding box around the numbers
          box = (start_x, start_y, end_x, end_y)
          # crop the image and only keep the area that contains the numbers
          image = image.crop(box)
          plt.imshow(image)
          # resize the image to size 128x128
          image = image.resize((128, 128))
          plt.imshow(image)
          # get the image data
          # center the data around 0, and normalize
          image_data = ((np.array(image.getdata(), np.float32).reshape(image.size[1], image.size[0])) - pixel_depth / 2) / pixel_depth
          plt.imshow(image)
          # add the image to the dataset
          dataset[i-1, :, :] = image_data
          # add the labels to the labels list
          # create a list with size 5-the number of digits of the number we are detecting
          l = [0]*(5-len(labels))
          #  the list that contains the labels will be the concatenation of the list with None
          # and the list with the actual digits.
          # Therefore, the the number of labels will always be 5, and the actual digits will always be
          # at the end of the list
          # Thus, we can use the list elements in this turn, as separate labels to a NN that has
          # to predict them. If the number is not 5-digits long, the first "numbers" will be 
          # '10' (instead of None), and so there is no need to predict a sixth variable that 
          # represents the number length
          labels = l + labels
          data_labels.append(labels)
      except IOError as e:
          print('Could not read:', ':', e, '- it\'s ok, skipping.')
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
  if folder == 'train':
      train_dataset = dataset[0:int(0.9*num_images), :, :]
      valid_dataset = dataset[int(0.9*num_images):num_images, :, :]
      train_labels = data_labels[0:int(0.9*num_images)]
      valid_labels = data_labels[int(0.9*num_images):num_images]
      #print('train: ', train_dataset[0])
      #print('train_l: ', train_labels[0])
      plt.imshow(train_dataset[0])
      plt.imshow(valid_dataset[-1])
      the_dataset = {'train_dataset':train_dataset, 'valid_dataset':valid_dataset, 'train_labels':train_labels, 'valid_labels':valid_labels}
  elif folder == 'test':
      the_dataset = {'test_dataset':dataset, 'test_labels':data_labels}
  else:
      the_dataset = {'extra_dataset':dataset, 'extra_labels':data_labels}
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return the_dataset
      
        
def maybe_pickle(data_folders, im_data, min_num_images_per_class, force=True):
  dataset_names = []
  print('data_folders', data_folders)
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_picture(folder, im_data, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  return dataset_names#, data_labels

train, test, extra = ds.getDigitStruct()
train_datasets = maybe_pickle([train_folders],train,  1000)
test_datasets = maybe_pickle([test_folders], test, 1800)
extra_datasets = maybe_pickle([extra_folders], extra, 1800)