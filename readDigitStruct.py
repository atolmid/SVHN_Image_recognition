# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:14:16 2016

@author: atolmid

#!/usr/bin/python

# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python 

# starting code was taken from https://github.com/sarahrn/Py-Gsvhn-DigitStruct-Reader/blob/master/digitStruct.py
"""

import h5py

#
# Bounding Box
#
class BBox:
    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None # List of BBox structs

# Function for debugging
def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print "{}".format(theObjName)
    print "    type(): {}".format(type(theObj))
    if isFile or isGroup or isDataSet:
        print "    id: {}".format(theObj.id)
    if isFile or isGroup:
        print "    keys: {}".format(theObj.keys())
    if not isReference:
        print "    Len: {}".format(len(theObj))

    if not (isFile or isGroup or isDataSet or isReference):
        print theObj

def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup

#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal 

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj

def getDigitStruct():
    # .mat files for training and test images 
    trainFileName = './train/digitStruct.mat'
    testFileName = './test/digitStruct.mat'
    # create data srtucture for training images
    # create an empty dictionary where the data will be stored
    print('generating the training dataset')
    trainDataset = {}
    # for each of the images in the training set,
    # get all labels that are included,
    # as well as the position of their bounding boxes in the image
    for dsObj in yieldNextDigitStruct(trainFileName):
        # create an empty temporary dictionary for each label
        item = {}
        # for each label store in the dictionary the label,
        #along with its bounding box position
        for bbox in dsObj.bboxList:
            item[bbox.label] = [bbox.left, bbox.top, bbox.width, bbox.height]
        # add the temporary dictionary of each image, to the dataset dictionary
        trainDataset[dsObj.name] = item
    # create data srtucture for testing images
    # create an empty dictionary where the data will be stored
    print('generating the testing dataset')
    testDataset = {}
    # for each of the images in the testing set,
    # get all labels that are included,
    # as well as the position of their bounding boxes in the image
    for dsObj in yieldNextDigitStruct(testFileName):
        # create an empty temporary dictionary for each label
        item = {}
        # for each label store in the dictionary the label,
        #along with its bounding box position
        for bbox in dsObj.bboxList:
            item[bbox.label] = [bbox.left, bbox.top, bbox.width, bbox.height]
        # add the temporary dictionary of each image, to the dataset dictionary
        testDataset[dsObj.name] = item
    # return the two datasets
    return trainDataset, testDataset

#def testMain():
#    train, test = getDigitStruct()
#    for i in range(5):
#        print('train ', i , 'name : ', str(i+1)+'.png', 'elements: ', train[str(i+1)+'.png'])
#        print('test ', i , 'name : ', str(i+1)+'.png', 'elements: ', test[str(i+1)+'.png'])
