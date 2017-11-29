# -*- coding: utf-8 -*-
import caffe
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, zero_one_loss
import numpy as np
import pandas as pd
import time

filePath = os.path.dirname(__file__)
deploy_proto = os.path.join(filePath,"var1","new_deploy.prototxt")
weights = os.path.join(filePath,"var1","snapshots","tut_classifier_iter_2000.caffemodel")
net = caffe.Net(deploy_proto, weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

imagesTest = os.getenv("TestDataSetDIR")
caffe.set_mode_cpu()

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    out_lines=[]
    for line in lines:
        if len(line) ==0:
            pass
        else:
            out_lines.append(line)
    #print out_lines
    for line in out_lines[1:]:
        row = {}
        row_data = []
        for l in line.split('  '):
            if len(l)!=0:
                row_data.append(l)
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = int(row_data[4])
        report_data.append(row)
    print report_data
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(os.path.join(filePath,'var1','classification_report.csv'), index = False)

def TrainAccuracyLoss():
    y_predict = []  # предсказанные значения
    y_true = []  # эталонные значения (labels)
    runtime_times = []
    a = 0
    b = 0
    train_file = open('mactrain.txt','r')
    imagesTrain = train_file.readlines()
    for line in imagesTrain:
        imagepath,label = line.strip('\n').split(' ')
        #net.blobs['data'].reshape(1,3,64,64)
        image = caffe.io.load_image(imagepath)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        #caffe.set_mode_cpu()
        start = time.clock()
        output = net.forward()
        stop = time.clock()
        output_prob = output['prob'][0]
        #print(output_prob[:3])
        predict = output_prob.argmax()
        if(int(label)>0):
            print(predict, int(label), predict==int(label), imagepath)
            if predict==int(label):
                a+=1
            if predict!=int(label):
                b+=1
        y_predict.append(predict)
        y_true.append(int(label))
        runtime_times.append(stop - start)
    print(a,b)
    print "Train Accuracy = ",accuracy_score(y_true,y_predict)
    print "Train Loss = ",zero_one_loss(y_true,y_predict)

def ValAccuracyLoss():
    y_predict = []  # предсказанные значения
    y_true = []  # эталонные значения (labels)
    runtime_times = []

    train_file = open('macval.txt', 'r')
    imagesTrain = train_file.readlines()
    for line in imagesTrain:
        imagepath,label = line.strip('\n').split(' ')
        #net.blobs['data'].reshape(1, 3, 64, 64)
        image = caffe.io.load_image(imagepath)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        start = time.clock()
        output = net.forward()
        stop = time.clock()
        output_prob = output['prob']
        #print net.blobs['score']
        print(output_prob[0][0][0])
        predict = output_prob.argmax()
        y_predict.append(predict)
        y_true.append(int(label))
        runtime_times.append(stop - start)
    print "Val Accuracy = ", accuracy_score(y_true, y_predict)
    print "Val Loss = ", zero_one_loss(y_true, y_predict)

def TestAccuracyLossPrecisionRecall():
    y_predict = []  # предсказанные значения
    y_true = []  # эталонные значения (labels)
    runtime_times = []

    train_file = open('mactest.txt', 'r')
    imagesTrain = train_file.readlines()
    for line in imagesTrain:
        imagepath, label = line.strip('\n').split(' ')
        # net.blobs['data'].reshape(1, 3, 64, 64)
        image = caffe.io.load_image(imagepath)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        start = time.clock()
        output = net.forward()
        stop = time.clock()
        output_prob = output['prob']
        # print net.blobs['score']
        #print(output_prob[0][0][0])
        predict = output_prob.argmax()
        y_predict.append(predict)
        y_true.append(int(label))
        runtime_times.append(stop - start)
    print "Test Accuracy = ", accuracy_score(y_true, y_predict)
    print "Test Loss = ", zero_one_loss(y_true, y_predict)

# print classes

    precision,recall,f1_score,support = precision_recall_fscore_support(y_true,y_predict)

    precision = np.append(precision,np.mean(precision))
    recall = np.append(recall,np.mean(recall))
    f1_score = np.append(f1_score,np.mean(f1_score))
    support = np.append(support,np.sum(support))

    report = classification_report(y_true,y_predict)

    print report
    # print "Avg Time = ",np.mean(runtime_times)

    # print "Test Accuracy = ", accuracy_score(y_true, y_predict)
    # print "Test Loss = ", zero_one_loss(y_true, y_predict)
    #classifaction_report_csv(report)

if __name__=='__main__':
    # TrainAccuracyLoss()
    # ValAccuracyLoss()
    TestAccuracyLossPrecisionRecall()
