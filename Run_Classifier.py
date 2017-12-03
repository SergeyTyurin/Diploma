# -*- coding: utf-8 -*-
import caffe
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, zero_one_loss
import numpy as np
import pandas as pd
import time
import math

filePath = os.path.dirname(__file__)
path_to_model = os.path.join(filePath,"Classifier")
deploy_proto = os.path.join(path_to_model,"var1","deploy.prototxt")
weights = os.path.join(path_to_model,"var1","snapshots","classifier_iter_8000.caffemodel")

net = caffe.Net(deploy_proto, weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

imagesTest = os.path.join(path_to_model,"mactest.txt")
imagesTrain = os.path.join(path_to_model,"mactrain.txt")
imagesVal = os.path.join(path_to_model,"macval.txt")

caffe.set_mode_cpu()

def CrossEntropyLoss(yy_true, yy_predict):
    num=0
    loss = 0
    for true,predict in zip(yy_true,yy_predict):
        loss+= -math.log(predict)
        num+=1
    return loss/num

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

def ForwardNet(path_to_images, color=True):
    y_predict = []  # предсказанные значения
    y_true = []  # эталонные значения (labels)
    y_predict_float = []
    runtime_times = []
    file = open(path_to_images, 'r')
    images = file.readlines()

    for line in images:
        imagepath, label = line.strip('\n').split(' ')
        image = caffe.io.load_image(imagepath, color=color)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        start = time.clock()
        output = net.forward()
        stop = time.clock()
        output_prob = output['prob']
        predict = output_prob.argmax()
        predict_float = max(output_prob[0][:3])
        y_predict.append(predict)
        y_true.append(int(label))
        y_predict_float.append(predict_float)
        runtime_times.append(stop - start)
    return y_true,y_predict,y_predict_float,runtime_times


def TrainAccuracyLoss():
    imagesTrain = os.path.join(path_to_model, "mactrain.txt")
    y_true, \
    y_predict, \
    y_predict_float, \
    runtime_times = ForwardNet(imagesTrain,False)
    report = classification_report(y_true, y_predict)
    print "Train Accuracy = ",accuracy_score(y_true,y_predict)
    print "Train Loss = ",zero_one_loss(y_true,y_predict)
    print "Train Loss Cross Entropy = ", CrossEntropyLoss(y_true, y_predict_float)
    print report

def ValAccuracyLoss():
    imagesVal = os.path.join(path_to_model, "macval.txt")
    y_true, \
    y_predict, \
    y_predict_float, \
    runtime_times = ForwardNet(imagesVal,False)
    report = classification_report(y_true, y_predict)
    print "Val Accuracy = ", accuracy_score(y_true, y_predict)
    print "Val Loss = ", zero_one_loss(y_true, y_predict)
    print "Val Loss Cross Entropy = ", CrossEntropyLoss(y_true, y_predict_float)
    print report

def TestAccuracyLossPrecisionRecall():
    imagesTest = os.path.join(path_to_model, "mactest.txt")
    y_true,\
    y_predict,\
    y_predict_float,\
    runtime_times = ForwardNet(imagesTest, False)

    # print classes
    precision,recall,f1_score,support = precision_recall_fscore_support(y_true,y_predict)

    precision = np.append(precision,np.mean(precision))
    recall = np.append(recall,np.mean(recall))
    f1_score = np.append(f1_score,np.mean(f1_score))
    support = np.append(support,np.sum(support))
    report = classification_report(y_true,y_predict)
    print "Avg Time = ",np.mean(runtime_times)
    print "Test Accuracy = ", accuracy_score(y_true, y_predict)
    print "Test Loss = ", zero_one_loss(y_true, y_predict)
    print "Test Loss Cross Entropy = ", CrossEntropyLoss(y_true, y_predict_float)

    print report
    #classifaction_report_csv(report)


if __name__=='__main__':
    TrainAccuracyLoss()
    ValAccuracyLoss()
    TestAccuracyLossPrecisionRecall()