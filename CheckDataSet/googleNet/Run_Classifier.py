# -*- coding: utf-8 -*-
import caffe
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

filePath = os.path.dirname(__file__)
deploy_proto = os.path.join(filePath,"var1","deploy.prototxt")
weights = os.path.join(filePath,"var1","snapshots","_iter_6000.caffemodel")
net = caffe.Net(deploy_proto, weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

imagesTest = os.getenv("TestDataSetDIR")

classes=[] # матрица ошибок (TP, FP, TN, FN)
y_predict=[] # предсказанные значения
y_true=[] # эталонные значения (labels)
runtime_times=[]

for root,dirs,files in os.walk(imagesTest):
    for d in dirs:
        classes.append([int(d),0,0]) # Формируем пустые столбцы матрицы ошибок
    for file in files:
        if(file.split('.')[1]=='png'):
            print os.path.join(root,file)
            image = caffe.io.load_image(os.path.join(root,file),color=False)
            net.blobs['data'].data[...] = transformer.preprocess('data',image)
            caffe.set_mode_cpu()
            start = time.clock()
            output = net.forward()
            stop = time.clock()
            output_prob = output['prob']
            predict = output_prob[:2].argmax()
            label = int(root[-1])-1
            print(predict,label)
            classes[label][predict+1]+=1
            y_predict.append(predict)
            y_true.append(label)
            runtime_times.append(stop-start)

# print classes

precision,recall,f1_score,support = precision_recall_fscore_support(y_true,y_predict)

precision = np.append(precision,np.mean(precision))
recall = np.append(recall,np.mean(recall))
f1_score = np.append(f1_score,np.mean(f1_score))
support = np.append(support,np.sum(support))

# print precision
# print recall
# print f1_score
# print support

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

report = classification_report(y_true,y_predict)

print report
print np.mean(runtime_times)
#classifaction_report_csv(report)




