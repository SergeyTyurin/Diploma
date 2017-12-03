# -*- coding: utf-8 -*-
import os

lmdb_var = '256_gray'
fileName = 'val.txt'
modelName = os.path.join('Classifier') #Указать путь до папки с prototxt

try:
    f=open(os.path.join(os.getenv("LMDBDataSetDIR"),lmdb_var,fileName),'r')
    #print f.read()
    correctPaths=[]
    for line in f.readlines():
        l = line.split('/')
        correctPaths.append(os.path.join(os.getenv("DataSetDIR"),l[-2],l[-1]))
    f.close()
    newFileName = 'mac' + fileName
    f1 = open(os.path.join(os.getenv("DiplomaDIR"),modelName,newFileName),'w')
    f1.writelines(correctPaths)
    f1.close()
except Exception as e:
    print e
