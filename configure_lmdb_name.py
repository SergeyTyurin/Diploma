# -*- coding: utf-8 -*-
import os

lmdb_var = 'Large_256_gray'
fileNames = ['train.txt','val.txt','test.txt']
modelName = os.path.join('Classifier') #Указать путь до папки с prototxt

try:
    for fileName in fileNames:
        if fileName=='test.txt' and  not os.path.exists(os.path.join(os.getenv("LMDBDataSetDIR"),lmdb_var,fileName)):
            correctPaths=[]
            for root,dirs,files in os.walk(os.getenv("TestDataSetDIR")):
                for file in files:
                    if (file.split('.')[1] == 'png'):
                        correctPaths.append("{0} {1}\n".format(os.path.join(root,file),int(root.split('/')[-1])-1))
            newFileName = 'mactest.txt'
            f1 = open(os.path.join(os.getenv("DiplomaDIR"), modelName, newFileName), 'w')
            f1.writelines(correctPaths)
            f1.close()

        else:
            f=open(os.path.join(os.getenv("LMDBDataSetDIR"),lmdb_var,fileName),'r')
            #print f.read()
            correctPaths=[]
            for line in f.readlines():
                l = line.split('/')
                correctPaths.append(os.path.join(os.getenv("DataSetDIR"),'256',l[-2],l[-1]))
            f.close()
            newFileName = 'mac' + fileName
            f1 = open(os.path.join(os.getenv("DiplomaDIR"),modelName,newFileName),'w')
            f1.writelines(correctPaths)
            f1.close()
except Exception as e:
    print e
