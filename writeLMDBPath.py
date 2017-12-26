from caffe.proto import caffe_pb2
import google.protobuf.text_format
import os
import argparse

def writeCorrectPath(model, lmdb_path):
    fullpath = lmdb_path
    net = caffe_pb2.NetParameter()
    #print net
    f = open(model, 'r')
    #print f.read()
    net = google.protobuf.text_format.Merge(str(f.read()), net)
    f.close()
    for layer in net.ListFields()[0][1]:
        if layer.type=="Data":
            if layer.include[0].phase==1:
                layer.data_param.source= os.path.join(fullpath,"train_db")
            elif layer.include[0].phase==0:
                layer.data_param.source = os.path.join(fullpath, "val_db")
    f1 = open(model,'w')
    f1.write(str(net))
    f1.close()

if __name__=='__main__':
    # parser = argparse.ArgumentParser(
    #      description="Write correct paths to lmdb"
    # )
    # parser.add_argument("--model",type=str,default='train_val.prototxt',required=True);
    # parser.add_argument("--lmdb", type=str, default=None,
    #                    required=True)
    #
    # args = parser.parse_args()
    path = "/Users/vladislavtyurin/Diploma/DataSet/lmdb"
    writeCorrectPath("/Users/vladislavtyurin/Diploma/Diploma/Classifier/var1/train_val.prototxt", os.path.join(path, "64"))
