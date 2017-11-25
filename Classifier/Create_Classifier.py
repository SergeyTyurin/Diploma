# -*- coding: utf-8 -*-
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import caffe
import os

# Шаблон создания prototxt
# Не учитывает путь к lmdb и фазы входного слоя
def create_net(lmdb_dir, batch_size):
    # Создаем prototxt своего классификатора
    classifier = caffe.NetSpec()

    classifier.data, classifier.label = L.Data(batch_size = batch_size,
                                               backend = P.Data.LMDB,
                                               source = lmdb_dir,
                                               transform_param = dict(scale = 1. / 255),
                                               ntop = 2) # входной слой

    classifier.conv1 = L.Convolution(classifier.data,
                                     kernel_size = 2,
                                     num_output = 20,
                                     weight_filler = dict(type = 'xavier'))

    classifier.pool1 = L.Pooling(classifier.conv1,
                                 kernel_size = 2,
                                 stride = 2,
                                 pool = P.Pooling.MAX)

    classifier.conv2 = L.Convolution(classifier.pool1,
                                     kernel_size = 4,
                                     num_output = 40,
                                     weight_filler = dict(type = 'xavier'))

    classifier.relu2 = L.ReLU(classifier.conv2,
                              in_place = True)

    classifier.pool2 = L.Pooling(classifier.relu2,
                                 kernel_size = 2,
                                 stride = 2,
                                 pool = P.Pooling.MAX)
    classifier.fc1 = L.InnerProduct(classifier.pool2,
                                    num_output = 500,
                                    weight_filler = dict(type = 'xavier'))

    classifier.relu3 = L.ReLU(classifier.fc1,
                              in_place = True)

    classifier.score = L.InnerProduct(classifier.relu3,
                                      num_output = 3,
                                      weight_filler = dict(type = 'xavier'))

    classifier.loss = L.SoftmaxWithLoss(classifier.score,
                                        classifier.label)

    return classifier.to_proto()

# шаблон создания solver
def create_solver(dir):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = "path_to_protoxt"
    s.test_interval = 500
    s.test_iter.append(100)
    s.max_iter = 10000
    s.type = "SGD"
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4

    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 1000

    s.snapshot = 5000
    s.snapshot_prefix = 'classifier'
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(os.path.join(dir,'solver.prototxt'), 'w') as f:
        f.write(str(s))

def main():
    current_dir = os.path.dirname(__file__)
    lmdb_dir = 'file_path'
    proto = str(create_net(lmdb_dir, 32))

    # with open(os.path.join(current_dir,'train_val.prototxt'),'w') as f:
    #     f.write(proto)

    create_solver(current_dir)


if __name__=="__main__":
    main()