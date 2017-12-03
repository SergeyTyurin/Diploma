import caffe

caffe.set_mode_cpu()
solver = caffe.get_solver('solver.prototxt')
solver.step(10)