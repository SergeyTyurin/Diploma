import caffe
import os

DiplomaDIR = os.path.dirname(__file__)
deploy_proto = os.path.join(DiplomaDIR,"deploy.prototxt")
weights = os.path.join(DiplomaDIR,"snapshots","classifier_iter_10000.caffemodel")
net = caffe.Net(deploy_proto, weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]


image = caffe.io.load_image('/Users/sergeytyurin/Desktop/104.png')
net.blobs['data'].data[...] = transformer.preprocess('data',image)

caffe.set_mode_cpu()
output = net.forward()

output_prob = output['prob']
Class1 = {'Private Home':0}
Class2 = {'City Building':1}
Class3 = {'Bridge':2}

Classes = {0:'Private Home', 1:'City Building', 2:'Bridge' }

print(output_prob.argmax())
print(Classes[output_prob.argmax()])
