import numpy as np
import glob
import sys
import caffe


input_inference_dir= "Input/Inference/CI/"
input_inference_data_dir= input_inference_dir + "Data/"
input_inference_model_dir= input_inference_dir + "Model/"

numero_monitor = int(sys.argv[1])
tipo_dataset = sys.argv[2]

caffe.set_mode_cpu() 

model_def1 = input_inference_model_dir +'resnet20_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights1=input_inference_model_dir+ 'resnet20_cifar10.caffemodel' 
net1 = caffe.Net(model_def1, model_weights1,caffe.TEST)

model_def2 = input_inference_model_dir +'resnet32_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights2 =input_inference_model_dir +'resnet32_cifar10.caffemodel' 
net2 = caffe.Net(model_def2, model_weights2,caffe.TEST)

model_def3 = input_inference_model_dir+ 'resnet56_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights3 = input_inference_model_dir+ 'resnet56_cifar10.caffemodel' 
net3 = caffe.Net(model_def3, model_weights3,caffe.TEST)

mean= np.array([0.49139,0.48215,0.44653])

transformer1 = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
transformer1.set_transpose('data', (2,0,1)) 
transformer1.set_mean('data', mean)  
transformer1.set_raw_scale('data',1) 

transformer2 = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
transformer2.set_transpose('data', (2,0,1)) 
transformer2.set_mean('data', mean)  
transformer2.set_raw_scale('data',1)   

transformer3 = caffe.io.Transformer({'data': net3.blobs['data'].data.shape})
transformer3.set_transpose('data', (2,0,1)) 
transformer3.set_mean('data', mean)  
transformer3.set_raw_scale('data',255)  

#images= glob.glob(''+str(tipo_dataset)+'_dataset_cifar10/*.png')

#for image in images:
    #with open(image, 'rb') as file :
        
image1= caffe.io.load_image(input_inference_data_dir + str(tipo_dataset)+'_dataset_cifar10/ae1.png')
image2= caffe.io.load_image(input_inference_data_dir +str(tipo_dataset)+'_dataset_cifar10/ae1.png')
image3= caffe.io.load_image(input_inference_data_dir +str(tipo_dataset)+'_dataset_cifar10/ae1.png')
transformed_image1 = transformer1.preprocess('data', image1)
transformed_image2 = transformer2.preprocess('data', image2)
transformed_image3 = transformer3.preprocess('data', image3)
net1.blobs['data'].data[...] =transformed_image1
net2.blobs['data'].data[...] =transformed_image2
net3.blobs['data'].data[...] =transformed_image3
output1 = net1.forward()
output2 = net2.forward()
output3 = net3.forward()
output_prob1 = output1['prob'][0] 
output_prob2 = output2['prob'][0] 
output_prob3 = output3['prob'][0] 
print ('predicted class is net1 :', output_prob1.argmax())
print ('predicted class is net2 :', output_prob2.argmax())
print ('predicted class is net3 :', output_prob3.argmax())



