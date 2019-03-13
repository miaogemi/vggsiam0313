import tensorflow as tf
import numpy as np
import os

class Vgg19:
    def __init__(self,load=True,trainable=True):
        self.trainable = trainable
        self.var_dict = {}          
#            #load读取读取npy参数文件，读出文件格式为数组，数组内容是字典，类似np.array({'cov1':'【123】'},dtype=object)
#            #item（）在数组内容为字典的情况下，可以逐条读取字典内容，但在np.array([1,2,3]）情况下不行
#            #items()可以直接遍历读取字典内容
#            #功能：读取参数文件，产生{conv1_1：[参数值]}字典
#            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
#            print("npy file loaded")
#            #w = self.data_dict['conv1_1'][0](3,3,3,64)weight值
#            #b = self.data_dict['conv1_1'][1]（64）bias值
#            #print(self.data_dict['conv1_1'])#可以查看每层参数值      
        if load == True:
#        返回当前文件路径
            currpath = os.path.abspath('vgg19.py')
#        返回当前文件的目录
            fatherpath = os.path.dirname(currpath)
            parapath = os.path.join(fatherpath, "vgg19.npy")
            self.datadict = np.load(parapath,encoding='latin1').item()
            print('Parameterfile is loaded.')
        else:
            print('No parameterfile.')
            
    def get_var(self, layername, idx, var_name):
#            #w = self.data_dict['conv1_1'][0](3,3,3,64)weight值
#            #b = self.data_dict['conv1_1'][1]（64）bias值
        #数据字典加载完成后并且name在字典里存在，就赋值
        
        if self.datadict is not None and layername in self.datadict:
            value = self.datadict[layername][idx]
        else:
            print(layername+'No parameter in datadict.')

        #如果网络是可训练的，那把value赋值给ternsor变量，如果不可训练，赋值为常量
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(layername, idx)] = var
        #{('conv1', 0): array([123])}
        #{('conv1', 0): <tf.Variable 'conv1_x:0' shape=(1,) dtype=int32_ref>}

        return var   
    
    def conv_layer(self,reuse,layername,bottom,stride=1):
        
        with tf.variable_scope(layername,reuse=reuse):
            filters = self.get_var(layername=layername,idx=0,var_name=layername+'_filters')
            biases =self.get_var(layername=layername,idx=1,var_name=layername+'_bias')
            
            conv = tf.nn.conv2d(bottom,filters,strides=[1,stride,stride,1],padding='SAME')
            bias = tf.nn.bias_add(conv,biases)
            relu = tf.nn.relu(bias)
        return relu


#将全连接层看做卷积层进行计算,初始化卷积层
#    def conv_layer_init(self,reuse,layername,bottom,size,channel,num,stride=1):
##        conv1-2maxpool(1,3,3,64)
#        with tf.variable_scope(layername,reuse=reuse):
#            filters = tf.get_variable('weights',shape=[size,size,channel,num],\
#                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
#            biases = tf.get_variable('biases',shape=[num],initializer=tf.constant_initializer(0))            
#            conv = tf.nn.conv2d(bottom,filters,strides=[1,stride,stride,1],padding='VALID')
#            bias = tf.nn.bias_add(conv,biases)
#            sigmoid = tf.nn.sigmoid(bias)
#        return sigmoid
    #多层感知机，实现注意力机制
#    def mlp(self,bottom,name,reuse,stride,size1,channel1,filternum1,size2,channel2,filternum2):
#        maxpoolsize = bottom.get_shape()[2]//3
#        feature = self.max_pool(bottom,name+'_feature',poolsize=maxpoolsize,poolstride=maxpoolsize)
#        with tf.variable_scope(name,reuse):
#            mlp1 = self.conv_layer_init(reuse,name+'_1',feature,size1,channel1,filternum1)
#            mlp2 = self.conv_layer_init(reuse,name+'_2',mlp1,size2,channel2,filternum2)
#        return mlp2
        
    def max_pool(self,bottom,name,poolsize=2,poolstride=2):
        pool = tf.nn.max_pool(bottom,ksize=[1,poolsize,poolsize,1],\
                              strides=[1,poolstride,poolstride,1],padding='VALID',name=name)
        return pool

#多层感知机
    def mlp(self,bottom,layername,reuse,channel,nueron_num1,nueron_num2):
        maxpoolsize = bottom.get_shape()[2]//3
        feature = self.max_pool(bottom,layername+'_feature',poolsize=maxpoolsize,poolstride=maxpoolsize)
        with tf.variable_scope(layername,reuse=reuse):
            weights1 = tf.get_variable(layername+'_hiddenlayer',[channel*9,nueron_num1],\
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            bias1 = tf.get_variable(layername+'_hiddenbiases',[nueron_num1],\
                                    initializer=tf.constant_initializer(0))
            weights2 = tf.get_variable(layername+'_outputlayer',[nueron_num1,nueron_num2],\
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            bias2 = tf.get_variable(layername+'_outputbiases',[nueron_num2],\
                                    initializer=tf.constant_initializer(0))
#       
        feature_shape = feature.get_shape().as_list()    
        feature = tf.reshape(feature,(feature_shape[0],feature_shape[1]*\
                                      feature_shape[2]*feature_shape[3]))
        hiddenlayer = tf.nn.sigmoid(tf.matmul(feature,weights1)+bias1)
        output = tf.nn.sigmoid(tf.matmul(hiddenlayer,weights2)+bias2)
#        print(output.get_shape())
        return output
    
    def multiple(self,mlp,bottom):
        mlp_shape = mlp.get_shape().as_list()
        mlp = tf.reshape(mlp,(mlp_shape[0],1,1,1))
        feature = mlp[0:]*bottom[0:]
        return feature
        


     
        
#整体网络结构            
    def build_network(self,picture,feature_size,batchsize,reuse=False):
        
        self.conv1_1 = self.conv_layer(reuse,"conv1_1",bottom=picture)
        self.conv1_2 = self.conv_layer(reuse,"conv1_2",bottom=self.conv1_1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(reuse,"conv2_1",bottom=self.pool1)
        self.conv2_2 = self.conv_layer(reuse,"conv2_2",bottom=self.conv2_1)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(reuse,"conv3_1",bottom=self.pool2)
        self.conv3_2 = self.conv_layer(reuse,"conv3_2",bottom=self.conv3_1)
        self.conv3_3 = self.conv_layer(reuse,"conv3_3",bottom=self.conv3_2)
        self.conv3_4 = self.conv_layer(reuse,"conv3_4",bottom=self.conv3_3)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(reuse,"conv4_1",bottom=self.pool3)
        self.conv4_2 = self.conv_layer(reuse,"conv4_2",bottom=self.conv4_1)
        self.conv4_3 = self.conv_layer(reuse,"conv4_3",bottom=self.conv4_2)
        self.conv4_4 = self.conv_layer(reuse,"conv4_4",bottom=self.conv4_3)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        
        self.conv5_1 = self.conv_layer(reuse,"conv5_1",bottom=self.pool4)
        self.conv5_2 = self.conv_layer(reuse,"conv5_2",bottom=self.conv5_1)
        self.conv5_3 = self.conv_layer(reuse,"conv5_3",bottom=self.conv5_2)
        self.conv5_4 = self.conv_layer(reuse,"conv5_4",bottom=self.conv5_3)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        
        binimg1 = tf.image.resize_bilinear(self.conv1_2,(feature_size,feature_size))
        binimg2 = tf.image.resize_bilinear(self.conv2_2,(feature_size,feature_size))
        binimg3 = tf.image.resize_bilinear(self.conv3_4,(feature_size,feature_size))
        binimg4 = tf.image.resize_bilinear(self.conv4_4,(feature_size,feature_size))
        binimg5 = tf.image.resize_bilinear(self.conv5_4,(feature_size,feature_size))

        self.concat = tf.concat([binimg1,binimg2,binimg3,binimg4,binimg5],axis=3)          
            
       
        print("vgg19Feature is extracted.")


            
            
            
            
        return self.concat
#Feature is extracted.
#conv1(1, 127, 127, 64)
#conv2(1, 63, 63, 128)
#conv3(1, 31, 31, 256)
#conv4(1, 15, 15, 512)
#conv5(1, 7, 7, 512)
#Feature is extracted.
#conv1(1, 255, 255, 64)
#conv2(1, 127, 127, 128)
#conv3(1, 63, 63, 256)
#conv4(1, 31, 31, 512)
#conv5(1, 15, 15, 512)        
#
#
#input
#._create_siamese_train(x_crops= size(24, 255, 255, 3), z_crops=size(8, 127, 127, 3),
# h=[11, 5, 3, 3, 3],w=[11, 5, 3, 3, 3],num=[96, 128, 96, 96, 32])

#output template_z, templates_x = #(8,17,17,32) [24, 49, 49, 32]
    def build_network2(self,picture,feature_size,batchsize,reuse=False):
        
        self.conv1_1 = self.conv_layer(reuse,"conv1_1",bottom=picture)
        self.conv1_2 = self.conv_layer(reuse,"conv1_2",bottom=self.conv1_1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(reuse,"conv2_1",bottom=self.pool1)
        self.conv2_2 = self.conv_layer(reuse,"conv2_2",bottom=self.conv2_1)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(reuse,"conv3_1",bottom=self.pool2)
        self.conv3_2 = self.conv_layer(reuse,"conv3_2",bottom=self.conv3_1)
        self.conv3_3 = self.conv_layer(reuse,"conv3_3",bottom=self.conv3_2)
        self.conv3_4 = self.conv_layer(reuse,"conv3_4",bottom=self.conv3_3)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(reuse,"conv4_1",bottom=self.pool3)
        self.conv4_2 = self.conv_layer(reuse,"conv4_2",bottom=self.conv4_1)
        self.conv4_3 = self.conv_layer(reuse,"conv4_3",bottom=self.conv4_2)
        self.conv4_4 = self.conv_layer(reuse,"conv4_4",bottom=self.conv4_3)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        
        self.conv5_1 = self.conv_layer(reuse,"conv5_1",bottom=self.pool4)
        self.conv5_2 = self.conv_layer(reuse,"conv5_2",bottom=self.conv5_1)
        self.conv5_3 = self.conv_layer(reuse,"conv5_3",bottom=self.conv5_2)
        self.conv5_4 = self.conv_layer(reuse,"conv5_4",bottom=self.conv5_3)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        

        self.feature1 = self.multiple(self.mlp1,self.conv1_2)
        self.feature2 = self.multiple(self.mlp2,self.conv2_2)
        self.feature3 = self.multiple(self.mlp3,self.conv3_4)
        self.feature4 = self.multiple(self.mlp4,self.conv4_4)
        self.feature5 = self.multiple(self.mlp5,self.conv5_4)
#            
#        self.mlp1 = tf.reshape(self.mlp1,[batchsize]) 
#        self.feature1 = self.mlp1*self.conv1_2
#        self.mlp2 = tf.reshape(self.mlp2,[batchsize]) 
#        self.feature2 = self.mlp2*self.conv2_2
#        self.mlp3 = tf.reshape(self.mlp3,[batchsize]) 
#        self.feature3 = self.mlp3*self.conv3_4
#        self.mlp4 = tf.reshape(self.mlp4,[batchsize]) 
#        self.feature4 = self.mlp4*self.conv4_4
#        self.mlp5 = tf.reshape(self.mlp5,[batchsize]) 
#        self.feature5 = self.mlp5*self.conv5_4
##        
#
        binimg1 = tf.image.resize_bilinear(self.feature1,(feature_size,feature_size))
        binimg2 = tf.image.resize_bilinear(self.feature2,(feature_size,feature_size))
        binimg3 = tf.image.resize_bilinear(self.feature3,(feature_size,feature_size))
        binimg4 = tf.image.resize_bilinear(self.feature4,(feature_size,feature_size))
        binimg5 = tf.image.resize_bilinear(self.feature5,(feature_size,feature_size))

        self.concat = tf.concat([binimg1,binimg2,binimg3,binimg4,binimg5],axis=3)
#           
            
       
        print("Feature is extracted.")
        print("conv1"+str(self.feature1.get_shape()))
        print("conv2"+str(self.feature2.get_shape()))
        print("conv3"+str(self.feature3.get_shape()))
        print("conv4"+str(self.feature4.get_shape()))
        print("conv5"+str(self.feature5.get_shape()))

            
            
            
            
        return self.concat      
def create_siamese_train(x_crops, z_crops,x_featuresize,z_featuresize,batchsize_z,batchsize_x):
    net = Vgg19()
    net_z = net.build_network(x_crops,x_featuresize,batchsize_z,False)
    net_x = net.build_network(z_crops,z_featuresize,batchsize_x,True)
#    print(net_z,net_x)
    return net_z, net_x#输出一个batch所有的图片经过网络的tensor结果

 
#测试网络构建情况        
#path = "./tiger.jpg"
#batch1 = ut.gen_batch(127,path)
#batch2 = ut.gen_batch(127,path)
##net_z, net_x=create_siamese_train(batch1,batch2,17,491,8,24)
#net = Vgg19()
#a = net.mlp(batch1,'hh',False,3,9,1)
#b = net.build_network(batch1,127,1)
###feature = a[0][0]*batch1[0]
###feature_shape = feature.get_shape().as_list()
###c = tf.reshape(feature,(1,feature_shape[0],feature_shape[1],feature_shape[2]))
###print(c.get_shape())
##net.multiple(a,batch1)