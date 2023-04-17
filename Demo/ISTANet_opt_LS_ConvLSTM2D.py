from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend
from keras import losses
from keras.constraints import nonneg, unit_norm
from keras.initializers import Constant, RandomNormal, RandomUniform
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
        
class ISTANet_opt_LS():
    def __init__(self):
        # Input shape
        self.ratio = 20
        self.channels = 1
        self.lr_height = 500                 # Low resolution height
        self.lr_width = 500                  # Low resolution width
        self.lr_shape = (self.lr_height, self.channels)
        self.hr_height = self.lr_height   # High resolution height
        self.hr_width = self.lr_width     # High resolution width
        self.hr_shape = (self.lr_height, self.lr_width, self.channels)
        self.lambda_step = 0.01
        self.soft_thr = 0.01

        #optimizer = Adam(lr=5e-4)
        optimizer = adam_v2.Adam(lr=5e-4)

        # Build the generator
        self.encoder = self.build_encoder()
        '''
        def PSNR(y_true, y_pred):
            dif = backend.squeeze(y_true-y_pred,axis=0)
            max_pixel = backend.max(dif)
            return 0.0001 - 1 / (10.0 * backend.log((max_pixel ** 2) / (backend.mean(backend.square(y_pred - y_true)))))
        
        def regularized_loss(y_true, y_pred):
            return PSNR(y_true, y_pred) + 0.01*backend.mean(backend.square(y_pred[:,:,:,1:289]))
        '''
        def regularized_loss(y_true, y_pred):
            return backend.mean(backend.square(y_pred[:,:,:,0] - backend.squeeze(y_true,axis=3))) + 0.01*backend.mean(backend.square(y_pred[:,:,:,1:289])) + 0.01*backend.mean(backend.abs(y_pred[:,:,:,0]))
        
        def rms(y_true, y_pred):
            return backend.sqrt(backend.mean(backend.square(y_pred[:,:,:,0] - backend.squeeze(y_true,axis=3))))
        
        self.encoder.compile(loss=regularized_loss, optimizer=optimizer, metrics=[rms])

    def build_encoder(self):
        print('************************************************ISTANet_opt_pulse************************************************')
        # Low resolution image input
        inputs = Input(shape=self.hr_shape)
        #print(inputs.shape,'inputs1')
        I = Reshape((1,500,500,1))(inputs)
        #print(inputs.shape,'inputs2')
        
        kernel_length = 10
        #self.scan = ConvLSTM2D(1, (1,kernel_length), strides=(1,kernel_length), activation = 'tanh', recurrent_activation='hard_sigmoid', padding = 'same', data_format='channels_last', kernel_initializer = Constant(value = 1/kernel_length), kernel_constraint=nonneg(),return_sequences=True)
        self.scan = Conv2D(1, (1,kernel_length), strides=(1,kernel_length), activation = None, padding = 'same', data_format='channels_last', kernel_initializer = Constant(value = 1/kernel_length), kernel_constraint=nonneg())
        #measurement1 = self.scan(I)
        measurement1 = self.scan(I)
        measurement1 = GaussianNoise(0.4/11)(measurement1)
        
        #tf.config.run_functions_eagerly(False)
        #measurement1 = backend.eval(measurement1)
        #measurement1 = tf.convert_to_tensor(measurement1, dtype=tf.float32)
        #with tf.compat.v1.Session().as_default():
            #measurement1 = measurement1.numpy()
        #measurement1 = np.random.poisson(measurement1)
        #measurement1 = backend.variable(measurement1)
        
        measurement1 = backend.squeeze(measurement1,axis=1)
        
        print('**********measurement1',measurement1)
        
        #self.recur = Bidirectional(LSTM(64, return_sequences=True),merge_mode='sum')
        #self.recur = LSTM(64, return_sequences=True, return_state = True)
        #measurement1 = backend.eval(measurement1)
        #measurement2,_,_ = self.recur(measurement1)
        #print('**********measurement2',measurement2)
        
        self.convt2d1 = Conv2D(1, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        self.convt2d2 = Conv2D(1, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        self.convt2d3 = Conv2D(1, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        self.convt2d4 = Conv2D(1, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        #measurement2 = Reshape((256,kernel_length,1))(measurement1)
        decon = self.convt2d1(UpSampling2D(size = (1,2))(measurement1))
        decon = self.convt2d2(UpSampling2D(size = (1,5))(decon))
        #decon = self.convt2d3(UpSampling2D(size = (1,2))(decon))
        #decon = self.convt2d4(UpSampling2D(size = (1,2))(decon)) #(None, 256, 256, 1)
        decon = Reshape((1,500,500,1))(decon)
        print('**********x0',decon)
        #print(decon.shape,'decon')

        #r1,_,_ = self.recur(backend.squeeze(self.scan(decon),axis=3))
        #r1 = Reshape((256,64,1))(r1)
        r1 = self.scan(decon)
        r1 = backend.squeeze(r1,axis=1)
        #print(r1.shape,'r1')
        r1 = self.convt2d1(UpSampling2D(size = (1,2))(r1))
        r1 = self.convt2d2(UpSampling2D(size = (1,5))(r1))
        #r1 = self.convt2d3(UpSampling2D(size = (1,2))(r1))
        #r1 = self.convt2d4(UpSampling2D(size = (1,2))(r1))
        temp = self.lambda_step
        decon = backend.squeeze(decon,axis=1)
        R1 = subtract([decon, Lambda(lambda x: temp*x)(subtract([r1,decon]))])
        conv1_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R1)
        f11 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv1_2 = f11(conv1_1)
        #print(conv1_2.shape,'conv1_2')
        conv1_symm = f11(conv1_1)
        #print(conv1_symm.shape,'conv1_symm')
        f12 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv1_3 = f12(conv1_2)
        conv1_symm = f12(conv1_symm)
        temp = self.soft_thr
        conv1_4 = multiply([Lambda(lambda x: backend.sign(x))(conv1_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv1_3)))])
        f13 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv1_5 = f13(conv1_4)
        conv1_symm = f13(conv1_symm)
        f14 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv1_6 = f14(conv1_5)       
        conv1_symm = f14(conv1_symm)       
        conv1_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv1_6)                
        conv1_7 = add([conv1_7, R1])
        conv1_8 = subtract([conv1_symm, conv1_1])

        conv1_7 = Reshape((1,500,500,1))(conv1_7)
        r2 = self.scan(conv1_7)
        r2 = backend.squeeze(r2,axis=1)
        r2 = self.convt2d1(UpSampling2D(size = (1,2))(r2))
        r2 = self.convt2d2(UpSampling2D(size = (1,5))(r2))
        #r2 = self.convt2d3(UpSampling2D(size = (1,2))(r2))
        #r2 = self.convt2d4(UpSampling2D(size = (1,2))(r2))
        temp = self.lambda_step
        conv1_7 = backend.squeeze(conv1_7,axis=1)
        R2 = subtract([conv1_7, Lambda(lambda x: temp*x)(subtract([r2,decon]))])
        conv2_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R2)
        f21 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv2_2 = f21(conv2_1)
        conv2_symm = f21(conv2_1)
        f22 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv2_3 = f22(conv2_2)
        conv2_symm = f22(conv2_symm)
        temp = self.soft_thr
        conv2_4 = multiply([Lambda(lambda x: backend.sign(x))(conv2_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv2_3)))])
        f23 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv2_5 = f23(conv2_4)
        conv2_symm = f23(conv2_symm)
        f24 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv2_6 = f24(conv2_5)       
        conv2_symm = f24(conv2_symm)       
        conv2_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv2_6)                
        conv2_7 = add([conv2_7, R2])
        conv2_8 = subtract([conv2_symm, conv2_1])
        
        conv2_7 = Reshape((1,500,500,1))(conv2_7)
        r3 = self.scan(conv2_7)
        r3 = backend.squeeze(r3,axis=1)
        r3 = self.convt2d1(UpSampling2D(size = (1,2))(r3))
        r3 = self.convt2d2(UpSampling2D(size = (1,5))(r3))
        #r3 = self.convt2d3(UpSampling2D(size = (1,2))(r3))
        #r3 = self.convt2d4(UpSampling2D(size = (1,2))(r3))
        temp = self.lambda_step
        conv2_7 = backend.squeeze(conv2_7,axis=1)
        R3 = subtract([conv2_7, Lambda(lambda x: temp*x)(subtract([r3,decon]))])
        conv3_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R3)
        f31 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv3_2 = f31(conv3_1)
        conv3_symm = f31(conv3_1)
        f32 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv3_3 = f32(conv3_2)
        conv3_symm = f32(conv3_symm)
        temp = self.soft_thr
        conv3_4 = multiply([Lambda(lambda x: backend.sign(x))(conv3_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv3_3)))])
        f33 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv3_5 = f33(conv3_4)
        conv3_symm = f33(conv3_symm)
        f34 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv3_6 = f34(conv3_5)       
        conv3_symm = f34(conv3_symm)       
        conv3_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv3_6)                
        conv3_7 = add([conv3_7, R3])
        conv3_8 = subtract([conv3_symm, conv3_1])

        conv3_7 = Reshape((1,500,500,1))(conv3_7)
        r4 = self.scan(conv3_7)
        r4 = backend.squeeze(r4,axis=1)
        r4 = self.convt2d1(UpSampling2D(size = (1,2))(r4))
        r4 = self.convt2d2(UpSampling2D(size = (1,5))(r4))
        #r4 = self.convt2d3(UpSampling2D(size = (1,2))(r4))
        #r4 = self.convt2d4(UpSampling2D(size = (1,2))(r4))
        temp = self.lambda_step
        conv3_7 = backend.squeeze(conv3_7,axis=1)
        R4 = subtract([conv3_7,Lambda(lambda x: temp*x)(subtract([r4,decon]))])
        conv4_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R4)
        f41 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv4_2 = f41(conv4_1)
        conv4_symm = f41(conv4_1)
        f42 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv4_3 = f42(conv4_2)
        conv4_symm = f42(conv4_symm)
        temp = self.soft_thr
        conv4_4 = multiply([Lambda(lambda x: backend.sign(x))(conv4_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv4_3)))])
        f43 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv4_5 = f43(conv4_4)
        conv4_symm = f43(conv4_symm)
        f44 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv4_6 = f44(conv4_5)       
        conv4_symm = f44(conv4_symm)       
        conv4_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv4_6)                
        conv4_7 = add([conv4_7, R4])
        conv4_8 = subtract([conv4_symm, conv4_1])

        conv4_7 = Reshape((1,500,500,1))(conv4_7)
        r5 = self.scan(conv4_7)
        r5 = backend.squeeze(r5,axis=1)
        r5 = self.convt2d1(UpSampling2D(size = (1,2))(r5))
        r5 = self.convt2d2(UpSampling2D(size = (1,5))(r5))
        #r5 = self.convt2d3(UpSampling2D(size = (1,2))(r5))
        #r5 = self.convt2d4(UpSampling2D(size = (1,2))(r5))        
        temp = self.lambda_step
        conv4_7 = backend.squeeze(conv4_7,axis=1)
        R5 = subtract([conv4_7, Lambda(lambda x: temp*x)(subtract([r5,decon]))])
        conv5_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R5)
        f51 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv5_2 = f51(conv5_1)
        conv5_symm = f51(conv5_1)
        f52 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv5_3 = f52(conv5_2)
        conv5_symm = f52(conv5_symm)
        temp = self.soft_thr
        conv5_4 = multiply([Lambda(lambda x: backend.sign(x))(conv5_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv5_3)))])
        f53 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv5_5 = f53(conv5_4)
        conv5_symm = f53(conv5_symm)
        f54 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv5_6 = f54(conv5_5)       
        conv5_symm = f54(conv5_symm)       
        conv5_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv5_6)                
        conv5_7 = add([conv5_7, R5])
        conv5_8 = subtract([conv5_symm, conv5_1])
        
        conv5_7 = Reshape((1,500,500,1))(conv5_7)
        r6 = self.scan(conv5_7)
        r6 = backend.squeeze(r6,axis=1)
        r6 = self.convt2d1(UpSampling2D(size = (1,2))(r6))
        r6 = self.convt2d2(UpSampling2D(size = (1,5))(r6))
        #r6 = self.convt2d3(UpSampling2D(size = (1,2))(r6))
        #r6 = self.convt2d4(UpSampling2D(size = (1,2))(r6))
        temp = self.lambda_step
        conv5_7 = backend.squeeze(conv5_7,axis=1)
        R6 = subtract([conv5_7, Lambda(lambda x: temp*x)(subtract([r6,decon]))])
        conv6_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R6)
        f61 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv6_2 = f61(conv6_1)
        conv6_symm = f61(conv6_1)
        f62 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv6_3 = f62(conv6_2)
        conv6_symm = f62(conv6_symm)
        temp = self.soft_thr
        conv6_4 = multiply([Lambda(lambda x: backend.sign(x))(conv6_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv6_3)))])
        f63 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv6_5 = f63(conv6_4)
        conv6_symm = f63(conv6_symm)
        f64 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv6_6 = f64(conv6_5)       
        conv6_symm = f64(conv6_symm)       
        conv6_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv6_6)                
        conv6_7 = add([conv6_7, R6])
        conv6_8 = subtract([conv6_symm, conv6_1])

        conv6_7 = Reshape((1,500,500,1))(conv6_7)
        r7 = self.scan(conv6_7)
        r7 = backend.squeeze(r7,axis=1)
        r7 = self.convt2d1(UpSampling2D(size = (1,2))(r7))
        r7 = self.convt2d2(UpSampling2D(size = (1,5))(r7))
        #r7 = self.convt2d3(UpSampling2D(size = (1,2))(r7))
        #r7 = self.convt2d4(UpSampling2D(size = (1,2))(r7))
        temp = self.lambda_step
        conv6_7 = backend.squeeze(conv6_7,axis=1)
        R7 = subtract([conv6_7, Lambda(lambda x: temp*x)(subtract([r7,decon]))])
        conv7_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R7)
        f71 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv7_2 = f71(conv7_1)
        conv7_symm = f71(conv7_1)
        f72 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv7_3 = f72(conv7_2)
        conv7_symm = f72(conv7_symm)
        temp = self.soft_thr
        conv7_4 = multiply([Lambda(lambda x: backend.sign(x))(conv7_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv7_3)))])
        f73 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv7_5 = f73(conv7_4)
        conv7_symm = f73(conv7_symm)
        f74 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv7_6 = f74(conv7_5)       
        conv7_symm = f74(conv7_symm)       
        conv7_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv7_6)                
        conv7_7 = add([conv7_7, R7])
        conv7_8 = subtract([conv7_symm, conv7_1])
        
        conv7_7 = Reshape((1,500,500,1))(conv7_7)
        r8 = self.scan(conv7_7)
        r8 = backend.squeeze(r8,axis=1)
        r8 = self.convt2d1(UpSampling2D(size = (1,2))(r8))
        r8 = self.convt2d2(UpSampling2D(size = (1,5))(r8))
        #r8 = self.convt2d3(UpSampling2D(size = (1,2))(r8))
        #r8 = self.convt2d4(UpSampling2D(size = (1,2))(r8))
        temp = self.lambda_step
        conv7_7 = backend.squeeze(conv7_7,axis=1)
        R8 = subtract([conv7_7, Lambda(lambda x: temp*x)(subtract([r8,decon]))])
        conv8_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R8)
        f81 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv8_2 = f81(conv8_1)
        conv8_symm = f81(conv8_1)
        f82 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv8_3 = f82(conv8_2)
        conv8_symm = f82(conv8_symm)
        temp = self.soft_thr
        conv8_4 = multiply([Lambda(lambda x: backend.sign(x))(conv8_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv8_3)))])
        f83 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv8_5 = f83(conv8_4)
        conv8_symm = f83(conv8_symm)
        f84 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv8_6 = f84(conv8_5)       
        conv8_symm = f84(conv8_symm)       
        conv8_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv8_6)                
        conv8_7 = add([conv8_7, R8])
        conv8_8 = subtract([conv8_symm, conv8_1])

        conv8_7 = Reshape((1,500,500,1))(conv8_7)
        r9 = self.scan(conv8_7)
        r9 = backend.squeeze(r9,axis=1)
        r9 = self.convt2d1(UpSampling2D(size = (1,2))(r9))
        r9 = self.convt2d2(UpSampling2D(size = (1,5))(r9))
        #r9 = self.convt2d3(UpSampling2D(size = (1,2))(r9))
        #r9 = self.convt2d4(UpSampling2D(size = (1,2))(r9))
        temp = self.lambda_step
        conv8_7 = backend.squeeze(conv8_7,axis=1)
        R9 = subtract([conv8_7, Lambda(lambda x: temp*x)(subtract([r9,decon]))])
        conv9_1 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(R9)
        f91 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv9_2 = f91(conv9_1)
        conv9_symm = f91(conv9_1)
        f92 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv9_3 = f92(conv9_2)
        conv9_symm = f92(conv9_symm)
        temp = self.soft_thr
        conv9_4 = multiply([Lambda(lambda x: backend.sign(x))(conv9_3), ReLU()(Lambda(lambda x: x-temp)(Lambda(lambda x: backend.abs(x))(conv9_3)))])
        f93 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv9_5 = f93(conv9_4)
        conv9_symm = f93(conv9_symm)
        f94 = Conv2D(32, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')
        conv9_6 = f94(conv9_5)       
        conv9_symm = f94(conv9_symm)       
        conv9_7 = Conv2D(1, 3, activation = None, padding = 'same', data_format='channels_last', kernel_initializer = 'he_normal')(conv9_6)                
        conv9_7 = add([conv9_7, R9])
        conv9_8 = subtract([conv9_symm, conv9_1])
        #print(conv9_7.shape,'conv9_7')
        #print(conv1_8.shape,'conv1_8')
        #print(conv9_8.shape,'conv9_8')
        outputs = Concatenate(axis = 3)([conv9_7,conv1_8,conv2_8,conv3_8,conv4_8,conv5_8,conv6_8,conv7_8,conv8_8,conv9_8])   

        print('*****Shape of the output layer*****',outputs)
        
        return Model(inputs = inputs, outputs = outputs)

    def train(self, imgs_hr1, imgs_lr1, epochs, batch_size=4):
        #wb = Workbook()
        #sheet = wb.add_sheet('loss', cell_overwrite_ok=True)
        #model = load_model('DCSRN.h5')
        #imgs_lr2 = model.predict(imgs_lr1, batch_size=2, verbose=1)

        #self.encoder.fit(imgs_lr1[0:400], imgs_hr1[0:400], epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(imgs_lr1[400:440], imgs_hr1[400:440]))
        self.encoder.fit(imgs_lr1[0:80], imgs_hr1[0:80], epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(imgs_lr1[80:100], imgs_hr1[80:100]))
        self.encoder.save('test.h5')
