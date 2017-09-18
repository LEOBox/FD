import keras.backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf

class Normalize(Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)
        
    def build(self, inputShape):
        self.inputSpec = [InputSpec(shape=inputShape)]
        shape = (inputShape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name = '{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        
    def call(self, inputTensor, mask = None):
        output = K.l2_normalize(inputTensor, self.axis)
        output *= self.gamma
        return output

class PriorBox(Layer):
    def __init__(self, imgSize, minSize, maxSize = None, aspectRations = [1.0], 
                 flip = True, variances = [0.1], clip = True, **kwargs):
        self.wAxis = 2
        self.hAxis = 1
        self.flip = flip
        self.imgSize = imgSize
        if minSize <= 0:
            raise Exception("Illegal mini size")
        self.minSize = minSize
        self.maxSize = maxSize
        self.aspectRations = []
        if self.maxSize:
            self.aspectRations.append(1.0)
        for ar in aspectRations:
            self.aspectRations.append(ar)
            if self.flip and ar != 1.0:
                self.aspectRations.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)
        
    def compute_output_shape(self, inputShape):
        numAspectRations = len(self.aspectRations)
        layerWidth = inputShape[self.wAxis]
        layerHeight = inputShape[self.hAxis]
        numBoxes = numAspectRations * layerWidth * layerHeight
        return inputShape[0], numBoxes, 8
    
    def call(self, inputTensor, mask = None):
        if hasattr(inputTensor, '_keras_shape'):
            inputShape = inputTensor._keras_shape
        elif hasattr(K, 'int_shape'):
            inputShape = K.int_shape(inputTensor)
            
        layerWidth = inputShape[self.wAxis]
        layerHeight = inputShape[self.hAxis]
        imgWidth = self.imgSize[0]
        imgHeight = self.imgSize[1]
        
        boxWidth = []
        boxHeight = []
        for ar in self.aspectRations:
            if ar == 1 and len(boxWidth) == 0:
                boxWidth.append(self.minSize)
                boxHeight.append(self.minSize)
            elif ar == 1 and len(boxWidth) >0:
                boxWidth.append(np.sqrt(self.minSize * self.maxSize))
                boxHeight.append(np.sqrt(self.minSize * self.maxSize))
            else:
                boxWidth.append(self.minSize * np.sqrt(ar))
                boxHeight.append(self.minSize / np.sqrt(ar))
                
        boxWidth = 0.5 * np.array(boxWidth)
        boxHeight = 0.5 * np.array(boxHeight)
        
        stepX = imgWidth / layerWidth
        stepY = imgHeight / layerHeight
        linX = np.linspace(0.5 * stepX, imgWidth - 0.5 * stepX, layerWidth)
        linY = np.linspace(0.5 * stepY, imgHeight - 0.5 * stepY, layerHeight)
        centerX,centerY = np.meshgrid(linX,linY)
        centerX = centerX.reshape(-1, 1)
        centerY = centerY.reshape(-1, 1)
        
        numAspectRations = len(self. aspectRations)
        priorBoxes = np.concatenate((centerX, centerY), axis = 1)
        priorBoxes = np.tile(priorBoxes, (1, 2 * numAspectRations))
        priorBoxes[:, ::4] -= boxWidth
        priorBoxes[:, 1::4] -= boxHeight
        priorBoxes[:, 2::4] += boxWidth
        priorBoxes[:, 3::4] += boxHeight
        priorBoxes[:, ::2] /= imgWidth
        priorBoxes[:, 1::2] /= imgHeight
        priorBoxes = priorBoxes.reshape(-1, 4)
        
        if self.clip:
            priorBoxes = np.minimum(np.maximum(priorBoxes, 0.0), 1.0)
        numBoxes = len(priorBoxes)
        if len(self.variances) == 1:
            variances = np.ones((numBoxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (numBoxes, 1))
        else:
            raise Exception("Length of variances must be 1 or 4")
            
        priorBoxes = np.concatenate((priorBoxes, variances), axis = 1)
        priorBoxesTensor = K.expand_dims(K.variable(priorBoxes), 0)
        pattern = [tf.shape(inputTensor)[0], 1, 1]
        priorBoxesTensor = tf.tile(priorBoxesTensor, pattern)
        
        return priorBoxesTensor
            
    def get_prior_boxes(self,imgSize = (300,300),layerWidth = None,layerHeight = None):
        imgWidth = imgSize[0]
        imgHeight = imgSize[1]
        boxWidth = []
        boxHeight = []
        for ar in self.aspectRations:
            if ar == 1 and len(boxWidth) == 0:
                boxWidth.append(self.minSize)
                boxHeight.append(self.minSize)
            elif ar == 1 and len(boxWidth) >0:
                boxWidth.append(np.sqrt(self.minSize * self.maxSize))
                boxHeight.append(np.sqrt(self.minSize * self.maxSize))
            else:
                boxWidth.append(self.minSize * np.sqrt(ar))
                boxHeight.append(self.minSize / np.sqrt(ar))
                
        boxWidth = 0.5 * np.array(boxWidth)
        boxHeight = 0.5 * np.array(boxHeight)
        
        stepX = imgWidth / layerWidth
        stepY = imgHeight / layerHeight
        linX = np.linspace(0.5 * stepX, imgWidth - 0.5 * stepX, layerWidth)
        linY = np.linspace(0.5 * stepY, imgHeight - 0.5 * stepY, layerHeight)
        centerX,centerY = np.meshgrid(linX,linY)
        centerX = centerX.reshape(-1, 1)
        centerY = centerY.reshape(-1, 1)
        
        numAspectRations = len(self. aspectRations)
        priorBoxes = np.concatenate((centerX, centerY), axis = 1)
        priorBoxes = np.tile(priorBoxes, (1, 2 * numAspectRations))
        priorBoxes[:, ::4] -= boxWidth
        priorBoxes[:, 1::4] -= boxHeight
        priorBoxes[:, 2::4] += boxWidth
        priorBoxes[:, 3::4] += boxHeight
        priorBoxes[:, ::2] /= imgWidth
        priorBoxes[:, 1::2] /= imgHeight
        priorBoxes = priorBoxes.reshape(-1, 4)
        
        if self.clip:
            priorBoxes = np.minimum(np.maximum(priorBoxes, 0.0), 1.0)
        numBoxes = len(priorBoxes)
        if len(self.variances) == 1:
            variances = np.ones((numBoxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (numBoxes, 1))
        else:
            raise Exception("Length of variances must be 1 or 4")
        return priorBoxes
