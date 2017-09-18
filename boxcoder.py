import numpy as np
import tensorflow as tf

def create_prior_boxes(model, layer_sizes = [38,19,10,5,3,1], variances = [0.1,0.1,0.2,0.2]):
    """
    Arguments:
        image_shape: The image shape (width, height) to the
        input model.
        model_configurations: The model configurations created by
        load_model_configurations that indicate the parameters
        inside the PriorBox layers.

    Returns:
        prior_boxes: A numpy array containing all prior boxes
    """
    boxes = {}
    for layer in model.layers:
        if layer.name.endswith('priorbox'):
            boxes[layer.name] = layer
    prior_boxes = []
    for box,layer_size in zip(boxes,layer_sizes):
        prior_boxes.append(boxes[box].get_prior_boxes(layerWidth = layer_size, layerHeight = layer_size))

    prior_boxes = np.concatenate(prior_boxes, axis = 0)
    variances = np.ones((len(prior_boxes), 4)) * variances
    prior_boxes = np.concatenate((prior_boxes, variances), axis = 1)

    return prior_boxes

class BoxCoder(object):
    def __init__(self, numClasses, priorBoxes = None, iouThreshold = 0.5,
                nmsThreshold = 0.4, topK = 400):
        self.numClasses = numClasses
        self.priorBoxes = priorBoxes
        self.numPriorBoxes = 0 if priorBoxes is None else len(priorBoxes)
        self.iouThreshold = iouThreshold
        self.topK = topK
        self.boxes = tf.placeholder(dtype='float32', shape=(None,4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nmsThreshold = nmsThreshold
        self.sess = tf.Session()
        
    def calIou(self, annotationBox):
        interLeftTop = np.maximum(self.priorBoxes[:,:2], annotationBox[:2])
        interRightDown = np.minimum(self.priorBoxes[:,2:4], annotationBox[2:])
        interSides = interRightDown - interLeftTop
        interSides = np.maximum(interSides, 0)
        interArea = interSides[:, 0] * interSides[:, 1]
        
        annotationBoxArea = (annotationBox[2] - annotationBox[0]) * (annotationBox[3] - annotationBox[1])
        priorBoxesArea = (self.priorBoxes[:, 2] - self.priorBoxes[:, 0]) * (self.priorBoxes[:, 3] - self.priorBoxes[:, 1])
        unionArea = annotationBoxArea + priorBoxesArea - interArea
        return interArea / unionArea
    
    def encoder(self, annotationBox, returnIou = True):
        iou = self.calIou(annotationBox)
        encodedBox = np.zeros((self.numPriorBoxes, 4 + returnIou))
        assignMask = iou > self.iouThreshold
        if not assignMask.any():
            assignMask[iou.argmax()] = True
        if returnIou:
            encodedBox[:, -1][assignMask] = iou[assignMask]
            
        assignedPriorBoxes = self.priorBoxes[assignMask]
        annotationBoxCenter = 0.5 * (annotationBox[:2] + annotationBox[2:])
        annotationBoxSides = annotationBox[2:] - annotationBox[:2]
        assignedPriorBoxesCenter = 0.5 * (assignedPriorBoxes[:, :2] + assignedPriorBoxes[:, 2:4])
        assignedPriorBoxesSides = assignedPriorBoxes[:, 2:4] - assignedPriorBoxes[:, :2]
        
        encodedBox[:, :2][assignMask] = annotationBoxCenter - assignedPriorBoxesCenter
        encodedBox[:, :2][assignMask] /= assignedPriorBoxesSides
        encodedBox[:, :2][assignMask] /= assignedPriorBoxes[:, -4:-2]
        encodedBox[:, 2:4][assignMask] = np.log(annotationBoxSides / assignedPriorBoxesSides)
        encodedBox[:, 2:4][assignMask] /= assignedPriorBoxes[:, -2:]
        
        return encodedBox.ravel()
    
    def assignBoxes(self, annotationBox):
        assignments = np.zeros((self.numPriorBoxes, 4 + self.numClasses + 8))
        assignments[:, 4] = 1.0
        if len(annotationBox) == 0:
            return assignments
        encodedBoxes = np.apply_along_axis(self.encoder, 1, annotationBox[:, :4])
        encodedBoxes = encodedBoxes.reshape(-1, self.numPriorBoxes, 5)
        bestIou = encodedBoxes[:, :, -1].argmax(axis = 0)
        bestIouIdx = encodedBoxes[:, :, -1].argmax(axis = 0)
        bestIouMask = bestIou > 0
        bestIouIdx = bestIouIdx[bestIouMask]
        numAssignment = len(bestIouIdx)
        encodedBoxes = encodedBoxes[:, bestIouMask, :]
        assignments[:, :4][bestIouMask] = encodedBoxes[bestIouIdx, np.arange(numAssignment), :4]
        assignments[:, 4][bestIouMask] = 0
        assignments[:, 5:-8][bestIouMask] = annotationBox[bestIouIdx, 4:]
        assignments[:, -8][bestIouMask] = 1
        return assignments
    
    def decoder(self, mboxLoc, mboxPriorBox, variances):
        priorBoxWidth = mboxPriorBox[:, 2] - mboxPriorBox[:, 0]
        priorBoxHeight = mboxPriorBox[:, 3] - mboxPriorBox[:, 1]
        priorBoxCenterX = 0.5 * (mboxPriorBox[:, 2] + mboxPriorBox[:, 0])
        priorBoxCenterY = 0.5 * (mboxPriorBox[:, 3] + mboxPriorBox[:, 1])
        decodeBoxCenterX = mboxLoc[:, 0] * priorBoxWidth * variances[:, 0]
        decodeBoxCenterX += priorBoxCenterX
        decodeBoxCenterY = mboxLoc[:, 1] * priorBoxHeight * variances[:, 1]
        decodeBoxCenterY += priorBoxCenterY
        decodeBoxWidth = np.exp(mboxLoc[:, 2] * variances[:, 2])
        decodeBoxWidth *= priorBoxWidth
        decodeBoxHeight = np.exp(mboxLoc[:, 3] * variances[:, 3])
        decodeBoxHeight *= priorBoxHeight
        decodeBoxTopX = decodeBoxCenterX - 0.5 * decodeBoxWidth
        decodeBoxTopY = decodeBoxCenterY - 0.5 * decodeBoxHeight
        decodeBoxDownX = decodeBoxCenterX + 0.5 * decodeBoxWidth
        decodeBoxDownY = decodeBoxCenterY + 0.5 * decodeBoxHeight
        decodeBox = np.concatenate((decodeBoxTopX[:, None],
                                   decodeBoxTopY[:, None],
                                   decodeBoxDownX[:, None],
                                   decodeBoxDownY[:, None]), axis = -1)
        decodeBox = np.minimum(np.maximum(decodeBox, 0.0), 1.0)
        return decodeBox
    
    def detectBoxes(self, predictions, backgroundID = 0, topK = 200, confidenceThrehold = 0.5):
        mboxLoc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mboxPriorBox = predictions[:, :, -8:-4]
        mboxConf = predictions[:, :, 4:-8]
        selectedBoxes = []
        for i in range(len(mboxLoc)):
            selectedBoxes.append([])
            decodeBox = self.decoder(mboxLoc[i], mboxPriorBox[i], variances[i])
            for cla in range(self.numClasses):
                if cla == backgroundID:
                    continue
                
                claConf = mboxConf[i, :, cla]
                claConfSelected = claConf > confidenceThrehold
                if len(claConf[claConfSelected] > 0):
                    boxesToProcess = decodeBox[claConfSelected]
                    confsToProcess = claConf[claConfSelected]
                    feedDict = {self.boxes: boxesToProcess,
                               self.scores: confsToProcess}
                    nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                      self.topK, iou_threshold = self.nmsThreshold)
                    idx = self.sess.run(nms, feed_dict = feedDict)
                    boxes = boxesToProcess[idx]
                    confs = confsToProcess[idx][:, None]
                    labels = cla * np.ones((len(idx), 1))
                    claPredict = np.concatenate((labels, confs, boxes), axis = 1)
                    selectedBoxes[-1].extend(claPredict)
                    
                if len(selectedBoxes[-1]) > 0:
                    selectedBoxes[-1] = np.array(selectedBoxes[-1])
                    argsort = np.argsort(selectedBoxes[-1][:, 1])[::-1]
                    selectedBoxes[-1] = selectedBoxes[-1][argsort]
                    selectedBoxes[-1] = selectedBoxes[-1][:topK]
                    
        return selectedBoxes
                    
