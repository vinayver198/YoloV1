from tensorflow import keras
import tensorflow as tf
from utils import intersection_over_union


class YoloLoss(tf.keras.losses.Loss):
    """
    Calculate the loss for yolo v1 model
    """

    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss, self).__init__(name="YoloLossV1")
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        """
        S is split size of image(in paper 7)
        B is number of boxes (in paper 2)
        C is number of classes 
        """

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    def call(self,predictions,target):

        """

        :param predictions: tensor (batch_size,S*S*(C+B*5))
        :param target: tensor (batch_size,S*S*(C+B*5))
        :return:
        """

        predictions = predictions.reshape(-1,self.S,self.S,self.C+self.B*5)

        # calculate iou for 1st predicted bbox with target
        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])


        # calculate iou for 2nd predicted bbox with target
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 2:25])
        ious = tf.concat([iou_b1.unsqueeze(),iou_b2.unsqueeze()],axis=0)

        
