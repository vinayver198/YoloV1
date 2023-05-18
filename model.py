import tensorflow as tf
from tensorflow import keras

architecture_config = [
(7,64,2,3),
"M",
(3,192,1,1),
"M",
(1,128,1,0),
(3,256,1,1),
(1,256,1,0),
(3,512,1,1),
"M",
[(1,256,1,0),(3,512,1,1),4],
(1,512,1,0),
    (3,1024,1,1),

[(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
(3,1024,2,1),
(3,1024,1,1),
(3,1024,1,1)
]

class CNNBlock(keras.layers.Layer):
    def __init__(self,filters,**kwargs):
        super(CNNBlock,self).__init__()
        self.conv = keras.layers.Conv2D(filters,**kwargs)
        self.batchnorm = keras.layers.BatchNormalization()
        self.leakyRelu = keras.layers.LeakyReLU()

    def call(self,inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        return self.leakyRelu(x)

class YoloV1(keras.Model):
    def __init__(self,**kwargs):
        super(YoloV1, self).__init__(name="YoloV1")
        self.architecture = architecture_config

        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def call(self,x):
        x = self.darknet(x)
        return self.fcs(x)

    def _create_conv_layers(self,architecture):
        layers = []
        for x in architecture:
            if type(x) == tuple:
                layers+=[CNNBlock(x[1],kernel_size=x[0],
                                 strides=x[2],padding='SAME')]

            elif type(x) == str:
                layers +=[keras.layers.MaxPool2D((2,2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repititions = x[2]
                for _ in range(repititions):
                    layers+=[CNNBlock(conv1[1],kernel_size=x[0],
                                 stride=x[2],padding='SAME')]
        return keras.Sequential(layers)

    def _create_fcs(self,split_size,num_boxes,num_classes):
        S,B,C = split_size,num_boxes,num_classes
        head = keras.Sequential()
        head.add(keras.layers.Flatten())
        head.add(keras.layers.Dense(4096,activation=keras.layers.LeakyReLU()))
        head.add(keras.layers.Dropout(0.4))
        head.add(keras.layers.Dense(S*S*(C+(B*5)))) # (S,S,30) where C+(B*5) = 30
        return head

num_classes = 10


def test(S=7,B=2,C=20):
    model = YoloV1(split_size=S,num_boxes=B,num_classes=C)
    x = tf.random.normal((2, 448, 448, 3))
    print(model(x).shape)

test()

