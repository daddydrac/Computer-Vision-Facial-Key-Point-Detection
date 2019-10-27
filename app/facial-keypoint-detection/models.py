## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        # torch.nn.Conv2d(
        # in_channels, 
        # out_channels, 
        # kernel_size, 
        # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        #############################################
        #                                           #
        #  A minimalistic version of NaimishNet:   #
        #  Facial Key Points Detection using Deep   #
        #  Convolutional Neural Network             #
        #                                           #    
        #  Source:                                  #
        #  https://arxiv.org/pdf/1710.00977.pdf     #
        #                                           #
        #############################################
        
        
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=1, 
                out_channels=32, 
                kernel_size=2, 
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2, 
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2, 
                stride=1,
            ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
                       
            # Layer 4
             nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=2, 
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )
        
        # Formula:     (W − F + 2*P ) / S + 1
        
        # Layer 1:
        # Conv2d:      (224 - 2 + 2*0) / 1 + 1 = 223
        # MaxPool2d:   (223 - 2 + 2*0) / 2 + 1 = 222
        
        # Layer 2:
        # Conv2d:      (222 - 2 + 2*0) / 1 + 1 = 221
        # MaxPool2d:   (221 - 2 + 2*0) / 2 + 1 = 220
        
        # Layer 3:
        # Conv2d:      (220 - 2 + 2*0) / 1 + 1 = 219
        # MaxPool2d:   (219 - 2 + 2*0) / 2 + 1 = 218
        
        # Layer 4:
        # Conv2d:      (218 - 2 + 2*0) / 1 + 1 = 217
        # MaxPool2d:   (217 - 2 + 2*0) / 2 + 1 = 216
        
        # Formula:     (W − F + 2P ) / S + 1
        
        self.fc = nn.Sequential(
            
             # FC Layer 1:
             nn.Linear(43264, 432),
             nn.ReLU(),
             nn.Dropout(0.5),
            
             # FC Layer 2:
             nn.Linear(432, 216),
             nn.ReLU(),
             nn.Dropout(0.5),
            
             # FC Layer 3:
             nn.Linear(216, 108), # output of 136 as suggested by instructions from Udacity
             nn.ReLU(),
             nn.Dropout(0.6)
         )

        #I.xavier_uniform(self.fc.weight.data)

        

        
    def forward(self, x):
        
        
        def flatten(t):
            t = t.reshape(1, t.size()[0] * x)
            t = t.squeeze()
            return t
        
             
        x = self.conv(x)
        x = flatten(x)
        x = self.fc(x)
          
        # a modified x, having gone through all the layers of your model, should be returned
        return x
