import cv2
import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet, BasicBlock


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=1)

        self.fc =  nn.Sequential(
                nn.Linear(512 * BasicBlock.expansion,206),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(206,1),
                nn.Sigmoid()
            )

def make_model():

    model = ImageClassifier()
    model.load_state_dict(torch.load("autism_cnn.pth"))
    print(model)

    return model

# torch-model-archiver --model-name autisum_resnet18 --version 1.0 --model-file model/autisum_model.py --serialized-file model/autism_cnn.pth --handler Image_classifier
#torchserve --start --model-store deployment/model-store --models autisumcnn=autisum_resnet18.mar -ncs