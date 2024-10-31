import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Encoder: ResNet pre-addestrato
        resnet = models.resnet50(weights = ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # esclude l'ultimo layer fully connected

        # Decoder: Convoluzioni trasposte per l'upsampling
        decoder = list()
        decoder +=[
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ]
        self.decoder = nn.Sequential(*decoder)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)  # Estrae le feature dell'immagine con ResNet
        x = self.decoder(x)  # Decodifica le feature nella depth map
        x = self.output(x) # Output finale ad un solo canale
        x = nn.functional.interpolate(x, size=(144, 256), mode='bilinear', align_corners=False)  # upsampling necessario per la dimensione dell'immagine
        return x
