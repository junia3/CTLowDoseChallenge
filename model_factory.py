from models.unet import UNet
from models.redcnn import RED_CNN, RED_CNN_LITE
from models.srresnet import SRResNet

def model_build(model_name:str):

    model_name = model_name.lower()
    if model_name == 'unet':
        model = UNet(1, 1, 32)

    if model_name == 'redcnn':
        model = RED_CNN()

    if model_name == 'redcnn_lite':
        model = RED_CNN_LITE()

    if model_name == 'srresnet':
        model = SRResNet(in_channels=1, out_channels=1, channels=32, num_rcb=16, upscale_factor=1)

    return model


