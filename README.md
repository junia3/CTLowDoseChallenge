![header](https://capsule-render.vercel.app/api?type=waving&color=timeGradient:F39F86&height=250&section=header&text=CT%20Imaging%20Project&fontSize=45&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
<!-- 
<p align="center"><a href="#">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:F9D976,100:F39F86&height=250&section=header&text="CT Imaging" &fontSize=40&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40" alt="header" />
</a></p>
 -->

## Dataset
We use 2016-Low Dose CT Grand Challenge [dataset](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/)
Place them in the dataset directory like below. I created .npy file for each patient CT image pair in order to reduce RAM usage on training process.
```bash
├── dataset
│   ├── train
│   ├── validation
│   ├── test
│   └── create_npy.py
``` 
And run script below
```bash
python create_npy.py
```
Then, it automatically create .npy files on the same directory like below
```bash
├── dataset
│   ├── train_dataset
│   │   ├── 0000.npy
│   │   ├── 0001.npy
│   │   ├── 0002.npy
│   │   ├── ...
│   ├── validation_dataset
│   ├── test_dataset
│   └── create_npy.py
``` 
After run 'create_npy.py', you can just remove redundant directory files.

## Model
I am currently on project, so there are only baseline models.
### 1. U-Net
The most basic model in pixel-to pixel prediction task such as segmentation, deblurring, image enhancement, ... etc.
I referenced on [pytorch official code](https://github.com/mateuszbuda/brain-segmentation-pytorch).

### 2. [REDCNN](https://arxiv.org/pdf/1702.00288.pdf)

RED-CNN is a good reference model in CT image enhancement. However the model was too heavy to train(high spatial dimension with 96 channels for each layer). Therefore I applied additional trick on this model, which is named "REDCNN-Lite"

### 3. REDCNN-Lite
All convolutional layers and transposed convolutional layers are replaced with depthwise-separable convolution and deconvolution layers.
The separable convolution layers implemented with pytorch is,
```python
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
```
And in the same way, separable deconvolution layers implemented with pytorch is,
```python
class depthwise_separable_trconv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_trconv, self).__init__()
        self.depthwise = nn.ConvTranspose2d(nin, nin * kernels_per_layer, kernel_size=5, padding=0, groups=nin)
        self.pointwise = nn.ConvTranspose2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
```
which is same with mobileNet implementation code. Kernel size and the number of kernel is slightly changed.

### SRResNet
SRResNet is super-resolution specified residual network.

![footer](https://capsule-render.vercel.app/api?type=waving&color=timeGradient:F39F86&height=150&section=footer&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
