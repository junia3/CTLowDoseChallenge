# CTLowDoseChallenge
Deblurring/Enhancing CT image
---
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
