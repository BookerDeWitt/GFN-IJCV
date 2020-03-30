# GFN-IJCV

**"Gated Fusion Network for Degraded Image Super Resolution"** by [Xinyi Zhang*](http://xinyizhang.tech), [Hang Dong*](https://sites.google.com/view/hdong/首页), [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)**(Accpeptd by IJCV, first two authors contributed equally)**.

[[arXiv](https://arxiv.org/abs/2003.00893)]

You can find our pervious conference version on [Project Website](http://xinyizhang.tech/bmvc2018/).

![Archi](https://drive.google.com/open?id=1yGVa_VmiKOjLn6OjKScAAevY-P1wbCkm)
![Gate](https://drive.google.com/open?id=1ZtvYX81PAyR2DyIq1dWKmYhzBWQIKyxN)

## Dependencies
* Python 3.6
* PyTorch >= 0.4.0
* torchvision
* numpy
* skimage
* h5py
* MATLAB

## Super-resolving non-uniform blurry images

1. Git clone this repository.
```bash
$git clone https://github.com/BookerDeWitt/GFN-IJCV
$cd GFN-IJCV/DBSR
```

2. Download the trained model ``GFN_G3D_4x.pkl`` from [here](https://drive.google.com/open?id=18A-xhZ0Gk5mJPKLUoXBLh6rLSwBPHUrv), then unzip and move the ``GFN_G3D_4x.pkl`` to ``GFN-IJCV/DBSR/models`` folder.

3. Then, you can follow the instructions [here](https://github.com/jacquelinelala/GFN) to test and train our network with our latest code and pre-trained model.

## Super-resolving non-uniform hazy images

## Super-resolving rainy images
### How to test:
**Test on LR-Rain1200 Validation**
This model is the result of the third step with 37 epoch.
1. Git clone this repository.
```bash
$git clone https://github.com/BookerDeWitt/GFN-IJCV
$cd GFN-IJCV/DRSR
```
2. Download the LR-Rain1200 dataset (including both the test and training sets) from [Google Drive]() or [BaiduYun](https://pan.baidu.com/s/1Z0tKjE_iDi4dpXuJFIMKDQ) (Code:v7e1).
3. Download the trained model ``GFN_epoch_37.pkl`` from [Google Drive](https://drive.google.com/open?id=1R4Ng5lAOfHyywNVXC3BB7mzf2ufvkHHF) or [BaiduYun](https://pan.baidu.com/s/1QHBOrT7eMnXfLtdU189udw) (Code:koeu), then unzip and move the ``GFN_epoch_37.pkl`` to ``GFN/models`` folder.
4. Run the ``GFN-IJCV/DRSR/test.py`` with cuda on command line: 
```bash
GFN-IJCV/DRSR/$python test.py --dataset your_downloads_directory/LR_Rain1200/Validation_4x
```
Then the deraining and super-solving images ending with GFN_4x.png are in the directory of LR_Rain1200/Validation_4x/Results.

5. Calculate the PSNR using Matlab function ``GFN-IJCV/DRSR/evaluation/test_RGB.m``. The output of the average PSNR is 25.248834 dB. You can also use the ``GFN-IJCV/DRSR/evaluation/test_bicubic.m`` to calculate the bicubic method.  
```bash
>> folder = 'your_downloads_directory/LR_Rain1200';
>> test_RGB(folder)
```

### How to train
**Train on LR-Rain1200 dataset**
You should accomplish the first two steps in **Test on LR-Rain1200 Validation** before the following steps.
#### Train from scratch
1. Generate the train hdf5 files of LR_Rain1200 dataset: Run the matlab function ``rain_hdf5_generator.m`` which is in the directory of GFN/h5_generator. The generated hdf5 files are stored in the your_downloads_directory/LR_Rain1200/Rain_HDF5.
```bash
>> folder = 'your_downloads_directory/LR_Rain1200';
>> rain_hdf5_generator(folder)
```
2. Run the ``GFN/train.py`` with cuda on command line:
```bash
GFN/$python train.py --dataset your_downloads_directory/LR_Rain1200/Rain_HDF5
```
3. The three step intermediate models will be respectively saved in models/1/ models/2 and models/3. You can also use the following command to test the intermediate results during the training process.
Run the ``GFN/test.py`` with cuda on command line: 
```bash
GFN/$python test.py --dataset your_downloads_directory/LR_Rain1200/Validation_4x --intermediate_process models/1/GFN_epoch_25.pkl # We give an example of step1 epoch25. You can replace another pkl file in models/.
```
#### Resume training from breakpoints
Since the training process will take 3 or 4 days, you can use the following command to resume the training process from any breakpoints.
Run the ``GFN/train.py`` with cuda on command line:
```bash
GFN/$python train.py --dataset your_downloads_directory/LR_Rain1200/Rain_HDF5 --resume models/1/GFN_epoch_25.pkl # Just an example of step1 epoch25.
```

## Citation

If you use these models in your research, please cite:

	@article{GFN_IJCV,
		author = {Xinyi, Zhang and Hang, Dong and Zhe, Hu and Wei-Sheng, Lai and Fei, Wang and Ming-Hsuan, Yang},
		title = {Gated Fusion Network for Degraded Image Super Resolution},
		journal={International Journal of Computer Vision},
		year = {2020},
    		pages={1 - 23}
	}

	@inproceedings{GFN_BMVC,
    		title = {Gated Fusion Network for Joint Image Deblurring and Super-Resolution},
		author = {Xinyi, Zhang and Hang, Dong and Zhe, Hu and Wei-Sheng, Lai and Fei, Wang and Ming-Hsuan, Yang},
		booktitle = {BMVC},
		year = {2018}
	}

