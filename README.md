# GFN-IJCV

**"Gated Fusion Network for Degraded Image Super Resolution"** by [Xinyi Zhang*](http://xinyizhang.tech), [Hang Dong*](https://sites.google.com/view/hdong/首页), [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)**(Accpeptd by IJCV, first two authors contributed equally)**.

[[arXiv](https://arxiv.org/abs/2003.00893)]

You can find our pervious conference version on [Project Website : http://xinyizhang.tech/bmvc2018](http://xinyizhang.tech/bmvc2018/).

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

3. Then, you can follow the instructions [Here](https://github.com/jacquelinelala/GFN) to test and train our network with our latest code and pre-trained model

## Super-resolving non-uniform hazy images

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

