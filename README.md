# [CVPR 2025] EGDeblurring

Code of CVPR2025 paper "Diffusion-based Event Generation for High-Quality Image Deblurring".

## Requirement
* Python 3.7
* Pytorch 1.7
```bash
pip install -r requirements.txt
```

## Test

Test the model

```python
python test.py -c config/deblur_test.json
```
We use the DDIM sampling to speed up the inference stage. The number of steps can be set as 5 or 25.

## Train
1. Train the network in 3 stages
```python
python train_stage1.py
```
```
python train_stage2.py -c config/deblur_1.json
```

```
python train_stage3.py -c config/deblur_s1.json
```



## References
Our implementation is based on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) and [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion). We would like to thank them.

Citation
-----
```
@inproceedings{zhu2025entityerasure,
  title={Diffusion-based Event Generation for High-Quality Image Deblurring},
  author={Xie, Xinan and Zhang, Qing and Zheng, Wei-Shi},
  booktitle={CVPR},
  year={2025}
}
```

