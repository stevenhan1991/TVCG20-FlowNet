# FlowNet: A Deep Learning Framework for Clustering and Selection of Streamlines and Stream Surfaces
Pytorch implementation for FlowNet: A Deep Learning Framework for Clustering and Selection of Streamlines and Stream Surfaces

In our code, we provide an example of learning hidden features of streamlines and stream surfaces traced from five critical points vector field. The dimension of this data set is 51 by 51 by 51.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.6
- Numpy
- Pytorch >= 0.4.0

## Data preparation
n binary volume files are required for the model training (n is the number of traced streamlines/stream surfaces). The binary volume is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.


## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train'
```

- inference
```
python3 main.py --mode 'inf'
```

## Citation 
```

@article{Han-TVCG20,
	Author = {J. Han and J. Tao and C. Wang},
	Journal = {IEEE Transactions on Visualization and Computer Graphics},
	Number = {4},
	Pages = {1732-1744},
	Title = {{FlowNet}: A Deep Learning Framework for Clustering and Selection of Streamlines and Stream Surfaces},
	Volume = {26},
	Year = {2020}}

```
