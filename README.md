# Frechet-Inception-Distance
A new Tensorflow implementation of the "Fréchet Inception Distance" (FID) between two image distributions, along with a numpy interface. The FID can be used to evaluate generative models by calculating the FID between real and fake data distributions (lower is better).
## Prequisites
- `numpy` and `tensorflow-gpu`

## Features
- Fast and memory-efficient
- No prior knowledge about Tensorflow is necessary to use this code
- Makes use of [TFGAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan)
- Downloads InceptionV1 automatically
- Compability with both Python2 and Python3

## Usage
- Call `get_fid(images1, images2)`, where images1, images2 are numpy arrays with values ranging from 0 to 255 and shape in the form `[N, 3, HEIGHT, WIDTH]` where `N`, `HEIGHT` and `WIDTH` can be arbitrary. `dtype` of the images is recommended to be `np.uint8` to save CPU memory.
- A smaller `BATCH_SIZE` reduces GPU memory usage, but at the cost of a slight slowdown.

## Links

- The Fréchet Inception Distance was proposed in the paper [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium ](https://arxiv.org/abs/1706.08500)
- Code for the [Inception Score](https://github.com/tsc2017/Inception-Score)
