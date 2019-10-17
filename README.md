# Fréchet Inception Distance
A new Tensorflow implementation of the "Fréchet Inception Distance" (FID) between two image distributions, along with a numpy interface. The FID can be used to evaluate generative models by calculating the FID between real and fake data distributions (lower is better).

## Major Dependency
- `tensorflow >= 1.14`

## Features
- Fast, easy-to-use and memory-efficient
- No prior knowledge about Tensorflow is necessary if your are using CPU or GPU
- Makes use of [TF-GAN](https://github.com/tensorflow/gan)
- Downloads InceptionV1 automatically
- Compatible with both Python 2 and Python 3

## Usage
- If you are working with GPU, use `fid.py`; if you are working with TPU, use `fid_tpu.py` and pass a Tensorflow Session and a [TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy) as additional arguments.
- Call `get_fid(images1, images2)`, where `images1`, `images2` are numpy arrays with values ranging from 0 to 255 and shape in the form `[N, 3, HEIGHT, WIDTH]` where `N`, `HEIGHT` and `WIDTH` can be arbitrary. `dtype` of the images is recommended to be `np.uint8` to save CPU memory.
- A smaller `BATCH_SIZE` reduces GPU/TPU memory usage, but at the cost of a slight slowdown.
- If you want to compute a general "Fréchet Classifier Distance" with activations (e.g., outputs of the last pooling layer) `act1` and `act2` from another classifier, call `activations2distance(act1, act2)`. `act1` and `act2` can be numpy arrays of a same arbitrary shape `[N, d]`.
## Links

- The Fréchet Inception Distance was proposed in the paper [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium ](https://arxiv.org/abs/1706.08500)
- Code for the [Inception Score](https://github.com/tsc2017/Inception-Score)
