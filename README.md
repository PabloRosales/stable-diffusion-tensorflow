# Stable Diffusion in TensorFlow / Keras

**Note**: This is a fork as a playground.

A Keras / Tensorflow implementation of Stable Diffusion. 

The weights were ported from the original implementation.

## Colab Notebooks

The easiest way to try it out is to use one of the Colab notebooks:


- [GPU Colab](https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK)
- [GPU Colab + Mixed Precision](https://colab.research.google.com/drive/15mQgITh3e9HQMNys0zR8JN4R2vp06d-N)
  - ~10s generation time per image (512x512) on default Colab GPU without drop in quality
    ([source](https://twitter.com/fchollet/status/1571954014845308928))
- [TPU Colab](https://colab.research.google.com/drive/17zQOm_2Iu6pcP8otT-v6rx0D-pKgfaLm).
  - Slower than GPU for single-image generation, faster for large batch of 8+ images
    ([source](https://twitter.com/fchollet/status/1572004717362028546)).
- [GPU Colab with Gradio](https://colab.research.google.com/drive/1ANTUur1MF9DKNd5-BTWhbWa7xUBfCWyI)



## Installation

### Install as a python package

Install using pip with the git repo:

```bash
pip install git+https://github.com/divamgupta/stable-diffusion-tensorflow
```

### Installing using the repo

Download the repo, either by downloading the
[zip](https://github.com/divamgupta/stable-diffusion-tensorflow/archive/refs/heads/master.zip)
file or by cloning the repo with git:

```bash
git clone git@github.com:divamgupta/stable-diffusion-tensorflow.git
```

#### Using pip without a virtual environment

Install dependencies using the `requirements.txt` file or the `requirements_m1.txt` file,:

```bash
pip install -r requirements.txt
```

#### Using a virtual environment with *virtualenv*

1) Create your virtual environment for `python3`:

    ```bash
    python3 -m venv venv
    ```
   
2) Activate your virtualenv:

    ```bash
    source venv/bin/activate
    ```

3) Install dependencies using the `requirements.txt` file or the `requirements_m1.txt` file,:

    ```bash
    pip install -r requirements.txt
    ```

#### Using a virtual environment with *conda*

Follow the steps for installing tensorflow with miniconda: [install tensorflow with pip](https://www.tensorflow.org/install/pip) 
or use an existing conda environment.

Then with your conda environment activated, install the dependencies using the `requirements.txt` file or the `requirements_m1.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Using the Python interface

If you installed the package, you can use it as follows:

```python
from stable_diffusion_tf.stable_diffusion import Text2Image
from PIL import Image

generator = Text2Image(
    img_height=512,
    img_width=512,
    jit_compile=False,
)
img = generator.generate(
    "An astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
)
Image.fromarray(img[0]).save("output.png")
```

### Using `text2image.py` from the git repo

Assuming you have installed the required packages, 
you can generate images from a text prompt using:

```bash
python text2image.py --prompt="An astronaut riding a horse"
```

The generated image(s) will be stored on the `output` directory 
in the following format `output/Your-Prompt/[# sample]-[# batch]-s[seed]-[timestamp].png`.

Check out the `text2image.py` file for more options, including image size, number of steps, etc.

## Troubleshooting

### Low end GPUs

If you have a lower-end GPU, using `--jit --mp` might help.

## Example outputs 

The following outputs have been generated using this implementation:

1) *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](https://user-images.githubusercontent.com/1890549/190841598-3d0b9bd1-d679-4c8d-bd5e-b1e24397b5c8.png)


2) *Spider-Gwen Gwen-Stacy Skyscraper Pink White Pink-White Spiderman Photo-realistic 4K*

![a](https://user-images.githubusercontent.com/1890549/190841999-689c9c38-ece4-46a0-ad85-f459ec64c5b8.png)


3) *A vision of paradise, Unreal Engine*

![a](https://user-images.githubusercontent.com/1890549/190841886-239406ea-72cb-4570-8f4c-fcd074a7ad7f.png)


## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
