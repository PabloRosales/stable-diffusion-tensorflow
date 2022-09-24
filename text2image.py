import os
import tensorflow as tf
from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import Text2Image
from stable_diffusion_tf.helpers import get_valid_filename
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render",
)

parser.add_argument(
    "--samples",
    type=int,
    default=4,
    help="the number of samples to render",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="the batch size to render",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=1,
    help="the temperature to render",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixels",
)

parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixels",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--steps", type=int, default=50, help="number of ddim sampling steps"
)

parser.add_argument(
    "--seed",
    type=int,
    help="optionally specify a seed integer for reproducible results",
)

parser.add_argument(
    "--mp",
    default=False,
    action="store_true",
    help="Enable mixed precision (fp16 computation)",
)

parser.add_argument(
    "--jit",
    default=False,
    action="store_true",
    help="Enable XLA JIT compilation",
)

args = parser.parse_args()

if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) <= 0:
    raise Exception("No GPU found.")

seed = args.seed
timestamp = tf.timestamp()
image_filename = get_valid_filename(f"s{seed}_{timestamp}" if seed else f"{timestamp}")
filename = f"t{args.temperature}_sc{args.scale}_st{args.steps}_{image_filename}.png"
prompt_output_directory = f"output/{args.prompt.replace(' ', '_')}"

os.makedirs('output', exist_ok=True)
os.makedirs(prompt_output_directory, exist_ok=True)

generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=args.jit)

for i in range(args.samples):
    print(f"Rendering sample {i+1} of {args.samples}...")
    img = generator.generate(
        args.prompt,
        num_steps=args.steps,
        unconditional_guidance_scale=args.scale,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=seed,
    )
    if args.batch_size == 1:
        image = Image.fromarray(img[0])
        image.save(prompt_output_directory + f"/{i+1}_{filename}")
    else:
        for j in range(args.batch_size):
            image = Image.fromarray(img[j])
            image.save(prompt_output_directory + f"/{i+1}_{j+1}_{filename}")
