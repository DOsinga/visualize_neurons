#!/usr/bin/env python
from __future__ import print_function

import argparse
import random
from itertools import islice

import imageio
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from scipy import ndimage
import numpy as np
from tqdm import tqdm

SCALE_STEPS = 30
LAYER_NAME = 'block5_conv1'

model = None
layer_dict = None


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def visstd(a, s=0.1):
    '''Normalize and clip the image range for visualization'''
    a = (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
    return np.clip(a, 0, 1) * 255


def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return ndimage.zoom(img, factors, order=1)


def generate_movie(steps, movie_name, *, target_size):
    if not steps:
        steps = random.sample(list(range(512)), 10)
    layer = layer_dict[LAYER_NAME]
    input_img = model.input

    sizes = []
    next_size = target_size
    for i in range(SCALE_STEPS):
        sizes.insert(0, int(next_size))
        next_size *= 0.9

    initial_size, *sizes = sizes
    img_data = np.random.uniform(size=(1, initial_size, initial_size, 3)) + 128.
    loss = K.mean(layer.output[:, :, :, steps[0]])

    print('about to start.')
    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img], [loss, grads])

    for octave, size in enumerate(sizes):
        img_data = resize_img(img_data, (size, size))
        loss_value, grads_value = iterate([img_data])
        img_data += grads_value
    print('initialized.')

    scaled = int(target_size * 1.05)
    with imageio.get_writer(movie_name, mode='I', fps=25) as writer:
        pairs = list(zip(steps[:-1], steps[1:]))
        for prev_step, next_step in tqdm(pairs):
            for i in range(SCALE_STEPS * 3):
                weight = max(1.0, i / SCALE_STEPS)
                loss = (K.mean(layer.output[:, :, :, next_step]) * weight +
                        K.mean(layer.output[:, :, :, prev_step]) * (1 - weight))
                grads = K.gradients(loss, input_img)[0]
                grads = normalize(grads)
                iterate = K.function([input_img], [loss, grads])
                img_data = resize_img(img_data, (scaled, scaled))
                start = (scaled - target_size) // 2
                # center crop
                img_data = img_data[:, start: start + target_size, start: start + target_size, :]
                # add a little noise
                img_data += np.random.uniform(size=(1, target_size, target_size, 3)) * 16
                for i in range(7):
                    loss_value, grads_value = iterate([img_data])
                    if i == 0:
                        grads_value[0, :, :, 0] = ndimage.gaussian_filter(grads_value[0, :, :, 0], sigma=2)
                        grads_value[0, :, :, 1] = ndimage.gaussian_filter(grads_value[0, :, :, 1], sigma=2)
                        grads_value[0, :, :, 2] = ndimage.gaussian_filter(grads_value[0, :, :, 2], sigma=2)
                    img_data += grads_value
                img_data[0] = visstd(img_data[0])
                writer.append_data(np.uint8(img_data[0]))
            prev_step = next_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_size', type=int, default=128)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--steps', type=str, default='4, 21, 25, 34, 38, 39, 44, 49, 50, 64, 4',
                        help='Neurons to generate a movie from')
    args = parser.parse_args()

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    print(model.summary())
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    steps =[int(step) for step in args.steps.split(',')]
    generate_movie(steps, args.output, target_size=args.target_size)