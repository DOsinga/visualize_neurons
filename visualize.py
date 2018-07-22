#!/usr/bin/env python
from __future__ import print_function

import argparse
import random
from itertools import islice

import imageio
from keras.applications import vgg16
from keras import backend as K
from scipy import ndimage
import numpy as np
from tqdm import tqdm

SCALE_STEPS = 30
model = None
layer_dict = None


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def visstd(a, s=0.1):
    '''Normalize and clip the image range for visualization'''
    a = (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(a, 0, 1) * 255)


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return ndimage.zoom(img, factors, order=1)


def generate_image(layer_name, neuron_idx, *, target_size, resize=True):
    layer = layer_dict[layer_name]
    input_img = model.input

    sizes = []
    for i in range(SCALE_STEPS):
        sizes.insert(0, int(target_size))
        if resize:
            target_size *= 0.9

    initial_size, *sizes = sizes
    img_data = np.random.uniform(size=(1, initial_size, initial_size, 3)) + 128.
    loss = K.mean(layer.output[:, :, :, neuron_idx])

    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img], [loss, grads])

    loss_value = None
    for octave, size in enumerate(sizes):
        if resize:
            img_data = resize_img(img_data, (size, size))
        loss_value, grads_value = iterate([img_data])
        img_data += grads_value

    return visstd(img_data[0]), loss_value


def generate_grid(target_size):
    cols = 10
    sample_size = 100
    grid = []
    layers = [layer_dict['block%d_conv%d' % (i, (i + 2) // 3)] for i in range(1, 6)]
    layers = layers
    for layer_idx, layer in enumerate(layers):
        row = []
        neurons = list(range(max(x or 0 for x in layer.output_shape)))
        if len(neurons) > sample_size:
            neurons = random.sample(neurons, sample_size)
        for neuron in tqdm(neurons, desc=layer.name):
            img_data, loss_value = generate_image(layer.name, neuron, target_size=target_size, resize=False)
            row.append((loss_value, img_data))
        grid.append([cell[1] for cell in islice(sorted(row, key=lambda t: -t[0]), 10)])

    target_size_4 = target_size + 4
    res = np.full(shape=(len(layers) * (target_size_4) + 4, cols * (target_size_4) + 4, 3), fill_value=128)
    for y in range(len(layers)):
        for x in range(cols):
            res[4 + y * (target_size_4): 4 + y * (target_size_4) + target_size,
                4 + x * (target_size_4): 4 + x * (target_size_4) + target_size,
                :] = grid[y][x]

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, default='block5_conv1')
    parser.add_argument('--neuron_idx', type=int, default=53)
    parser.add_argument('--target_size', type=int, default=128)
    parser.add_argument('--task', type=str, choices=['image', 'grid', 'movie'])
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    if args.task == 'grid':
        res = generate_grid(args.target_size)
    else:
        print(args.layer)
        print(args.neuron_idx)
        res, _ = generate_image(args.layer, args.neuron_idx, target_size=args.target_size)

    imageio.imwrite(args.output, res)
