# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/18 15:26
@Author  : Johnson
@FileName: diy_tfci.py
"""

import argparse
import io
import os
import sys
import urllib
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc  # pylint:disable=unused-import

# Default URL to fetch metagraphs from.
URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
# Default location to store cached metagraphs.
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    return tf.expand_dims(image, 0)


def write_png(filename, image):
    """Writes a PNG image file."""
    image = tf.squeeze(image, 0)
    if image.dtype.is_floating:
        image = tf.round(image)
    if image.dtype != tf.uint8:
        image = tf.saturate_cast(image, tf.uint8)
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


def load_cached(filename):
    """Downloads and caches files from web storage."""
    pathname = os.path.join(METAGRAPH_CACHE, filename)
    try:
        with tf.io.gfile.GFile(pathname, "rb") as f:
            string = f.read()
    except tf.errors.NotFoundError:
        url = f"{URL_PREFIX}/{filename}"
        request = urllib.request.urlopen(url)
        try:
            string = request.read()
        finally:
            request.close()
        tf.io.gfile.makedirs(os.path.dirname(pathname))
        with tf.io.gfile.GFile(pathname, "wb") as f:
            f.write(string)
    return string


def instantiate_model_signature(model, signature, inputs=None, outputs=None):
    """Imports a trained model and returns one of its signatures as a function."""
    string = load_cached(model + ".metagraph")
    metagraph = tf.compat.v1.MetaGraphDef()
    metagraph.ParseFromString(string)
    wrapped_import = tf.compat.v1.wrap_function(
        lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
    graph = wrapped_import.graph
    if inputs is None:
        inputs = metagraph.signature_def[signature].inputs
        inputs = [graph.as_graph_element(inputs[k].name) for k in sorted(inputs)]
    else:
        inputs = [graph.as_graph_element(t) for t in inputs]
    if outputs is None:
        outputs = metagraph.signature_def[signature].outputs
        outputs = [graph.as_graph_element(outputs[k].name) for k in sorted(outputs)]
    else:
        outputs = [graph.as_graph_element(t) for t in outputs]
    return wrapped_import.prune(inputs, outputs)


def compress_image(model, input_image):
    """Compresses an image tensor into a bitstring."""
    sender = instantiate_model_signature(model, "sender")
    tensors = sender(input_image)
    packed = tfc.PackedTensors()
    packed.model = model
    packed.pack(tensors)
    return packed.string


def compress(model, input_file, output_file, target_bpp=None, bpp_strict=False):
    """Compresses a PNG file to a TFCI file."""
    if not output_file:
        output_file = input_file + ".tfci"

    # Load image.
    input_image = read_png(input_file)
    num_pixels = input_image.shape[-2] * input_image.shape[-3]

    if not target_bpp:
        # Just compress with a specific model.
        bitstring = compress_image(model, input_image)
    else:
        # Get model list.
        models = load_cached(model + ".models")
        models = models.decode("ascii").split()

        # Do a binary search over all RD points.
        lower = -1
        upper = len(models)
        bpp = None
        best_bitstring = None
        best_bpp = None
        while bpp != target_bpp and upper - lower > 1:
            i = (upper + lower) // 2
            bitstring = compress_image(models[i], input_image)
            bpp = 8 * len(bitstring) / num_pixels
            is_admissible = bpp <= target_bpp or not bpp_strict
            is_better = (best_bpp is None or
                         abs(bpp - target_bpp) < abs(best_bpp - target_bpp))
            if is_admissible and is_better:
                best_bitstring = bitstring
                best_bpp = bpp
            if bpp < target_bpp:
                lower = i
            if bpp > target_bpp:
                upper = i
        if best_bpp is None:
            assert bpp_strict
            raise RuntimeError(
                "Could not compress image to less than {} bpp.".format(target_bpp))
        bitstring = best_bitstring

    # Write bitstring to disk.
    with tf.io.gfile.GFile(output_file, "wb") as f:
        f.write(bitstring)


def main_compress():
    import os
    dst_root = '/home/dingchaofan/data/cc_compression/tfci/'
    os.mkdir(dst_root)

    source_root = '/home/dingchaofan/data/cc_compression/source/'
    sources = os.listdir('/home/dingchaofan/data/cc_compression/source/')
    for src in sources:
        compress(model='hific-lo', input_file=source_root + src, output_file=dst_root + src + ".tfci")
        print('save to ', dst_root + src + '.tfci')


def decompress(input_file, output_file):
    """Decompresses a TFCI file and writes a PNG file."""
    if not output_file:
        output_file = input_file + "11.png"
    with tf.io.gfile.GFile(input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    receiver = instantiate_model_signature(packed.model, "receiver")
    tensors = packed.unpack([t.dtype for t in receiver.inputs])
    output_image, = receiver(*tensors)
    write_png(output_file, output_image)

from tqdm import tqdm


def main_decompress():
    import os
    dst_root = '/home/dingchaofan/data/cc_compression/outputs/'
    os.mkdir(dst_root)

    source_root = '/home/dingchaofan/data/cc_compression/tfci/'
    sources = os.listdir(source_root)
    for src in tqdm(sources):
        decompress(input_file = source_root + src, output_file= dst_root + src + '.png')
        print('save to ', dst_root + src + '.png')

main_decompress()