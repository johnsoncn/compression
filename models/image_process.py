# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/18 10:57
@Author  : Johnson
@FileName: image_process.py
"""

from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

def copy_img():
    src_dir = '/home/dingchaofan/data/cc_compression/train/'
    dst_dir = '/home/dingchaofan/data/cc_compression/source/'

    os.mkdir(dst_dir)
    sources = os.listdir(src_dir)
    # print(sources)


    for src in sources:
        img = Image.open('/home/dingchaofan/data/cc_compression/train/' + src)
        width, height = img.size
        # print(width, height)
        if width > 1000 and height > 1000:
            print(width, height)
            shutil.copy(src_dir + src, dst_dir + src)

def image_names():
    dst_dir = '/home/dingchaofan/data/cc_compression/source/'
    sources = os.listdir(dst_dir)
    for i, src in enumerate(sources):
        os.rename(src=dst_dir+src, dst=dst_dir+str(i)+'.png')
        print(i)
    print(os.listdir(dst_dir))


def test():
    import glob
    res = glob.glob('/home/dingchaofan/data/cc_compression/source/')
    print(res)

test()

