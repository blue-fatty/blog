---
title: "Data Augmentation"
date: 2018-03-08T22:06:06+08:00
slug: augmentation
---

After reanding this post, you will know:

- Common data augmentation methods.
- Image augmentation with `imgaug`.
- Popular tools for data augmentation.

<!--more-->

## Methods

In this section, I will introduce these augmentation methods:

1. cropping
1. shifting
1. rotating
1. flipping
1. shearing
1. color jittering
    1. brigheness
    1. contrast
    1. fancy PCA
1. salt and pepper

The original images get from cifar-10 is like this:

{{< img "https://i.imgur.com/2w0WeeM.png" "original images" >}}

Code for this:

``` py
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

from imgaug import augmenters as iaa
import imgaug as ia

%matplotlib inline

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

def show9(imgs):
    ''' Create a grid of 3x3 images
    '''
    if imgs.max() > 1:
        imgs = imgs / 255.

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(imgs[i], cmap=plt.get_cmap())
    plt.show()

imgs = X_train[0:9]
show9(imgs)
```

### cropping

{{< img "https://i.imgur.com/bvkU57j.png" "images with cropping" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Crop(percent = (0, 0.3))
])
show_img_augs(imgs, seq)
```

### shifting

{{< img "https://i.imgur.com/sFLIDvK.png" "images with shifting" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
])
show_img_augs(imgs, seq)
```

### rotating

{{< img "https://i.imgur.com/f8rLLZn.png" "images with rotating" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25))
])
show_img_augs(imgs, seq)
```

### flipping

Also called as reflection.

{{< img "https://i.imgur.com/UoyJQ3D.png" "images with horizontal flipping" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Fliplr(0.5)
])
show_img_augs(imgs, seq)
```

{{< img "https://i.imgur.com/bfH3ttq.png" "images with vertical flipping" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Flipud(0.5)
])
show_img_augs(imgs, seq)
```

### shearing

{{< img "https://i.imgur.com/rZEzakR.png" "images with shearing" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Affine(shear=(-20, 20))
])
show_img_augs(imgs, seq)
```

### color jitter

#### brightness

{{< img "https://i.imgur.com/9gO6BJa.png" "change the brightness of images" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.Multiply((0.1, 1.5))
])
show_img_augs(imgs, seq)
```

#### contrast

{{< img "https://i.imgur.com/Ef9no0K.png" "change the contrast of images" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.ContrastNormalization((0.75, 1.5))
])
show_img_augs(imgs, seq)
```

#### fancy PCA

### salt and pepper

Add salt (white points) and pepper (black points) to images.

{{< img "https://i.imgur.com/HJ2VhEn.png" "add salt and pepper to images" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.SaltAndPepper(p=(0, 0.1))
])
show_img_augs(imgs, seq)
```


You can read [REDME.md of imgaug](https://github.com/aleju/imgaug) for more augmentation methods.

## Tools

### imgaug

>`imgaug` is a library for image augmentation in machine learning experiments. It supports a wide range of augmentation techniques, allows to easily combine these, has a simple yet powerful stochastic interface, can augment images and keypoints/landmarks on these and offers augmentation in background processes for improved performance.
<br/>
>
[Github](https://github.com/aleju/imgaug)

#### A standard use case

The following example shows a standard use case. An augmentation sequence (crop + horizontal flips + gaussian blur) is defined once at the start of the script. Then many batches are loaded and augmented before being used for training.

``` py
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

for batch_idx in range(1000):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = load_batch(batch_idx)
    images_aug = seq.augment_images(images)
    train_on_images(images_aug)
```

### Links

- [Github homepage of imgaug](https://github.com/aleju/imgaug)
- [Brief and complex Examples](http://imgaug.readthedocs.io/en/latest/source/examples_basics.html)
- [Quick reference](http://imgaug.readthedocs.io/en/latest/source/augmenters.html)

