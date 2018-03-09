---
title: "Data Augmentation"
date: 2018-03-08T22:06:06+08:00
slug: augmentation
---

After reanding this post, you will know:

- Common data augmentation methods
- Implementation with `ImageDataGenerator` in keras
- Popular tools for data augmentation

## Methods

In this section, I will introduce these augmentation methods:

1. cropping
1. rotating
1. flipping

### cropping

### rotating

### flipping



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

