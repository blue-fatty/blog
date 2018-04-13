---
title: "Data Augmentation"
date: 2018-03-08T22:06:06+08:00
slug: augmentation
post: true
group: deep
weight: 2
---

Data augmentation is the process of increasing the size of a dataset by transforming it
in ways that a neural network is unlikely to learn by itself.

This article will introduce:

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
from sklearn.decomposition import PCA

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
# simplify code
def show_img_augs(imgs, imgaug_seq):
    ia.seed(1)
    imgs_aug = seq.augment_images(imgs)
    show9(imgs_aug)

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

Also called as reflection or mirroring.

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

For color jitter, the knowledge of [color space](/note/color/#color-space) may help.


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

Fancy PCA or PCA Color Augmentation is a type of data augmentation technique
first mentioned in Alex's paper *ImageNet Classification with Deep Convolutional Neural Networks*.

The original words in the paper:

>We perform PCA on the set of RGB pixel values throughout the ImageNet training set.
>
**To each training image, we add multiples of the found principal components,
with magnitudes proportional to the corresponding eigenvalues times a random variable
drawn from a Gaussian with mean zero and standard deviation 0.1.**
>
Therefore **to each RGB image pixel \\(I\_{xy} = [I\_{xy}^R,I\_{xy}^G,I\_{xy}^B]^T\\)
we add the following quantity:**
$$
[p\_1,p\_2,p\_3][\alpha\_1 \lambda_1 , \alpha\_2 \lambda_2 , \alpha\_3 \lambda_3]^T
$$
**where \\(p_i\\) and \\(\lambda_i\\) are
\\(i\\)th eigenvector and eigenvalue of the \\(3 \times 3\\) covariance matrix of RGB pixel vales,
respectively, and \\(\alpha_i\\) is the aforementioned random variable.**
>
Each \\(\alpha_i\\) is drawn only once for all the pixels of a particular training image
until that image is used for training again,
at which point it is re-drawn. This scheme aproximately captures an important property of natural images,
namely, that object identity is invariant to changes in the intensity and color of illuminiation.
This scheme reduces the top-1 error rate by over 1%.

So the whole steps of fancy PCA is:

1. Resize data's shape into `(n * width * height, 3)`
1. Standardize data into unit scale ( mean=0, variance=1 )
1. Compute the eigen values and eigen vectors.
    - Decomposition of covariance matrix
    - Decomposition of correlation matrix
    - Use skit-learn's tool
1. Image augmentation

Code for this:

``` py
# 1. Resize
res = np.array([]).reshape([0,3])
for img in imgs:
    img = img / 255.
    arr = img.reshape(img.shape[0]*img.shape[1], 3)
    res = np.vstack([res, arr])

# 2. Standardize
mean = res.mean(axis=0)
std = res.std(axis=0)
res_std = (res - mean) / std

# 3. Eigendecomposition
pca = PCA()
rgb_pca = pca.fit(res_std)
eigen_values = rgb_pca.explained_variance_
eigen_vectors = rgb_pca.components_.T
print 'Eigen Values \n%s' % eigen_values
print 'Eigen Vectors \n%s' % eigen_vectors

# 4. Image augmentation
def data_aug(img, eig_vals, eig_vecs):

    if len(eig_vals.shape) == 1:
        eig_vals = eig_vals[np.newaxis, :]

    mu = 0
    sigma = 0.1

    # 3 x 1 scaled eigenvalue matrix
    w = np.random.normal(mu, sigma, (1,3)) * eig_vals
    noise = eig_vecs.dot(w.T).reshape([1,1,3])

    # perturbe the image
    img_aug = img + noise

    return img_aug

def data_augs(imgs, eig_vals, eig_vecs):

    img_augs = imgs.copy()

    for i in xrange(img_augs.shape[0]):
        img_augs[i] = data_aug(img_augs[i], eig_vals, eig_vecs)

    return img_augs

img_augs = data_augs(imgs/255., eigen_values, eigen_vectors)
show9(img_augs)
```

The entire code in notebook is [here](https://nbviewer.jupyter.org/github/blue-fatty/notebooks/blob/master/scripts/keras-image-augmentation.ipynb)
or you can find it in my [notebooks repo](https://github.com/blue-fatty/notebooks).

See also: [Fancy PCA (Data Augmentation) with Scikit-Image](https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image)

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

### distortion

{{< img "https://i.imgur.com/874Tjkp.png" "images with distortion" >}}

Code for this:

``` py
seq = iaa.Sequential([
    iaa.PiecewiseAffine(scale=(0.01, 0.07))
])
show_img_augs(imgs, seq)
```
---

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

#### Links

- [Github homepage of imgaug](https://github.com/aleju/imgaug)
- [Brief and complex Examples](http://imgaug.readthedocs.io/en/latest/source/examples_basics.html)
- [Quick reference](http://imgaug.readthedocs.io/en/latest/source/augmenters.html)

### ImageDataGenerator in Keras

- [Keras - Image Preprocessing](https://keras.io/preprocessing/image/)

Data from memory:

``` py
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# One way: Fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# Another way: "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

When training on cifar-10 dataset, rotation has little effects.
Width shift, height shift and horizontal flip are used and usually can improve accuracy about 10%.
By using these data augmentation methods, we can avoid overfitting very efficiently.

---

Data from disk:

``` py
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```
