import cv2
import numpy
import itertools
import tensorflow
import scipy.ndimage
from pathlib import Path
from multiprocessing import Pool
from tensorflow.keras.applications import resnet_v2

def computeFeatureImage(imagePath):
    # Get the path of the output feature image.
    featureImageName = imagePath.with_suffix(".npy").name
    featureImagePath = Path(imagePath.parents[1], "featureImages", featureImageName)

    # If the feature image was already computed, skip computations.
    if featureImagePath.exists(): return

    # Compute the luminance of the image in YUV space.
    image = cv2.imread(str(imagePath), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    luminance = image[:, :, 0] / 255

    # Compute the gradient of the luminance and the gradient of its Laplacian.
    gradientX = scipy.ndimage.sobel(luminance, axis = 1)
    gradientY = scipy.ndimage.sobel(luminance, axis = 0)
    laplacian = scipy.ndimage.laplace(luminance)
    laplacianGradientX = scipy.ndimage.sobel(laplacian, axis = 1)
    laplacianGradientY = scipy.ndimage.sobel(laplacian, axis = 0)

    # Normalize the gradient of the luminance.
    gradientLength = numpy.sqrt(gradientX * gradientX + gradientY * gradientY)
    gradientX = numpy.divide(gradientX, gradientLength,
            out = numpy.zeros_like(gradientX), where = gradientLength != 0)
    gradientY = numpy.divide(gradientY, gradientLength,
            out = numpy.zeros_like(gradientY), where = gradientLength != 0)

    # Compute the absolute value of the directional gradient of the Laplacian along
    # the normalized gradient of the luminance.
    directionalGradient = abs(
        laplacianGradientX * gradientY - laplacianGradientY * gradientX
    )

    # Compute local binary pattern.
    localBinaryPattern = numpy.zeros_like(luminance)
    localBinaryPattern += (numpy.roll(luminance, (1,   0), axis = (0, 1)) >= luminance) * (2 ** 0)
    localBinaryPattern += (numpy.roll(luminance, (1,   1), axis = (0, 1)) >= luminance) * (2 ** 1)
    localBinaryPattern += (numpy.roll(luminance, (0,   1), axis = (0, 1)) >= luminance) * (2 ** 2)
    localBinaryPattern += (numpy.roll(luminance, (-1,  1), axis = (0, 1)) >= luminance) * (2 ** 3)
    localBinaryPattern += (numpy.roll(luminance, (-1,  0), axis = (0, 1)) >= luminance) * (2 ** 4)
    localBinaryPattern += (numpy.roll(luminance, (-1, -1), axis = (0, 1)) >= luminance) * (2 ** 5)
    localBinaryPattern += (numpy.roll(luminance, (0,  -1), axis = (0, 1)) >= luminance) * (2 ** 6)
    localBinaryPattern += (numpy.roll(luminance, (1,  -1), axis = (0, 1)) >= luminance) * (2 ** 7)
    localBinaryPattern = localBinaryPattern / 255

    # Compute the absolute value of the normalized angle that the gradient makes
    # with the x axis. The range of the values will be in [0, 1].
    gradientAngleFactor = abs(numpy.arctan2(gradientY, gradientX)) / numpy.pi

    # Compose a color image with luminance, directional gradient, and local binary pattern as channels.
    channels = (localBinaryPattern, directionalGradient, luminance)
    featureImage = numpy.stack(channels, axis = -1)

    # Scale the features to be appropriate for ImageNet models.
    preprocessedFeatureImage = resnet_v2.preprocess_input(featureImage * 255)

    # Resize the features to be appropriate for ImageNet models.
    resizedFeatureImage = tensorflow.image.resize(preprocessedFeatureImage, (224, 224))

    numpy.save(featureImagePath, resizedFeatureImage.numpy())

with Pool() as pool:
    imagesDirectory = Path("images")
    imagePaths = imagesDirectory.glob("*")
    inpaintedImagesDirectory = Path("inpaintedImages")
    inpaintedImagePaths = inpaintedImagesDirectory.glob("*")
    allImagePaths = itertools.chain(inpaintedImagePaths, imagePaths)
    pool.map(computeFeatureImage, allImagePaths)
