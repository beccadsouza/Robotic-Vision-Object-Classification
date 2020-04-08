import os
import glob
import numpy
import cv2

imagePaths = []
# input images
for img in glob.glob("Data/*.jpg"):  # folder train1 contains multiple dog and cat images in .jpg
    imagePaths = list(glob.glob("Data/*.jpg"))

# Extract the image into vector
def image_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


# initialize the pixel intensities matrix, labels list
imagematrix = []
imagelabels = []
pixels = None
# Build image vector matrix
for (i, path) in enumerate(imagePaths):
    # load the image and extract the class label, image intensities
    image = cv2.imread(path)
    label = path.split(os.path.sep)[-1].split(".")[0]
    pixels = image_vector(image)

    # update the images and labels matricies respectively
    imagematrix.append(pixels)
    imagelabels.append(label)

imagematrix = numpy.array(imagematrix)
imagelabels = numpy.array(imagelabels)

# save numpy arrays for future use
numpy.save("matrix.npy", imagematrix)
numpy.save("labels.npy", imagelabels)