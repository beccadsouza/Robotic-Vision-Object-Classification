from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy
import cv2

# Extract the image into vector
def image_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


imagematrix = numpy.load("matrix.npy")
imagelabels = numpy.load("labels.npy")

# Prepare data for training and testing
(train_img, test_img, train_label, test_label) = train_test_split(imagematrix, imagelabels, test_size=0.2, random_state=50)

'''SVM MODEL IN SKLEARN'''
model1 = SVC(max_iter=-1, kernel='linear', class_weight='balanced',gamma='scale')  # kernel linear is better Gausian kernel here
model1.fit(train_img, train_label)
acc1 = model1.score(test_img, test_label)
print("SVM model accuracy: {:.2f}%".format(acc1 * 100))


'''KNN MODEL IN SKLEARN'''
model2 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model2.fit(train_img, train_label)
acc2 = model2.score(test_img, test_label)
print("KNN model accuracy: {:.2f}%".format(acc2 * 100))

'''PREDICATION SAMPLE'''
for t in range(1,5):
  pixel = image_vector(cv2.imread("case{0}.jpg".format(t)))
  rawImage = numpy.array([pixel])
  prediction1 = model1.predict(rawImage)
  prediction2 = model2.predict(rawImage)
  print("Test Case {0}".format(t))
  print("Prediction by SVM - {0}".format(prediction1[0]))
  print("Prediction by KNN - {0}".format(prediction1[0]))
