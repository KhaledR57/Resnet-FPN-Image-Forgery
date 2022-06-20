import math
import numpy
import random
from pathlib import Path
from tensorflow.math import sigmoid
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.utils import Sequence, set_random_seed
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from resnet_keras_fpn import FPNRes101e200

######################
# Global Configuration
######################
set_random_seed(0)
IMAGE_SHAPE = (256,256,3)
NUM_CLASSES = 1
###########
# Utilities
###########

# The BinaryAccuracy metric assume a probabilistic output For binary
# classification, the recommendation, however, is to use the logits aware
# BinaryCrossentropy loss function and assign a linear activation function to
# the last layer, consequently this metrics can not be used. The following
# class reimplements the metric but with manual sigmoid application for use
# with binary classifiers.
class LogitsBinaryAccuracy(BinaryAccuracy):
    def update_state(self, reference, prediction, sample_weight = None):
        return super().update_state(reference, sigmoid(prediction), sample_weight)

##################
# Dataset Sequence
##################
class FeatureImagesSequence(Sequence):
    def __init__(self, imagePaths, batchSize = 32):
        self.imagePaths = imagePaths
        self.batchSize = batchSize

    def __len__(self):
        return math.ceil(len(self.imagePaths) / self.batchSize)

    def __getitem__(self, batchIndex):
        batch = self.imagePaths[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize]
        features = numpy.array([numpy.load(path) for path in batch])
        labels = numpy.array([not path.name.startswith("0") for path in batch])
        return features, labels

featureImagePaths = list(Path("dataset/featureImages/").glob("*"))
random.shuffle(featureImagePaths)
trainingCount = int(len(featureImagePaths) * 0.8)
trainingFeatureImagePaths = featureImagePaths[:trainingCount]
validationFeatureImagePaths = featureImagePaths[trainingCount:]

#################
# Construct Model
#################

baseModel = ResNet50V2(include_top = False, pooling = "avg") #FPNRes101e200(img_shape, num_classes)
"""baseModel.trainable = False
lastLayerOutput = baseModel.layers[-1].output
predictionLayerOutput = Dense(1, name = "predictions")(lastLayerOutput)
predictionModel = Model(inputs = baseModel.input, outputs = predictionLayerOutput)"""
predictionModel = FPNRes101e200(IMAGE_SHAPE, NUM_CLASSES)
predictionModel.compile(optimizer = Adam(), loss = BinaryCrossentropy(from_logits = True),
                    metrics = [LogitsBinaryAccuracy()])

####################
# Train And Validate
####################
earlyStopping = EarlyStopping(monitor = "val_binary_accuracy", mode = "max",
    patience = 10, restore_best_weights = True)
modelCheckpoint = ModelCheckpoint(filepath = "checkpoints/weights_checkpoint", save_weights_only = True,
    monitor = "val_binary_accuracy", mode = "max", save_best_only = True)
history = predictionModel.fit(FeatureImagesSequence(trainingFeatureImagePaths), epochs = 100,
    validation_data = FeatureImagesSequence(validationFeatureImagePaths),
    callbacks = (earlyStopping, modelCheckpoint))

"""history = model.fit(train_img,
                    train_lab,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_img, test_lab),
                    shuffle=True)"""

#####################
# Analyse And Record
#####################
accuracy = numpy.max(history.history["val_binary_accuracy"])

predictionProbabilities = sigmoid(predictionModel(validationFeatures))
rocCurve = roc_curve(validationLabels, predictionProbabilities)

predictions = predictionProbabilities > 0.5
confusionMatrix = confusion_matrix(validationLabels, predictions).ravel()

lossCurve = history.history["loss"]

################
# Output Results
################
print(f"Accuracy: {accuracy}")

print(f"True Negative: {confusionMatrix[0]}")
print(f"False Positive: {confusionMatrix[1]}")
print(f"False Negative: {confusionMatrix[2]}")
print(f"True Positive: {confusionMatrix[3]}")

rocCurve = rocCurve
numpy.savetxt("documentation/ROC.data", numpy.column_stack(rocCurve),
        comments = "", header = "falsePositiveRate truePositiveRate threshold")

lossCurve = lossCurve
numpy.savetxt("documentation/loss.data", lossCurve, comments = "", header = "loss")
