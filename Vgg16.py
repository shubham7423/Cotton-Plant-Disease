import glob
import cv2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.utils import shuffle
import pickle
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
import time

# Import train data
train_imagePaths = glob.glob('data/data/train/*/*')
train_labels = []
train_images = []
print("=> Loading Training images")
for i,imagePath in enumerate(train_imagePaths):
    train_labels.append(imagePath.split('/')[-2])
    image = cv2.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    train_images.append(image)
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print("=> Loaded {} train images".format(train_images.shape[0]))
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)

# Import validation data
val_imagePaths = glob.glob('data/data/val/*/*')
val_labels = []
val_images = []
print("=> Loading Validation images")
for i,imagePath in enumerate(val_imagePaths):
    val_labels.append(imagePath.split('/')[-2])
    image = cv2.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    val_images.append(image)
    
val_images = np.array(val_images)
val_labels = np.array(val_labels)
print("=> Loaded {} validation images".format(val_images.shape[0]))
val_labels = lb.transform(val_labels)

train_images, train_labels = shuffle(train_images, train_labels)
val_images, val_labels = shuffle(val_images, val_labels)

# model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
predictions = Dense(4, activation='softmax')(x)
for layer in base_model.layers:
    layer.trainable = True
model = Model(inputs=base_model.input, outputs=predictions)



checkpoint = ModelCheckpoint("VGG16.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')

#compile model
EPOCHS = 400
BS = 32
opt = Adam(learning_rate=1e-6,  decay=1e-8)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

training_dataset = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
testing_dataset = ImageDataGenerator()

X = np.concatenate([train_images, val_images])
y = np.concatenate([train_labels, val_labels])


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)
training = training_dataset.flow(X_train, y_train, batch_size=32)
testing = testing_dataset.flow(X_test, y_test, batch_size=32)

print("=> Training model")
start_time = time.time()
r = model.fit(training, 
              validation_data=testing,
              steps_per_epoch=len(X_train)//BS,
              validation_steps=len(X_test)//BS,
              epochs=EPOCHS,
              callbacks=[checkpoint, early]
              )
print("\n\nTraining Time time: {}".format(time.time() - start_time))

model.save('model/VGG16-final.h5')

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
