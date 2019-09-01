from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model, model_from_json
from keras.callbacks import TensorBoard

# Variables
rows, cols = 64, 64
batch_size = 32
epochs = 10

# Input Data
rash_path = Path('Training Data') / 'Dogs'
not_rash_path = Path('Training Data') / 'Not_Dogs'

images = []
labels = []

for img in not_rash_path.glob('*.png'):
    img = image.load_img(img) # Disk Load
    img = image.img_to_array(img) # Numpy array
    images.append(img)
    labels.append(0)

for img in rash_path.glob('*.png'):
    img = image.load_img(img) # Disk Load
    img = image.img_to_array(img) # Numpy array
    images.append(img)
    labels.append(1)

x_train = np.array(images)
y_train = np.array(labels)
x_train = vgg16.preprocess_input(x_train) # VGG16 Normalise

# Feature Extraction
pretrained_network = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(rows, cols, 3))

x_features = pretrained_network.predict(x_train) # Extract features in one pass

# Save features
joblib.dump(x_features, "x_train.dat")
joblib.dump(y_train, "y_train.dat")

# Load features
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Define Model
logger = TensorBoard(log_dir='logs', write_graph=True)
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:], name='Flatten_Layer'))

model.add(Dense(256, activation='relu', name='Final_Hidden_Layer'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid', name='Output_Layer'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train Model
model.fit(x_train, y_train, epochs=epochs, shuffle=True, callbacks=[logger])
#results = model.evaluate(x_test, y_test, verbose=0)
#print('Validation Loss: {} Validation Accuracy: {}'.format(results[0], results[1]))

# Save Model
#model_structure = model.to_json()
#f = Path("model_structure.json")
#f.write_text(model_structure)
#model.save_weights("model_weights.h5")

# Load Model
f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

# Input Image
img = image.load_img("dog.png", target_size=(rows, cols))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = vgg16.preprocess_input(img) # VGG16 Normalise

# Test model
features = pretrained_network.predict(img) # Extract Features
result = model.predict(features)[0][0]
print("Dog Confidence: {:.4f}%".format(int(result * 100)))



















