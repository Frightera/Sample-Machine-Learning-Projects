# %% Necessary Things
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from pca_plotter import PCAPlotter

# %% Gpu Setup for TF 2.x
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
# %% Import Data

(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train.shape # (60000, 28, 28)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1] * x_train.shape[2])) / 255.0 # normalize
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1] * x_test.shape[2])) / 255.0 # normalize

# %% Plot Triplets

def plot_triplets(triplet):
    plt.figure(figsize= (10,3))
    for each in range(len(triplet)):
        plt.subplot(1, 3, each+1)
        plt.imshow(np.reshape(triplet[each], (28,28)), cmap = 'binary')
    plt.show()

plot_triplets((x_train[2], x_train[4], x_train[3]))    

# %% Batch Of Triplets
def create_batch(batch_size=256):
    x_anchors = np.zeros((batch_size, 784))
    x_positives = np.zeros((batch_size, 784))
    x_negatives = np.zeros((batch_size, 784))
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, x_train.shape[0] - 1)
        x_anchor = x_train[random_index]
        y = y_train[random_index]
        
        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))
        
        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        
    return [x_anchors, x_positives, x_negatives]

for each in range(3):
    each = create_batch(1)
    plot_triplets(each)
    
# %% Embedding Model

emb_size = 64

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(emb_size, activation='sigmoid')
])

embedding_model.summary()
"""
Total params: 54,400
Trainable params: 54,400
Non-trainable params: 0
"""

# %% Siamese Network

input_anchor = tf.keras.layers.Input(shape=(784,))
input_positive = tf.keras.layers.Input(shape=(784,))
input_negative = tf.keras.layers.Input(shape=(784,))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model(inputs = [input_anchor, input_positive, input_negative], outputs = output)
net.summary()
"""  
Total params: 54,400 # Number of parameters are same because they are shared.
Trainable params: 54,400
Non-trainable params: 0
"""  
    
# %% Triplet Loss
alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

# %% Data Generator
def data_generator(batch_size=256):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y
        
# %% Train
batch_size = 1024
epochs = 24
steps_per_epoch = int(x_train.shape[0]/batch_size)

es_cb = tf.keras.callbacks.EarlyStopping(monitor = 'loss', verbose = 1, patience = 3) 

net.compile(loss=triplet_loss, optimizer='adam')

tf.keras.utils.plot_model(net, 'siamese.png', show_shapes= True)

history = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=1,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            x_test[:1000], y_test[:1000]
        ),
        es_cb]
)
#@ If you train longer, clusters will be more recognizable
"""
Epoch 24/24
58/58 [==============================] - 32s 553ms/step - loss: 0.0105
""" 

# %% Accuracy
from tensorflow.keras import backend as K

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# After you create some pairs, you can measure the accuracy. 
    
    
