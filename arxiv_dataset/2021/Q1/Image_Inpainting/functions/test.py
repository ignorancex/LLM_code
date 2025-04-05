#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINITE-VOLUMES FULL 2D
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes as fv



import mnist
test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[0,:]

#plots.plot_image(example)


intensity = 0.3

damage = np.random.choice(np.arange(example.size), replace=False, size=int(example.size * intensity))
example[damage] = 0
#plots.plot_image(example)
import time
start = time.time()
restored_example = fv.temporal_loop(example, damage)
print("Total time: {:.2f}".format(time.time()-start))

plots.plot_image(restored_example)

# %%
"""
FINITE-VOLUMES SPLITTING
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes_split as fvs



import mnist
test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[0,:]

#plots.plot_image(example)


intensity = 0.3

damage = np.random.choice(np.arange(example.size), replace=False, size=int(example.size * intensity))
example[damage] = 0
#plots.plot_image(example)

import time
start = time.time()
restored_example = fvs.temporal_loop_split(example, damage)
print("Total time: {:.2f}".format(time.time()-start))


plots.plot_image(restored_example)

# %%
"""
FINITE-VOLUMES PARALLELIZATION
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes_par as fvp

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
num_proc = mp.cpu_count()


import mnist
test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[0,:]

#plots.plot_image(example)


intensity = 0.3

damage = np.random.choice(np.arange(example.size), replace=False, size=int(example.size * intensity))
example[damage] = 0
#plots.plot_image(example)

import time
start = time.time()
restored_example = fvp.temporal_loop_par(example, damage, num_proc)
print("Total time: {:.2f}".format(time.time()-start))


plots.plot_image(restored_example)


# %%
"""
TRAIN NN
"""
import mnist
import neural_network as nn

train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Normalize the test image as CH Equation work for (-1,1)
train_images = (train_images / 255) *2-1

# Flatten the test images
train_images = train_images.reshape((-1,784))

model, history = nn.training(train_images, train_labels)

# %%
"""
VALIDATE NN
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


plots.loss_acc_plots(history)

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the test image as CH Equation work for (-1,1)
test_images = (test_images / 255) *2-1

# Flatten the test images
test_images = test_images.reshape((-1,784))
test_loss, test_accuracy = model.evaluate(test_images, to_categorical(test_labels), verbose=2)



# %%
"""
FINITE-VOLUMES FULL 2D FOR A GROUP OF IMAGES
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes as fv



import mnist

n_images = 5
indices_images = range(n_images)

test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[indices_images,:]


intensity = 0.3

damage = np.zeros((len(indices_images), int(example.shape[1] * intensity)), dtype=int)

for i in indices_images:

    
    damage[i, :] = np.random.choice(np.arange(example.shape[1]), replace=False, size=int(example.shape[1] * intensity))
#    print(damage[i, :])
    example[i, damage[i, :]] = 0
    
restored_example = np.zeros(example.shape)

for i in range(n_images):
    restored_example[i,:] = fv.temporal_loop(example[i, :], damage[i, :])
    

# %%

"""
PREDICTION OF A GROUP OF IMAGES
"""

predictions_damaged = np.argmax(model.predict(example), axis=1)  
predictions_restored = np.argmax(model.predict(restored_example), axis=1)    
    
print(predictions_damaged)
print(predictions_restored)

#to_categorical(test_labels[indices_images], num_classes=10)
model.evaluate(
 example,
 to_categorical(test_labels[indices_images], num_classes=10)
)

model.evaluate(
 restored_example,
 to_categorical(test_labels[indices_images], num_classes=10)
)
# %%
"""
CUSTOMIZED TRAINING LOOP
"""

import mnist
import neural_network as nn

train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Normalize the test image as CH Equation work for (-1,1)
train_images = (train_images / 255) *2-1

# Flatten the test images
train_images = train_images.reshape((-1,784))

model = nn.customized_training(train_images, train_labels)

# %%
"""
VALIDATION OF CUSTOMIZED TRAINING LOOP
"""

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the test image as CH Equation work for (-1,1)
test_images = (test_images / 255) *2-1

# Flatten the test images
test_images = test_images.reshape((-1,784))

nn.customized_validation(test_images, test_labels, model)