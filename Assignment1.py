import pickle
import numpy as np
import matplotlib.pyplot as plt
 
# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH = 'Datasets/cifar-10-batches-py/'

with open(DATASET_PATH + 'data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

X = dict[b'data'].astype(np.float64) / 255.0
X = X.transpose()
nn = X.shape[1]
 
# Reshape each image from a column vector to a 3d array
X_im = X.reshape((32, 32, 3, nn), order='F')
X_im = np.transpose(X_im, (1, 0, 2, 3))

ni = 5
fig, axs = plt.subplots(1, 5, figsize=(10, 5))
for i in range(ni):
    axs[i].imshow(X_im[:, :, :, i])
    axs[i].axis('off')
plt.pause(8)

# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.1 — LoadBatch
# ══════════════════════════════════════════════════════════════════════════

def LoadBatch(filename):
    """
    Read a CIFAR-10 batch file and return image data, one-hot labels, labels.
 
    Returns:
        X -- pixel data,   shape (d, n) = (3072, 10000), dtype float64, range [0,1]
        Y -- one-hot labels shape (K, n) = (10, 10000),  dtype float64
        y -- raw labels,   shape (n,)   = (10000,),      dtype int
    """
    with open(DATASET_PATH + filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].T.astype(np.float64) / 255.0  # shape (3072, 10000)
    y = np.array(batch[b'labels']) 
    Y = np.eye(10)[y].T  # shape (10, 10000)
    return X, Y, y


X_train, Y_train, y_train = LoadBatch('data_batch_1')
X_val,   Y_val,   y_val   = LoadBatch('data_batch_2')
X_test,  Y_test,  y_test  = LoadBatch('test_batch')
 
# ── Sanity checks ──────────────────────────────────────────────────────────
print('X_train shape:', X_train.shape, '  expected: (3072, 10000)')
print('Y_train shape:', Y_train.shape, '  expected: (10,   10000)')
print('y_train shape:', y_train.shape, '  expected: (10000,)')
print('X dtype:',       X_train.dtype, '  expected: float64')
print('Y dtype:',       Y_train.dtype, '  expected: float64')
print('X range:  min =', X_train.min(), ' max =', X_train.max(), '  expected: 0.0 to 1.0')
print('y range:  min =', y_train.min(), ' max =', y_train.max(), '  expected: 0 to 9')