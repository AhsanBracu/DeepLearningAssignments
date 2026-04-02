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
plt.pause(3)

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

mean_X = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1) 
 
# ── Sanity checks ──────────────────────────────────────────────────────────

print('Mean check:', mean_X.shape, '  expected: (3072, 1)')
print('X_train shape:', X_train.shape, '  expected: (3072, 10000)')
print('Y_train shape:', Y_train.shape, '  expected: (10,   10000)')
print('y_train shape:', y_train.shape, '  expected: (10000,)')
print('X dtype:',       X_train.dtype, '  expected: float64')
print('Y dtype:',       Y_train.dtype, '  expected: float64')
print('X range:  min =', X_train.min(), ' max =', X_train.max(), '  expected: 0.0 to 1.0')
print('y range:  min =', y_train.min(), ' max =', y_train.max(), '  expected: 0 to 9')


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.2 — Normalize data
# ══════════════════════════════════════════════════════════════════════════
 
def NormalizeData(X_train, X_val, X_test):
     mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
     std_X  = np.std(X_train,  axis=1).reshape(-1, 1)
     X_train = (X_train - mean_X) / std_X
     X_val   = (X_val   - mean_X) / std_X
     X_test  = (X_test  - mean_X) / std_X
     return X_train, X_val, X_test, mean_X, std_X

X_train, X_val, X_test, mean_X, std_X = NormalizeData(X_train, X_val, X_test)


# ── Sanity checks ──────────────────────────────────────────────────────────
# print('\n--- After normalization ---')
# print('X_train mean (should be ~0):', np.mean(X_train).round(6))
# print('X_train std  (should be ~1):', np.std(X_train).round(6))
# print('X_val   mean (close to 0):  ', np.mean(X_val).round(6))
# print('X_test  mean (close to 0):  ', np.mean(X_test).round(6))
# print('mean_X shape:', mean_X.shape, '  expected: (3072, 1)')
# print('std_X  shape:', std_X.shape,  '  expected: (3072, 1)')


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.3 — Initialize network parameters W and b
# ══════════════════════════════════════════════════════════════════════════
 
# K = number of classes, d = number of pixel features
K = 10
d = X_train.shape[0]   # 3072
 
# set up reproducible random number generator (seed=42 as assignment suggests)
rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state
 
# store W and b in a dictionary
init_net = {}
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))  # (10, 3072) Gaussian, std=0.01
init_net['b'] = np.zeros((K, 1))                          # (10, 1)    all zeros
 
# ── Sanity checks ──────────────────────────────────────────────────────────
print('\n--- Network initialization ---')
print('W shape:', init_net['W'].shape, '  expected: (10, 3072)')
print('b shape:', init_net['b'].shape, '  expected: (10, 1)')
print('W mean: ', init_net['W'].mean().round(6), '  expected: ~0')
print('W std:  ', init_net['W'].std().round(6),  '  expected: ~0.01')
print('b values:', init_net['b'].flatten(), '  expected: all zeros')

# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.4 — ApplyNetwork
# ══════════════════════════════════════════════════════════════════════════
 
def softmax(s):
    """Convert scores to probabilities. Numerically stable version."""
    # subtract max to prevent overflow — does not change result
    e = np.exp(s - np.max(s, axis=0))
    return e / np.sum(e, axis=0)
 
 
def ApplyNetwork(X, network):
    """
    Apply the network to input X and return probabilities P.
 
    Equations:
        s = Wx + b      (1) — compute scores
        p = softmax(s)  (2) — convert to probabilities
 
    Args:
        X       -- input images,  shape (d, n) = (3072, n)
        network -- dictionary with keys 'W' (K×d) and 'b' (K×1)
 
    Returns:
        P -- probabilities, shape (K, n) = (10, n)
             each column sums to 1.0
    """
    W = network['W']   # (K, d) = (10, 3072)
    b = network['b']   # (K, 1) = (10, 1)
 
    s = W @ X + b      # (10, 3072) @ (3072, n) + (10, 1) = (10, n)
    P = softmax(s)     # (10, n) — probabilities between 0 and 1
 
    return P
 
 
# ── Top-level: check on first 100 training images ─────────────────────────
P = ApplyNetwork(X_train[:, 0:100], init_net)
 
print('\n--- ApplyNetwork ---')
print('P shape:       ', P.shape,         '  expected: (10, 100)')
print('P min:         ', P.min().round(6), '  expected: > 0')
print('P max:         ', P.max().round(6), '  expected: < 1')
print('P col 0 sum:   ', P[:, 0].sum().round(6), '  expected: 1.0')
print('P col 99 sum:  ', P[:, 99].sum().round(6), '  expected: 1.0')


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.5 — ComputeLoss
# ══════════════════════════════════════════════════════════════════════════
 
def ComputeLoss(P, y):
    """
    Compute mean cross-entropy loss over a batch of images.
 
    Formula: L = (1/n) * sum(-log(p_y))
    p_y = probability the model gave to the true class of each image
 
    Args:
        P -- probabilities, shape (K, n) = (10, n)
        y -- true labels,   shape (n,)   integers 0-9
 
    Returns:
        L -- scalar, mean cross-entropy loss
    """
    n = P.shape[1]
 
    # pick the probability of the true class for each image
    # P[y, np.arange(n)] picks P[y[0],0], P[y[1],1], ..., P[y[n-1],n-1]
    correct_class_probs = P[y, np.arange(n)]   # shape (n,)
 
    # cross-entropy loss = -log of those probabilities, averaged
    L = -np.sum(np.log(correct_class_probs)) / n
 
    return L
 
 
# ── Top-level: check loss on first 100 training images ────────────────────
P = ApplyNetwork(X_train[:, 0:100], init_net)
L = ComputeLoss(P, y_train[0:100])
 
print('\n--- ComputeLoss ---')
print('Loss:', L.round(6))
print('Expected: close to log(10) =', np.log(10).round(6),
      ' (random model gives ~equal prob to all 10 classes)')



# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.6 — ComputeAccuracy
# ══════════════════════════════════════════════════════════════════════════
 
def ComputeAccuracy(P, y):
    """
    Compute the accuracy of the network's predictions.
 
    Equation (4): predicted class = argmax of probabilities per image
    Accuracy = percentage of images where predicted class == true class
 
    Args:
        P -- probabilities, shape (K, n) = (10, n)
        y -- true labels,   shape (n,)   integers 0-9
 
    Returns:
        acc -- scalar, percentage of correct predictions (0.0 to 1.0)
    """
    # argmax along axis=0 — find highest prob class for each image (column)
    predicted = np.argmax(P, axis=0)   # shape (n,)
 
    # compare predicted class with true class
    acc = np.mean(predicted == y)      # fraction of correct predictions
 
    return acc
 
 
# ── Top-level: check accuracy on first 100 training images ────────────────
P = ApplyNetwork(X_train[:, 0:100], init_net)
acc = ComputeAccuracy(P, y_train[0:100])
 
print('\n--- ComputeAccuracy ---')
print('Accuracy:', acc.round(4))
print('Expected: ~0.10 (10%) — random model guesses 1 out of 10 classes correctly')


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.7 — BackwardPass (Part 1: error matrix G)
# ══════════════════════════════════════════════════════════════════════════

def BackwardPass(X, Y, P, network, lam):
    """
    Compute gradients of cost J w.r.t. W and b.
 
    Equations (10, 11):
        grad_W = (1/n) * G @ X.T  +  2*lam*W
        grad_b = (1/n) * sum(G, axis=1)
 
    Args:
        X       -- input images,      shape (d, n)
        Y       -- one-hot labels,    shape (K, n)
        P       -- predicted probs,   shape (K, n)
        network -- dictionary with W (K×d) and b (K×1)
        lam     -- regularization parameter λ
 
    Returns:
        grads -- dictionary with keys 'W' (K×d) and 'b' (K×1)
    """
    n = X.shape[1]        # number of images
    W = network['W']      # (K, d)
 
    # ── Part 1: error matrix G = P - Y ───────────────────────────────
    # shape (K, n) = (10, n)
    G = P - Y
 
    # ── Part 2: gradients ─────────────────────────────────────────────
    # grad_W — equation (10)
    # (K×n) @ (n×d) = (K×d) then average and add regularization
    grad_W = G @ X.T / n + 2 * lam * W   # shape (K, d) = (10, 3072)
 
    # grad_b — equation (11)
    # sum G across columns (images), average, make column vector
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n   # shape (K, 1) = (10, 1)
 
    # store in dictionary — keys match network dictionary
    grads = {}
    grads['W'] = grad_W
    grads['b'] = grad_b
 
    return grads

# ── Quick check ───────────────────────────────────────────────────────────
P_check = ApplyNetwork(X_train[:, 0:3], init_net)
grads = BackwardPass(X_train[:, 0:3], Y_train[:, 0:3], P_check, init_net, 0)
 
print('\n--- BackwardPass Part 2: grad_W and grad_b ---')
print('grad_W shape:', grads['W'].shape, '  expected: (10, 3072)')
print('grad_b shape:', grads['b'].shape, '  expected: (10, 1)')
print('grad_W mean: ', grads['W'].mean().round(8))
print('grad_b mean: ', grads['b'].mean().round(8))


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.7 — Part 3: Verify gradients with PyTorch
# ══════════════════════════════════════════════════════════════════════════
 
import torch
 
def ComputeGradsWithTorch(X, y, network_params):
    """Compute gradients using PyTorch — lam=0 (loss only)."""
    Xt = torch.from_numpy(X)
    W  = torch.tensor(network_params['W'], requires_grad=True)
    b  = torch.tensor(network_params['b'], requires_grad=True)
    N  = X.shape[1]
    scores = torch.matmul(W, Xt) + b
    P = torch.nn.Softmax(dim=0)(scores)
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))
    loss.backward()
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()
    return grads

def ComputeGradsWithTorchLam(X, y, network_params, lam):
    """Compute gradients using PyTorch — includes L2 regularization (lam > 0)."""
    Xt = torch.from_numpy(X)
    W  = torch.tensor(network_params['W'], requires_grad=True)
    b  = torch.tensor(network_params['b'], requires_grad=True)
    N  = X.shape[1]
    scores = torch.matmul(W, Xt) + b
    P = torch.nn.Softmax(dim=0)(scores)
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))
    cost = loss + lam * torch.sum(torch.multiply(W, W))
    cost.backward()
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()
    return grads
 
 
def ComputeRelativeError(ga, gn, eps=1e-10):
    """
    Relative error between analytical (ga) and numerical/torch (gn) gradients.
    Should be < 1e-6 to be considered correct.
    """
    return np.max(np.abs(ga - gn) / np.maximum(eps, np.abs(ga) + np.abs(gn)))
 
def BackwardPass(X, Y, P, network, lam):
    """
    Compute gradients of cost J w.r.t. W and b.
 
    Equations (10, 11):
        grad_W = (1/n) * G @ X.T  +  2*lam*W
        grad_b = (1/n) * sum(G, axis=1)
 
    Args:
        X       -- input images,      shape (d, n)
        Y       -- one-hot labels,    shape (K, n)
        P       -- predicted probs,   shape (K, n)
        network -- dictionary with W (K×d) and b (K×1)
        lam     -- regularization parameter λ
 
    Returns:
        grads -- dictionary with keys 'W' (K×d) and 'b' (K×1)
    """
    n = X.shape[1]        # number of images
    W = network['W']      # (K, d)
 
    # ── Part 1: error matrix G = P - Y ───────────────────────────────
    # shape (K, n) = (10, n)
    G = P - Y
 
    # ── Part 2: gradients ─────────────────────────────────────────────
    # grad_W — equation (10)
    # (K×n) @ (n×d) = (K×d) then average and add regularization
    grad_W = G @ X.T / n + 2 * lam * W   # shape (K, d) = (10, 3072)
 
    # grad_b — equation (11)
    # sum G across columns (images), average, make column vector
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n   # shape (K, 1) = (10, 1)
 
    # store in dictionary — keys match network dictionary
    grads = {}
    grads['W'] = grad_W
    grads['b'] = grad_b
 
    return grads
 
 
# # ── Quick check ───────────────────────────────────────────────────────────
# P_check = ApplyNetwork(X_train[:, 0:3], init_net)
# grads = BackwardPass(X_train[:, 0:3], Y_train[:, 0:3], P_check, init_net, 0)
 
# print('\n--- BackwardPass Part 2: grad_W and grad_b ---')
# print('grad_W shape:', grads['W'].shape, '  expected: (10, 3072)')
# print('grad_b shape:', grads['b'].shape, '  expected: (10, 1)')
# print('grad_W mean: ', grads['W'].mean().round(8))
# print('grad_b mean: ', grads['b'].mean().round(8))
 

 # ── Test on small data — as assignment suggests ────────────────────────────
d_small = 10
n_small = 3
lam     = 0
 
# small network
small_net = {}
rng.bit_generator.state = BitGen(seed).state   # reset seed for reproducibility
small_net['W'] = 0.01 * rng.standard_normal(size=(10, d_small))
small_net['b'] = np.zeros((10, 1))
 
# small data
X_small = X_train[0:d_small, 0:n_small]
Y_small = Y_train[:, 0:n_small]
y_small = y_train[0:n_small]
 
# compute gradients both ways
P_small    = ApplyNetwork(X_small, small_net)
my_grads   = BackwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)
 
# compare
err_W = ComputeRelativeError(my_grads['W'], torch_grads['W'])
err_b = ComputeRelativeError(my_grads['b'], torch_grads['b'])
 
print('\n--- Gradient verification (lam=0) ---')
print('Relative error grad_W:', err_W, '  expected: < 1e-6')
print('Relative error grad_b:', err_b, '  expected: < 1e-6')
print('grad_W OK:', err_W < 1e-6)
print('grad_b OK:', err_b < 1e-6)
 
# ── Also test with lam > 0 ─────────────────────────────────────────────────
lam = 0.1
my_grads_reg    = BackwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads_reg = ComputeGradsWithTorchLam(X_small, y_small, small_net, lam)
 
err_W_reg = ComputeRelativeError(my_grads_reg['W'], torch_grads_reg['W'])
err_b_reg = ComputeRelativeError(my_grads_reg['b'], torch_grads_reg['b'])
 
print('\n--- Gradient verification (lam=0.1) ---')
print('Relative error grad_W:', err_W_reg, '  expected: < 1e-6')
print('Relative error grad_b:', err_b_reg, '  expected: < 1e-6')
print('grad_W OK:', err_W_reg < 1e-6)
print('grad_b OK:', err_b_reg < 1e-6)


# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.8 — MiniBatchGD (Part 1: function signature + setup)
# ══════════════════════════════════════════════════════════════════════════
 
import copy
 
def MiniBatchGD(X, Y, GDparams, init_net, lam, rng):
    """
    Train the network using mini-batch gradient descent.
 
    Args:
        X        -- training images,  shape (d, n)
        Y        -- one-hot labels,   shape (K, n)
        GDparams -- dictionary with keys:
                      n_batch  — mini-batch size
                      eta      — learning rate
                      n_epochs — number of epochs
        init_net -- dictionary with keys 'W' and 'b' (initial parameters)
        lam      -- regularization parameter λ
        rng      -- random number generator (for reproducibility)
 
    Returns:
        trained_net -- dictionary with keys 'W' and 'b' (trained parameters)
    """
    # deep copy so init_net is not modified
    trained_net = copy.deepcopy(init_net)
 
    # unpack parameters
    n_batch  = GDparams['n_batch']
    eta      = GDparams['eta']
    n_epochs = GDparams['n_epochs']
 
    n = X.shape[1]   # number of training images
 
    return trained_net   # temporary — will add training loop next
 
 
# ── Quick check: function runs and returns correct shapes ──────────────────
GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 20}
trained_net = MiniBatchGD(X_train, Y_train, GDparams, init_net, lam=0, rng=rng)
 
print('\n--- MiniBatchGD Part 1: setup ---')
print('trained_net W shape:', trained_net['W'].shape, '  expected: (10, 3072)')
print('trained_net b shape:', trained_net['b'].shape, '  expected: (10, 1)')
print('init_net unchanged: ', np.allclose(init_net['W'], trained_net['W']), '  expected: True (deep copy)')



# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.8 — MiniBatchGD (Part 1: function signature + setup)
# ══════════════════════════════════════════════════════════════════════════
 
import copy
 
def MiniBatchGD(X, Y, X_val, Y_val, GDparams, init_net, lam, rng):
    """
    Train the network using mini-batch gradient descent.
 
    Args:
        X        -- training images,   shape (d, n)
        Y        -- one-hot labels,    shape (K, n)
        X_val    -- validation images, shape (d, n_val)
        Y_val    -- validation labels, shape (K, n_val)
        GDparams -- dictionary with keys:
                      n_batch  — mini-batch size
                      eta      — learning rate
                      n_epochs — number of epochs
        init_net -- dictionary with keys 'W' and 'b' (initial parameters)
        lam      -- regularization parameter λ
        rng      -- random number generator (for reproducibility)
 
    Returns:
        trained_net  -- dictionary with keys 'W' and 'b' (trained parameters)
        train_losses -- list of training loss after each epoch
        val_losses   -- list of validation loss after each epoch
    """
    # deep copy so init_net is not modified
    trained_net = copy.deepcopy(init_net)
 
    # unpack parameters
    n_batch  = GDparams['n_batch']
    eta      = GDparams['eta']
    n_epochs = GDparams['n_epochs']
 
    n = X.shape[1]   # number of training images
 
    # lists to track loss history for plotting
    train_losses = []
    val_losses   = []
 
    # ── loop over epochs ──────────────────────────────────────────────
    for epoch in range(n_epochs):
 
        # shuffle training data at start of each epoch
        indices    = rng.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]
 
        # loop over mini-batches
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end   = (j + 1) * n_batch
 
            X_batch = X_shuffled[:, j_start:j_end]   # (d, n_batch)
            Y_batch = Y_shuffled[:, j_start:j_end]   # (K, n_batch)
 
            # forward pass
            P_batch = ApplyNetwork(X_batch, trained_net)
 
            # backward pass
            grads = BackwardPass(X_batch, Y_batch, P_batch, trained_net, lam)
 
            # update W and b — equations (8, 9)
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']
 
        # ── compute and save loss after each epoch ─────────────────────
        # training loss
        P_train    = ApplyNetwork(X, trained_net)
        train_loss = ComputeLoss(P_train, np.argmax(Y, axis=0))
        train_cost = train_loss + lam * np.sum(trained_net['W'] ** 2)
 
        # validation loss
        P_val    = ApplyNetwork(X_val, trained_net)
        val_loss = ComputeLoss(P_val, np.argmax(Y_val, axis=0))
        val_cost = val_loss + lam * np.sum(trained_net['W'] ** 2)
 
        # save for plotting
        train_losses.append(train_cost)
        val_losses.append(val_cost)
 
        print(f'Epoch {epoch+1}/{n_epochs} — train loss: {train_cost:.4f}  val loss: {val_cost:.4f}')
 
    return trained_net, train_losses, val_losses
 
 
# ── Top-level: train with assignment parameters ───────────────────────────
GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}
 
# reset seed so results match assignment expected values
rng.bit_generator.state = BitGen(seed).state
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))
init_net['b'] = np.zeros((K, 1))
 
print('\n--- MiniBatchGD training (n_batch=100, eta=0.001, n_epochs=20, lam=0) ---')
trained_net, train_losses, val_losses = MiniBatchGD(
    X_train, Y_train, X_val, Y_val, GDparams, init_net, lam=0, rng=rng)
 
# final accuracy on test set
P_test   = ApplyNetwork(X_test, trained_net)
acc_test = ComputeAccuracy(P_test, y_test)
print(f'\nTest accuracy: {acc_test:.4f}  expected: ~0.35')


# ══════════════════════════════════════════════════════════════════════════
# Plotting — loss curves and W visualization
# ══════════════════════════════════════════════════════════════════════════
 
def PlotLossCurves(train_losses, val_losses, title=''):
    """
    Plot training and validation loss curves after each epoch.
    Matches Figure 3 in the assignment.
    """
    epochs = range(1, len(train_losses) + 1)
 
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, val_losses,   label='validation loss', color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def VisualizeWeights(trained_net, title=''):
    """
    Visualize each row of W as a class template image.
    Follows assignment code exactly.
    """
    # rearrange W rows into displayable images — exactly as assignment says
    Ws   = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
 
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    classes  = ['airplane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
 
    for i in range(10):
        w_im      = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        row, col  = i // 5, i % 5
        axs[row, col].imshow(w_im_norm)
        axs[row, col].set_title(classes[i])
        axs[row, col].axis('off')
        # save each class template image as assignment suggests
        plt.imsave(f'w_class_{classes[i]}_{title}.png', w_im_norm)
 
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
 
# ── Plot results from our training run ────────────────────────────────────
PlotLossCurves(train_losses, val_losses,
               title='lam=0, eta=0.001, n_epochs=40, n_batch=100')
VisualizeWeights(trained_net,
                 title='Learned W — lam=0, eta=0.001, n_epochs=40')