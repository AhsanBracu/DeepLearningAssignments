import pickle
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH = 'Datasets/cifar-10-batches-py/'

# Load a batch of training data
with open(DATASET_PATH + 'data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

X = dict[b'data'].astype(np.float64) / 255.0
X = X.transpose()
nn = X.shape[1]
X_im = X.reshape((32, 32, 3, nn), order='F')
X_im = np.transpose(X_im, (1, 0, 2, 3))

# Display the first 5 images
ni = 5
fig, axs = plt.subplots(1, 5, figsize=(10, 5))
for i in range(ni):
    axs[i].imshow(X_im[:, :, :, i])
    axs[i].axis('off')
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.1 — LoadBatch
# ══════════════════════════════════════════════════════════════════════════

def LoadBatch(filename):
    """
    Read a CIFAR-10 batch file and return image data, one-hot labels, labels.
    """
    with open(DATASET_PATH + filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].T.astype(np.float64) / 255.0  # shape (3072, 10000)
    y = np.array(batch[b'labels'])                    # shape (10000,)

    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n), dtype=np.float64)            # shape (10, 10000)
    Y[y, np.arange(n)] = 1                            # set correct class to 1

    return X, Y, y


# ── Used data from these three files for full project─────────
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

# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.2 — Normalize data
# ══════════════════════════════════════════════════════════════════════════

def NormalizeData(X_train, X_val, X_test):
    """
    Shape the data according to the assignement requirements.
    """
    mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
    std_X  = np.std(X_train,  axis=1).reshape(-1, 1)

    X_train = (X_train - mean_X) / std_X
    X_val   = (X_val   - mean_X) / std_X
    X_test  = (X_test  - mean_X) / std_X

    return X_train, X_val, X_test, mean_X, std_X


# ── Top-level: normalize training, validation and test data ───────────────
X_train, X_val, X_test, mean_X, std_X = NormalizeData(X_train, X_val, X_test)

# ── Sanity checks ──────────────────────────────────────────────────────────
print('\n--- After normalization ---')
print('X_train mean (should be ~0):', np.mean(X_train).round(6))
print('X_train std  (should be ~1):', np.std(X_train).round(6))
print('X_val   mean (close to 0):  ', np.mean(X_val).round(6))
print('X_test  mean (close to 0):  ', np.mean(X_test).round(6))
print('mean_X shape:', mean_X.shape, '  expected: (3072, 1)')
print('std_X  shape:', std_X.shape,  '  expected: (3072, 1)')

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
print('b values:', init_net['b'].T,               '  expected: all zeros')

# ══════════════════════════════════════════════════════════════════════════
# Exercise 1.4 — ApplyNetwork
# ══════════════════════════════════════════════════════════════════════════

def softmax(s):
    """Get probability ."""
    # subtract max to prevent overflow — does not change result
    e = np.exp(s - np.max(s, axis=0))
    return e / np.sum(e, axis=0)


def ApplyNetwork(X, network):
    """
    use  s = Wx + b
    """
    W = network['W'] 
    b = network['b'] 

    s = W @ X + b 
    P = softmax(s) 

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
    Compute mean cross-entropy loss.
    """
    n = P.shape[1]
    probes_n = P[y, np.arange(n)]   # shape (n,)

    # cross-entropy loss = -log of those probabilities, averaged
    L = -np.sum(np.log(probes_n)) / n

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
    Compute the accuracy - get maximum probability.
    """
    predicted = np.argmax(P, axis=0) 
    acc = np.mean(predicted == y)   

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
    """
    n = X.shape[1]        # number of images
    W = network['W']   
    G = P - Y

    # gradients ───
    grad_W = G @ X.T / n + 2 * lam * W 

    # b vector
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n   #  (10, 1)

    # dictionarty to hold gradients
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
   Compare code results with PyTorch results. Returns relative error — should be < 1e-6.
    """
    return np.max(np.abs(ga - gn) / np.maximum(eps, np.abs(ga) + np.abs(gn)))


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

def MiniBatchGD(X, Y, X_val, Y_val, GDparams, init_net, lam, rng):
    """
    Train the network using mini-batch gradient descent.
    after each epoch, compute and save training and validation loss for plotting.
    """
    # deep copy so init_net is not modified
    trained_net = copy.deepcopy(init_net)

    # unpack parameters
    n_batch  = GDparams['n_batch']
    eta      = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    n = X.shape[1]   # number of training images

    # lists to track loss and cost history for plotting
    train_losses = []
    val_losses   = []
    train_costs  = []
    val_costs    = []

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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        print(f'Epoch {epoch+1}/{n_epochs} — train cost: {train_cost:.4f}  val cost: {val_cost:.4f}')

    return trained_net, train_losses, val_losses, train_costs, val_costs


# ── Top-level: train with assignment parameters ───────────────────────────
GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}

# reset seed so results match assignment expected values
rng.bit_generator.state = BitGen(seed).state
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))
init_net['b'] = np.zeros((K, 1))

print('\n--- MiniBatchGD training (n_batch=100, eta=0.001, n_epochs=40, lam=0) ---')
trained_net, train_losses, val_losses, train_costs, val_costs = MiniBatchGD(
    X_train, Y_train, X_val, Y_val, GDparams, init_net, lam=0, rng=rng)

# final accuracy on test set
P_test   = ApplyNetwork(X_test, trained_net)
acc_test = ComputeAccuracy(P_test, y_test)
print(f'\nTest accuracy: {acc_test:.4f}  expected: ~0.3913')

# ══════════════════════════════════════════════════════════════════════════
# Plotting — loss curves and W visualization
# ══════════════════════════════════════════════════════════════════════════

def PlotLossCurves(train_losses, val_losses, train_costs, val_costs, title=''):
    """
    Plot loss and cost curves for training and validation.
    Assignment requires both loss and cost graphs.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    # loss plot
    axs[0].plot(epochs, train_losses, label='training loss')
    axs[0].plot(epochs, val_losses,   label='validation loss', color='red')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss function')
    axs[0].legend()

    # cost plot
    axs[1].plot(epochs, train_costs, label='training cost')
    axs[1].plot(epochs, val_costs,   label='validation cost', color='red')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('cost')
    axs[1].set_title('Cost function')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'loss_curves_{title}.png', dpi=150, bbox_inches='tight')
    plt.pause(5)


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

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'weights_{title}.png', dpi=150, bbox_inches='tight')
    plt.pause(5)


# ── Plot results from our training run ────────────────────────────────────
PlotLossCurves(train_losses, val_losses, train_costs, val_costs,
               title='lam=0, eta=0.001, n_epochs=40, n_batch=100')
VisualizeWeights(trained_net,
                 title='Learned W — lam=0, eta=0.001, n_epochs=40')

# ══════════════════════════════════════════════════════════════════════════
# 4 Experiments — as assignment requires
# ══════════════════════════════════════════════════════════════════════════

def RunExperiment(X_train, Y_train, y_train,
                  X_val,   Y_val,
                  X_test,  y_test,
                  lam, eta, n_epochs, n_batch, rng):
    """
    Run one full experiment — train, plot, visualize, report accuracy.
    """
    title = f'lam={lam}, eta={eta}, n_epochs={n_epochs}, n_batch={n_batch}'
    print(f'\n{"="*60}')
    print(f'Experiment: {title}')
    print(f'{"="*60}')

    # fresh W and b for each experiment
    rng.bit_generator.state = BitGen(seed).state
    net = {}
    net['W'] = 0.01 * rng.standard_normal(size=(K, d))
    net['b'] = np.zeros((K, 1))

    # train
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}
    trained, train_losses, val_losses, train_costs, val_costs = MiniBatchGD(
        X_train, Y_train, X_val, Y_val, GDparams, net, lam, rng)

    # test accuracy
    P_test   = ApplyNetwork(X_test, trained)
    acc_test = ComputeAccuracy(P_test, y_test)
    print(f'\nTest accuracy: {acc_test:.4f}')

    # plot loss curves
    PlotLossCurves(train_losses, val_losses, train_costs, val_costs, title=title)

    # visualize W
    VisualizeWeights(trained, title=title)

    return acc_test


# ── Run all 4 experiments ──────────────────────────────────────────────────
experiments = [
    {'lam': 0,   'eta': 0.1,   'n_epochs': 40, 'n_batch': 100},
    {'lam': 0,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 0.1, 'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 1,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
]

results = []
for exp in experiments:
    acc = RunExperiment(
        X_train, Y_train, y_train,
        X_val,   Y_val,
        X_test,  y_test,
        lam=exp['lam'], eta=exp['eta'],
        n_epochs=exp['n_epochs'], n_batch=exp['n_batch'],
        rng=rng)
    results.append((exp, acc))

# ── Summary ────────────────────────────────────────────────────────────────
print('\n--- Summary of all experiments ---')
for exp, acc in results:
    print(f"lam={exp['lam']}, eta={exp['eta']} → test accuracy: {acc:.4f}")


def LoadAllBatches():
    # load all 5 batches and concatenate
    X_all, Y_all, y_all = [], [], []
    for i in range(1, 6):
        X, Y, y = LoadBatch(f'data_batch_{i}')
        X_all.append(X)
        Y_all.append(Y)
        y_all.append(y)
 
    X_all = np.concatenate(X_all, axis=1)   # (3072, 50000)
    Y_all = np.concatenate(Y_all, axis=1)   # (10,   50000)
    y_all = np.concatenate(y_all, axis=0)   # (50000,)
 
    # first 1000 images → validation, rest → training
    X_val_b   = X_all[:, :1000]
    Y_val_b   = Y_all[:, :1000]
    y_val_b   = y_all[:1000]
    X_train_b = X_all[:, 1000:]
    Y_train_b = Y_all[:, 1000:]
    y_train_b = y_all[1000:]
 
    # test data
    X_test_b, Y_test_b, y_test_b = LoadBatch('test_batch')
 
    # normalize using training statistics
    X_train_b, X_val_b, X_test_b, _, _ = NormalizeData(X_train_b, X_val_b, X_test_b)
 
    return X_train_b, Y_train_b, y_train_b, X_val_b, Y_val_b, y_val_b, X_test_b, Y_test_b, y_test_b
 
 
# ── Load all batches ───────────────────────────────────────────────────────
X_train_b, Y_train_b, y_train_b, \
X_val_b,   Y_val_b,   y_val_b,   \
X_test_b,  Y_test_b,  y_test_b   = LoadAllBatches()
 
print('\n--- Bonus 2.1(a): All batches loaded ---')
print('X_train_b shape:', X_train_b.shape, '  expected: (3072, 49000)')
print('X_val_b   shape:', X_val_b.shape,   '  expected: (3072, 1000)')
print('X_test_b  shape:', X_test_b.shape,  '  expected: (3072, 10000)')    