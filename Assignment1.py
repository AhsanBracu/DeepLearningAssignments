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

# Exercise 1.1 — LoadBatch

def loadBatch(filename):
    with open(DATASET_PATH + filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].T.astype(np.float64) / 255.0  
    y = np.array(batch[b'labels'])                  

    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n), dtype=np.float64)        
    Y[y, np.arange(n)] = 1                            

    return X, Y, y


#  Used data from these three files for full project
X_train, Y_train, y_train = loadBatch('data_batch_1')
X_val,   Y_val,   y_val   = loadBatch('data_batch_2')
X_test,  Y_test,  y_test  = loadBatch('test_batch')

# check data
print('X_train shape:', X_train.shape, '  correct: (3072, 10000)')
print('Y_train shape:', Y_train.shape, '  correct: (10,   10000)')
print('y_train shape:', y_train.shape, '  correct: (10000,)')
print('X dtype:',       X_train.dtype, '  correct: float64')
print('Y dtype:',       Y_train.dtype, '  correct: float64')
print('X range:  min =', X_train.min(), ' max =', X_train.max(), '  correct: 0.0 to 1.0')
print('y range:  min =', y_train.min(), ' max =', y_train.max(), '  correct: 0 to 9')

# Exercise 1.2 — Normalize data

def normalizeData(X_train, X_val, X_test):

    mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
    std_X  = np.std(X_train,  axis=1).reshape(-1, 1)

    X_train = (X_train - mean_X) / std_X
    X_val   = (X_val   - mean_X) / std_X
    X_test  = (X_test  - mean_X) / std_X

    return X_train, X_val, X_test, mean_X, std_X


# After normalizeing
X_train, X_val, X_test, mean_X, std_X = normalizeData(X_train, X_val, X_test)

# check 1.2
print('\n--- After normalization ---')
print('X_train mean (should be ~0):', np.mean(X_train).round(6))
print('X_train std  (should be ~1):', np.std(X_train).round(6))
print('X_val   mean (close to 0):  ', np.mean(X_val).round(6))
print('X_test  mean (close to 0):  ', np.mean(X_test).round(6))
print('mean_X shape:', mean_X.shape, '  correct: (3072, 1)')
print('std_X  shape:', std_X.shape,  '  correct: (3072, 1)')

# Exercise 1.3 — Initialize network parameters W and b

K = 10
d = X_train.shape[0]  

# seed=42 as assignment suggests)
rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

# store W and b in a dictionary
init_net = {}
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))  
init_net['b'] = np.zeros((K, 1))                        

# check 1.3
print('\n--- Network initialization ---')
print('W shape:', init_net['W'].shape, '  correct: (10, 3072)')
print('b shape:', init_net['b'].shape, '  correct: (10, 1)')
print('W mean: ', init_net['W'].mean().round(6), '  correct: ~0')
print('W std:  ', init_net['W'].std().round(6),  '  correct: ~0.01')
print('b values:', init_net['b'].T,               '  correct: all zeros')

# Exercise 1.4 — ApplyNetwork

def softmax(s):
    # subtract max to prevent overflow — does not change result
    e = np.exp(s - np.max(s, axis=0))
    return e / np.sum(e, axis=0)


def applyNetwork(X, network):
    W = network['W'] 
    b = network['b'] 

    s = W @ X + b 
    P = softmax(s) 

    return P

P = applyNetwork(X_train[:, 0:100], init_net)

print('\n--- ApplyNetwork ---')
print('P shape:       ', P.shape,         '  correct: (10, 100)')
print('P min:         ', P.min().round(6), '  correct: > 0')
print('P max:         ', P.max().round(6), '  correct: < 1')
print('P col 0 sum:   ', P[:, 0].sum().round(6), '  correct: 1.0')
print('P col 99 sum:  ', P[:, 99].sum().round(6), '  correct: 1.0')

# Exercise 1.5 — ComputeLoss

def computeLoss(P, y):
    n = P.shape[1]
    probes_n = P[y, np.arange(n)]  

    # cross-entropy loss = -log of those probabilities, averaged
    L = -np.sum(np.log(probes_n)) / n

    return L


# check loss on first 100 training images
P = applyNetwork(X_train[:, 0:100], init_net)
L = computeLoss(P, y_train[0:100])

print('\n--- ComputeLoss ---')
print('Loss:', L.round(6))
print('correct: close to log(10) =', np.log(10).round(6),
      ' (random model gives ~equal prob to all 10 classes)')

# Exercise 1.6 — ComputeAccuracy

def computeAccuracy(P, y):
   
    predicted = np.argmax(P, axis=0) 
    acc = np.mean(predicted == y)   

    return acc


# check accuracy on first 100 training images
P = applyNetwork(X_train[:, 0:100], init_net)
acc = computeAccuracy(P, y_train[0:100])

print('\n--- ComputeAccuracy ---')
print('Accuracy:', acc.round(4))
print('correct: ~0.10 (10%) — random model guesses 1 out of 10 classes correctly')

# Exercise 1.7 — BackwardPass (Part 1: error matrix G)

def backwardPass(X, Y, P, network, lam):
    n = X.shape[1]      
    W = network['W']   
    G = P - Y

    # gradients ───
    grad_W = G @ X.T / n + 2 * lam * W 

    # b vector
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n  

    # dictionarty to hold gradients
    grads = {}
    grads['W'] = grad_W
    grads['b'] = grad_b

    return grads


#Quick check on first 3 training images
P_check = applyNetwork(X_train[:, 0:3], init_net)
grads = backwardPass(X_train[:, 0:3], Y_train[:, 0:3], P_check, init_net, 0)

print('\n--- BackwardPass Part 2: grad_W and grad_b ---')
print('grad_W shape:', grads['W'].shape, '  correct: (10, 3072)')
print('grad_b shape:', grads['b'].shape, '  correct: (10, 1)')
print('grad_W mean: ', grads['W'].mean().round(8))
print('grad_b mean: ', grads['b'].mean().round(8))

# Exercise 1.7 — Part 3: Verify gradients with PyTorch

import torch

def computeGradsWithTorch(X, y, network_params):
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


def computeGradsWithTorchLam(X, y, network_params, lam):
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

# Compare code results with PyTorch results. Returns relative error — should be < 1e-6.
def computeRelativeError(ga, gn, eps=1e-10):
    return np.max(np.abs(ga - gn) / np.maximum(eps, np.abs(ga) + np.abs(gn)))


# Test on small data — as assignment suggests 
d_small = 10
n_small = 3
lam     = 0

small_net = {}
rng.bit_generator.state = BitGen(seed).state   
small_net['W'] = 0.01 * rng.standard_normal(size=(10, d_small))
small_net['b'] = np.zeros((10, 1))

# small data
X_small = X_train[0:d_small, 0:n_small]
Y_small = Y_train[:, 0:n_small]
y_small = y_train[0:n_small]

# compute gradients both ways
P_small    = applyNetwork(X_small, small_net)
my_grads   = backwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads = computeGradsWithTorch(X_small, y_small, small_net)

# compare
err_W = computeRelativeError(my_grads['W'], torch_grads['W'])
err_b = computeRelativeError(my_grads['b'], torch_grads['b'])

print('\n--- Gradient verification (lam=0) ---')
print('Relative error grad_W:', err_W, '  correct: < 1e-6')
print('Relative error grad_b:', err_b, '  correct: < 1e-6')
print('grad_W OK:', err_W < 1e-6)
print('grad_b OK:', err_b < 1e-6)

# ── Also test with lam > 0 ─────────────────────────────────────────────────
lam = 0.1
my_grads_reg    = backwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads_reg = computeGradsWithTorchLam(X_small, y_small, small_net, lam)

err_W_reg = computeRelativeError(my_grads_reg['W'], torch_grads_reg['W'])
err_b_reg = computeRelativeError(my_grads_reg['b'], torch_grads_reg['b'])

print('\n--- Gradient verification (lam=0.1) ---')
print('Relative error grad_W:', err_W_reg, '  correct: < 1e-6')
print('Relative error grad_b:', err_b_reg, '  correct: < 1e-6')
print('grad_W OK:', err_W_reg < 1e-6)
print('grad_b OK:', err_b_reg < 1e-6)


# Exercise 1.8 — MiniBatchGD (Part 1: function signature + setup)

import copy

def miniBatchGD(X, Y, X_val, Y_val, GDparams, init_net, lam, rng, augment=False,decay_every= None):
   # deep copy so init_net is not modified
    trained_net = copy.deepcopy(init_net)

    # unpack parameters
    n_batch  = GDparams['n_batch']
    eta      = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    n = X.shape[1]   

    train_losses = []
    val_losses   = []
    train_costs  = []
    val_costs    = []

       # (Bonus 2.1b) code given 
    if augment:
        aa = np.int32(np.arange(32)).reshape((32, 1))
        bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
        vv = np.tile(32*aa, (1, 32))
        ind_flip  = vv.reshape((32*32, 1)) + np.tile(bb, (32, 1))
        inds_flip = np.vstack((ind_flip, 1024+ind_flip))
        inds_flip = np.vstack((inds_flip, 2048+ind_flip))
        inds_flip = inds_flip.flatten()  
 

    for epoch in range(n_epochs):

        indices    = rng.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        # loop over mini-batches
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end   = (j + 1) * n_batch

            X_batch = X_shuffled[:, j_start:j_end]   # (d, n_batch)
            Y_batch = Y_shuffled[:, j_start:j_end]   # (K, n_batch)

              # ── Bonus 2.1(b): flip
            if augment:
                for i in range(X_batch.shape[1]):
                    if rng.random() < 0.5:
                        X_batch[:, i] = X_batch[inds_flip, i]

            # forward pass
            P_batch = applyNetwork(X_batch, trained_net)

            # backward pass
            grads = backwardPass(X_batch, Y_batch, P_batch, trained_net, lam)

            # update W and b — equations (8, 9)
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        # ── compute and save loss after each epoch ─────────────────────
        # training loss
        P_train    = applyNetwork(X, trained_net)
        train_loss = computeLoss(P_train, np.argmax(Y, axis=0))
        train_cost = train_loss + lam * np.sum(trained_net['W'] ** 2)

        # validation loss
        P_val    = applyNetwork(X_val, trained_net)
        val_loss = computeLoss(P_val, np.argmax(Y_val, axis=0))
        val_cost = val_loss + lam * np.sum(trained_net['W'] ** 2)

        # save for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        print(f'Epoch {epoch+1}/{n_epochs} — train cost: {train_cost:.4f}  val cost: {val_cost:.4f}')

        # ── Part (d): step decay — reduce eta every decay_every epochs ─
        if decay_every is not None and (epoch + 1) % decay_every == 0:
            eta = eta / 10
            print(f'  → learning rate decayed to {eta:.6f}')

    return trained_net, train_losses, val_losses, train_costs, val_costs


# train with assignment parameters 
GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}

rng.bit_generator.state = BitGen(seed).state
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))
init_net['b'] = np.zeros((K, 1))

print('\n--- MiniBatchGD training (n_batch=100, eta=0.001, n_epochs=40, lam=0) ---')
trained_net, train_losses, val_losses, train_costs, val_costs = miniBatchGD(
    X_train, Y_train, X_val, Y_val, GDparams, init_net, lam=0, rng=rng)

# final accuracy on test set
P_test   = applyNetwork(X_test, trained_net)
acc_test = computeAccuracy(P_test, y_test)
print(f'\nTest accuracy: {acc_test:.4f}  expected: ~0.3913')

# Plotting — loss curves and W visualization

def plotLossCurves(train_losses, val_losses, train_costs, val_costs, title=''):
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


def visualizeWeights(trained_net, title=''):
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


# Plot results from our training run 
plotLossCurves(train_losses, val_losses, train_costs, val_costs,
               title='lam=0, eta=0.001, n_epochs=40, n_batch=100')
visualizeWeights(trained_net,
                 title='Learned W — lam=0, eta=0.001, n_epochs=40')

# 4 Experiments — as assignment requires

def runExperiment(X_train, Y_train, y_train,
                  X_val,   Y_val,
                  X_test,  y_test,
                  lam, eta, n_epochs, n_batch, rng):

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
    trained, train_losses, val_losses, train_costs, val_costs = miniBatchGD(
        X_train, Y_train, X_val, Y_val, GDparams, net, lam, rng)

    # test accuracy
    P_test   = applyNetwork(X_test, trained)
    acc_test = computeAccuracy(P_test, y_test)
    print(f'\nTest accuracy: {acc_test:.4f}')

    # plot loss curves
    plotLossCurves(train_losses, val_losses, train_costs, val_costs, title=title)

    # visualize W
    visualizeWeights(trained, title=title)

    return acc_test


# ── Run all 4 experiments 
experiments = [
    {'lam': 0,   'eta': 0.1,   'n_epochs': 40, 'n_batch': 100},
    {'lam': 0,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 0.1, 'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 1,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
]

results = []
# for exp in experiments:
#     acc = runExperiment(
#         X_train, Y_train, y_train,
#         X_val,   Y_val,
#         X_test,  y_test,
#         lam=exp['lam'], eta=exp['eta'],
#         n_epochs=exp['n_epochs'], n_batch=exp['n_batch'],
#         rng=rng)
#     results.append((exp, acc))

#Summary
print('\n--- Summary of all experiments ---')
for exp, acc in results:
    print(f"lam={exp['lam']}, eta={exp['eta']} → test accuracy: {acc:.4f}")

def LoadAllBatches():
    # load all 5 batches and concatenate
    X_all, Y_all, y_all = [], [], []
    for i in range(1, 6):
        X, Y, y = loadBatch(f'data_batch_{i}')
        X_all.append(X)
        Y_all.append(Y)
        y_all.append(y)
 
    X_all = np.concatenate(X_all, axis=1) 
    Y_all = np.concatenate(Y_all, axis=1)   
    y_all = np.concatenate(y_all, axis=0)   
 
    X_val_b   = X_all[:, :1000]
    Y_val_b   = Y_all[:, :1000]
    y_val_b   = y_all[:1000]
    X_train_b = X_all[:, 1000:]
    Y_train_b = Y_all[:, 1000:]
    y_train_b = y_all[1000:]
 
    # test data
    X_test_b, Y_test_b, y_test_b = loadBatch('test_batch')
 
    # normalize using training statistics
    X_train_b, X_val_b, X_test_b, _, _ = normalizeData(X_train_b, X_val_b, X_test_b)
 
    return X_train_b, Y_train_b, y_train_b, X_val_b, Y_val_b, y_val_b, X_test_b, Y_test_b, y_test_b


X_train_b, Y_train_b, y_train_b, X_val_b,   Y_val_b,   y_val_b,   X_test_b,  Y_test_b,  y_test_b   = LoadAllBatches()
 
print('\n--- Bonus training 1 -----')
 
rng.bit_generator.state = BitGen(seed).state
bonus_net = {}
bonus_net['W'] = 0.01 * rng.standard_normal(size=(K, d))
bonus_net['b'] = np.zeros((K, 1))
 
GDparams_bonus = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}
 
trained_bonus, train_losses_b, val_losses_b, train_costs_b, val_costs_b = miniBatchGD(
    X_train_b, Y_train_b, X_val_b, Y_val_b,
    GDparams_bonus, bonus_net, lam=0.01, rng=rng, augment=True)
 
# test accuracy
P_bonus  = applyNetwork(X_test_b, trained_bonus)
acc_bonus = computeAccuracy(P_bonus, y_test_b)
print(f'\nBonus test accuracy: {acc_bonus:.4f}  (baseline was 0.3913)')
 
# plot and visualize
plotLossCurves(train_losses_b, val_losses_b, train_costs_b, val_costs_b,
               title='Bonus: all batches + augmentation, lam=0.01, eta=0.001')
visualizeWeights(trained_bonus, title='Bonus: all batches + augmentation')


# Bonus 2.1(a+b+d) — All batches + augmentation + learning rate decay
 
print('\n--- Bonus training: all batches + augmentation + lr decay ---')
 
# fresh network
rng.bit_generator.state = BitGen(seed).state
bonus_net2 = {}
bonus_net2['W'] = 0.01 * rng.standard_normal(size=(K, d))
bonus_net2['b'] = np.zeros((K, 1))
 
# decay eta by 10 every 20 epochs: 0.001 → 0.0001 → 0.00001
GDparams_bonus2 = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}
 
trained_bonus2, train_losses_b2, val_losses_b2, train_costs_b2, val_costs_b2 = miniBatchGD(
    X_train_b, Y_train_b, X_val_b, Y_val_b,
    GDparams_bonus2, bonus_net2, lam=0.01, rng=rng,
    augment=True, decay_every=20)
 
# test accuracy
P_bonus2   = applyNetwork(X_test_b, trained_bonus2)
acc_bonus2 = computeAccuracy(P_bonus2, y_test_b)
print(f'\nBonus test accuracy (a+b+d): {acc_bonus2:.4f}')
print(f'Baseline:                     0.3913')
print(f'With (a+b):                   0.4164')
print(f'With (a+b+d):                 {acc_bonus2:.4f}')
 
# plot
plotLossCurves(train_losses_b2, val_losses_b2, train_costs_b2, val_costs_b2,
               title='Bonus a+b+d: augment + lr decay, lam=0.01')
visualizeWeights(trained_bonus2, title='Bonus a+b+d: augment + lr decay')