import pickle
import numpy as np
import matplotlib.pyplot as plt

# images path
cifar_dir = 'Datasets/cifar-10-batches-py/'

with open(cifar_dir + 'data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

X = dict[b'data'].astype(np.float64) / 255.0
X = X.transpose()
nn = X.shape[1]
X_im = X.reshape((32, 32, 3, nn), order='F')
X_im = np.transpose(X_im, (1, 0, 2, 3))

# Display some images
ni = 5
fig, axs = plt.subplots(1, 5, figsize=(10, 5))
for i in range(ni):
    axs[i].imshow(X_im[:, :, :, i])
    axs[i].axis('off')
plt.show()

# Exercise 1.1
def loadBatch(filename):
    with open(cifar_dir + filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].T.astype(np.float64) / 255.0  
    y = np.array(batch[b'labels'])                  

    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n), dtype=np.float64)        
    Y[y, np.arange(n)] = 1                            

    return X, Y, y


X_train, Y_train, y_train = loadBatch('data_batch_1')
X_val,   Y_val,   y_val   = loadBatch('data_batch_2')
X_test,  Y_test,  y_test  = loadBatch('test_batch')


# Exercise 1.2
def normalizeData(X_train, X_val, X_test):
    mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
    std_X  = np.std(X_train,  axis=1).reshape(-1, 1)

    X_train = (X_train - mean_X) / std_X
    X_val   = (X_val   - mean_X) / std_X
    X_test  = (X_test  - mean_X) / std_X
    return X_train, X_val, X_test, mean_X, std_X

X_train, X_val, X_test, mean_X, std_X = normalizeData(X_train, X_val, X_test)


# Exercise 1.3
K = 10
d = X_train.shape[0]  

rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

init_net = {}
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))  
init_net['b'] = np.zeros((K, 1))                        


# Exercise 1.4
def softmax(s):
    e = np.exp(s - np.max(s, axis=0))
    return e / np.sum(e, axis=0)


def applyNetwork(X, network):
    W = network['W'] 
    b = network['b']
    s = W @ X + b 
    P = softmax(s)
    return P

P = applyNetwork(X_train[:, 0:100], init_net)

# Exercise 1.5
def computeLoss(P, y):
    n = P.shape[1]
    probes_n = P[y, np.arange(n)]  

    L = -np.sum(np.log(probes_n)) / n
    return L


# Exercise 1.6
def computeAccuracy(P, y):
    predicted = np.argmax(P, axis=0) 
    acc = np.mean(predicted == y)
    return acc


# Exercise 1.7
def backwardPass(X, Y, P, network, lam):
    n = X.shape[1]      
    W = network['W']   
    G = P - Y
    grad_W = G @ X.T / n + 2 * lam * W
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n

    grads = {}
    grads['W'] = grad_W
    grads['b'] = grad_b
    return grads


# Exercise 1.7  (PyTorch)
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

def computeRelativeError(ga, gn, eps=1e-10):
    return np.max(np.abs(ga - gn) / np.maximum(eps, np.abs(ga) + np.abs(gn)))


d_small = 10
n_small = 3
lam     = 0

small_net = {}
rng.bit_generator.state = BitGen(seed).state   
small_net['W'] = 0.01 * rng.standard_normal(size=(10, d_small))
small_net['b'] = np.zeros((10, 1))

X_small = X_train[0:d_small, 0:n_small]
Y_small = Y_train[:, 0:n_small]
y_small = y_train[0:n_small]

P_small    = applyNetwork(X_small, small_net)
my_grads   = backwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads = computeGradsWithTorch(X_small, y_small, small_net)

err_W = computeRelativeError(my_grads['W'], torch_grads['W'])
err_b = computeRelativeError(my_grads['b'], torch_grads['b'])

print('Relative error grad_W:', err_W)
print('Relative error grad_b:', err_b,)
print('grad_W OK:', err_W < 1e-6)
print('grad_b OK:', err_b < 1e-6)

lam = 0.1
my_grads_reg    = backwardPass(X_small, Y_small, P_small, small_net, lam)
torch_grads_reg = computeGradsWithTorchLam(X_small, y_small, small_net, lam)

err_W_reg = computeRelativeError(my_grads_reg['W'], torch_grads_reg['W'])
err_b_reg = computeRelativeError(my_grads_reg['b'], torch_grads_reg['b'])

print('Relative error grad_W:', err_W_reg)
print('Relative error grad_b:', err_b_reg)
print('grad_W OK:', err_W_reg < 1e-6)
print('grad_b OK:', err_b_reg < 1e-6)

# Exercise 1.8
import copy

def miniBatchGD(X, Y, X_val, Y_val, GDparams, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)

    n_batch  = GDparams['n_batch']
    eta      = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    n = X.shape[1]   

    train_losses = []
    val_losses   = []
    train_costs  = []
    val_costs    = []

    for epoch in range(n_epochs):
        indices    = rng.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end   = (j + 1) * n_batch

            X_batch = X_shuffled[:, j_start:j_end]   
            Y_batch = Y_shuffled[:, j_start:j_end]   

            P_batch = applyNetwork(X_batch, trained_net)

            grads = backwardPass(X_batch, Y_batch, P_batch, trained_net, lam)

            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        P_train    = applyNetwork(X, trained_net)
        train_loss = computeLoss(P_train, np.argmax(Y, axis=0))
        train_cost = train_loss + lam * np.sum(trained_net['W'] ** 2)

        P_val    = applyNetwork(X_val, trained_net)
        val_loss = computeLoss(P_val, np.argmax(Y_val, axis=0))
        val_cost = val_loss + lam * np.sum(trained_net['W'] ** 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        print(f'Epoch {epoch+1}/{n_epochs} train cost: {train_cost:.4f}  val cost: {val_cost:.4f}')
    return trained_net, train_losses, val_losses, train_costs, val_costs


GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}

rng.bit_generator.state = BitGen(seed).state
init_net['W'] = 0.01 * rng.standard_normal(size=(K, d))
init_net['b'] = np.zeros((K, 1))

trained_net, train_losses, val_losses, train_costs, val_costs = miniBatchGD(
    X_train, Y_train, X_val, Y_val, GDparams, init_net, lam=0, rng=rng)

# final accuracy on test set
P_test   = applyNetwork(X_test, trained_net)
acc_test = computeAccuracy(P_test, y_test)
print(f'\nTest accuracy: {acc_test:.4f}')


def plotLossCurves(train_losses, val_losses, train_costs, val_costs, title=''):
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axs[0].plot(epochs, train_losses, label='training loss')
    axs[0].plot(epochs, val_losses,   label='validation loss', color='red')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss function')
    axs[0].legend()

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


plotLossCurves(train_losses, val_losses, train_costs, val_costs,
               title='lam=0, eta=0.001, n_epochs=40, n_batch=100')
visualizeWeights(trained_net,
                 title='Learned W — lam=0, eta=0.001, n_epochs=40')

# 4 Experiments
def runExperiment(X_train, Y_train, y_train,
                  X_val,   Y_val,
                  X_test,  y_test,
                  lam, eta, n_epochs, n_batch, rng):

    title = f'lam={lam}, eta={eta}, n_epochs={n_epochs}, n_batch={n_batch}'
    print(f'Experiment: {title}')

    rng.bit_generator.state = BitGen(seed).state
    net = {}
    net['W'] = 0.01 * rng.standard_normal(size=(K, d))
    net['b'] = np.zeros((K, 1))

    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}
    trained, train_losses, val_losses, train_costs, val_costs = miniBatchGD(
        X_train, Y_train, X_val, Y_val, GDparams, net, lam, rng)

    P_test   = applyNetwork(X_test, trained)
    acc_test = computeAccuracy(P_test, y_test)

    plotLossCurves(train_losses, val_losses, train_costs, val_costs, title=title)

    visualizeWeights(trained, title=title)

    return acc_test


experiments = [
    {'lam': 0,   'eta': 0.1,   'n_epochs': 40, 'n_batch': 100},
    {'lam': 0,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 0.1, 'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
    {'lam': 1,   'eta': 0.001, 'n_epochs': 40, 'n_batch': 100},
]

results = []
for exp in experiments:
    acc = runExperiment(
        X_train, Y_train, y_train,
        X_val,   Y_val,
        X_test,  y_test,
        lam=exp['lam'], eta=exp['eta'],
        n_epochs=exp['n_epochs'], n_batch=exp['n_batch'],
        rng=rng)
    results.append((exp, acc))

for exp, acc in results:
    print(f"lam={exp['lam']}, eta={exp['eta']} test accuracy: {acc:.4f}")