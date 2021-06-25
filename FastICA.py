import numpy as np


def decorrelation(x):
    '''
    To decorrelation of matrix
    :param x: input matrix
    :return: matrix after decorrelation
    '''
    eigenValue, eigenVector = np.linalg.eigh(x @ x.T)
    de_x = np.dot(eigenVector * (1. / np.sqrt(eigenValue)) @ eigenVector.T, x)
    return de_x


def pca(x, feature_num, component_num):
    '''
    do pca white
    :param x: centered input signal (feature_number, sample_number)
    :param feature_num: length of signal
    :param component_num: number of signal sample
    :return: x_hat (feature_number, sample_number)
                topK (sample_number, sample_number)
    '''
    u, sigma, v = np.linalg.svd(x.T, full_matrices=False)
    topK = (u / sigma).T[:component_num]
    x_hat = np.dot(topK, x.T)
    x_hat *= np.sqrt(feature_num)
    return x_hat, topK


# non-linear functions model for negentropy fast ICA
def generateW(x, w, model):
    '''
    choose a non-linear function g(z), z = w @ x
    and generate new w:
                    w_new <- E(x(g(z))) - E(g^'(z))w
    :param x: preprocess input signal (sample_number, feature_number)
    :param w: old w (sample_number, sample_number)
    :param model: specific non-linear function
    :return: new w (sample_number, sample_number)
    '''
    gz, g_z = None, None
    if model == 'logconsh':
        gz, g_z = logcosh(w @ x)
    elif model == 'exp':
        gz, g_z = exp(w @ x)
    elif model == 'cub':
        gz, g_z = cube(w @ x)
    w_new = gz @ x.T / float(x.shape[1]) - g_z[:, np.newaxis] * w
    return w_new


# Some standard non-linear functions.
def logcosh(x):
    '''
    g=tanh(ax)
    :param x:(sample_number, feature_number)
    :return: gx (sample_number, feature_number)
            g_x derivative of gx (sample_number,)
    '''
    alpha = 1.0
    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0])
    for i, gx_i in enumerate(gx):
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
    return gx, g_x


def exp(x):
    '''
    g = xexp(-x^2/2)
    :param x:(sample_number, feature_number)
    :return: gx (sample_number, feature_number)
            g_x derivative of gx (sample_number,)
    '''
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)


def cube(x):
    '''
    g=x^3
    :param x: (sample_number, feature_number)
    :return: gx (sample_number, feature_number)
            g_x derivative of gx (sample_number,)
    '''
    return x ** 3, (3 * x ** 2).mean(axis=-1)


def kurtosis_ICA(x, eta=0.1, max_iterate=500, fast=False):
    '''
    The implement of gradient Ascent Kurtosis Maximization with/without fixed pointed
    :param x: input signal (feature_number, sample_number)
    :param eta: learning rate : float
    :param max_iterate: number of iterate : int
    :param fast: use to start fixed-point
    :return: separate signal:(feature_number, sample_number)
            separate matrix: (sample_number,sample_number)
            kurt_history: history of kurt
    '''
    # get number of components and number of features
    feature_num, component_num = x.shape
    # center x (feature_number, sample_number)
    x_centered = (x.T - np.mean(x, axis=0).reshape(-1, 1)).T
    # PCA white x_hat (sample_number, feature_number) K (sample_number, sample_number)
    x_hat, K = pca(x_centered, feature_num, component_num)

    # initialize w and decorrelation
    w = np.random.normal(size=(component_num, component_num))
    w = decorrelation(w)

    i = 0
    kurt_history = []
    while i < max_iterate:
        if fast:
            w_new = (w @ x.T) ** 3 @ x / float(x.shape[1]) - 3 * w
        else:
            kurt = ((w @ x.T) ** 4).mean(axis=-1) - 3 * (((w @ x.T) ** 2).mean(axis=-1)) ** 2
            kurt_history.append(kurt)
            w_new = w + eta * ((w @ x.T) ** 3 @ (np.sign(kurt) * x))
        w_new = decorrelation(w_new)
        if max(abs(abs(np.diag(np.dot(w_new, w.T))) - 1)) < 0.00001:
            w = w_new
            break
        w = w_new
        i += 1
    #print(i)
    return np.dot((w @ K), x_centered.T).T, kurt_history, w @ K


# FastICA
def negentropy_fastICA(x, model='logconsh', max_iterate=500):
    '''
    The implement of fixed-point Negentropy Maximization method
    :param x: input signal (feature_number, sample_number)
    :param model: non-linear function gz : string
    :param max_iterate: number of iterate : int
    :return: separate signal:(feature_number, sample_number)
            separate matrix: (sample_number,sample_number)
    '''
    # get number of components and number of features
    feature_num, component_num = x.shape
    # center x (feature_number, sample_number)
    x_centered = (x.T - np.mean(x, axis=0).reshape(-1, 1)).T
    # PCA white x_hat (sample_number, feature_number) K (sample_number, sample_number)
    x_hat, K = pca(x_centered, feature_num, component_num)

    # initialize w and decorrelation
    w = np.random.normal(size=(component_num, component_num))
    w = decorrelation(w)

    i = 0
    while i < max_iterate:
        w_new = generateW(x_hat, w, model)
        w_new = decorrelation(w_new)
        if max(abs(abs(np.diag(w_new @ w.T)) - 1)) < 0.00001:
            w = w_new
            break
        w = w_new
        i += 1
    #print(i)
    return np.dot((w @ K), x_centered.T).T, w @ K


def Infomax_nature_ICA(x, max_iterate=10000, miu=0.0001, miu_gamma=0.3):
    '''
    The implement of infomax natural gradient descent ICA
    :param x: input signal (feature_number, sample_number)
    :param max_iterate: number of iterate : int
    :param miu: learning rate : float
    :param miu_gamma: learning rate : float
    :return: separation signal: (feature_number, sample_number)
                separate matrix: (sample_number,sample_number)
    '''
    # get number of components and number of features
    feature_num, component_num = x.shape
    # center x
    x_centered = (x.T - np.mean(x, axis=0).reshape(-1, 1)).T
    # PCA white
    x_hat = decorrelation(x_centered.T)
    # Initialize separation matrix w and gamma
    w = np.random.rand(component_num, component_num)
    gamma = np.random.rand(component_num)
    gy = np.zeros((component_num, feature_num))
    I = np.eye(component_num)
    iteration = 0
    while iteration < max_iterate:
        # Update y
        y = w @ x_hat
        # Non-linear form is not determined 
        for i in range(component_num):
            gamma[i] = (1 - miu_gamma) * gamma[i] + miu_gamma * np.mean(
                -np.tanh(y[i, :]) * y[i, :] + (1 - np.tanh(y[i, :]) ** 2))
            if gamma[i] > 0:
                gy[i, :] = -2 * np.tanh(y[i, :])
            else:
                gy[i, :] = np.tanh(y[i, :]) - y[i, :]
        w_new = w + miu * (I + gy @ y.T) @ w
        if max(abs(abs(np.diag(np.dot(w_new, w.T))) - 1)) < 0.000001:
            w = w_new
            break
        w = w_new
        iteration += 1
    #print(iteration)
    return np.dot(w, x_hat).T, w
