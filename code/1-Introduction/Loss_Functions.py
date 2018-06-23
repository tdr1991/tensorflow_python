"""
 * @Author: 汤达荣 
 * @Date: 2018-06-21 15:31:08 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-06-21 15:31:08 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
# 回归问题损失函数
# 均方误差（MSE）
def mse(true, pred):
    """
    true: array of true values
    pred: array of predicted values
    return: mean square error loss
    """
    return np.sum((true - pred)**2)

# 平均绝对值误差
def mae(true, pred):
    """
    true: array of true values
    pred: array of predicted values
    return: mean absolute error loss
    """
    return np.sum(np.abs(true - pred))

# Huber损失
def sm_mae(true, pred, delta):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

# Log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)



# 绘图
def plot_base(pred, loss, title, path):
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.plot(pred, loss)
    ax1.set_xlabel("Predictions")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    fig.tight_layout()
    #plt.show()
    plt.savefig(path)

def quan(true, pred, theta):
    loss = np.where(true >= pred, theta*(np.abs(true-pred)), (1-theta)*(np.abs(true-pred)))
    return np.sum(loss)

def plot_quan():
    fig, ax1 = plt.subplots(1,1, figsize = (7,5))

    target = np.repeat(0, 1000) 
    pred = np.arange(-10,10, 0.02)

    quantiles = [0.25, 0.5, 0.75]

    losses_quan = [[quan(target[i], pred[i], q) for i in range(len(pred))] for q in quantiles]

    # plot 
    for i in range(len(quantiles)):
        ax1.plot(pred, losses_quan[i], label = quantiles[i])
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Quantile Loss')
    ax1.set_title("Loss with Predicted values (Color: Quantiles)")
    ax1.legend()

    fig.tight_layout()
    plt.savefig("img/ls/quan.png")

# 绘制 Huber Loss
def plot_huber():
    fig, ax1 = plt.subplots(1,1, figsize = (7,5))

    target = np.repeat(0, 1000) 
    pred = np.arange(-10,10, 0.02)

    delta = [0.1, 1, 10]

    losses_huber = [[sm_mae(target[i], pred[i], q) for i in range(len(pred))] for q in delta]

    # plot 
    for i in range(len(delta)):
        ax1.plot(pred, losses_huber[i], label = delta[i])
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Loss')
    ax1.set_title("Huber Loss/ Smooth MAE Loss vs. Predicted values (Color: Deltas)")
    ax1.legend()
    ax1.set_ylim(bottom=-1, top = 15)

    fig.tight_layout()
    plt.savefig("img/ls/huber.png")

# 所有的损失绘在一个图中
def plot_all():
    fig, ax1 = plt.subplots(1,1, figsize = (10,6.5))

    target = np.repeat(0, 1000) 
    pred = np.arange(-10,10, 0.02)

    # calculating loss function for all predictions. 
    loss_mse = [mse(target[i], pred[i]) for i in range(len(pred))]
    loss_mae = [mae(target[i], pred[i]) for i in range(len(pred))]
    loss_sm_mae1 = [sm_mae(target[i], pred[i], 5) for i in range(len(pred))]
    loss_sm_mae2 = [sm_mae(target[i], pred[i], 10) for i in range(len(pred))]
    loss_logcosh = [logcosh(target[i], pred[i]) for i in range(len(pred))]
    loss_quan1 = [quan(target[i], pred[i], 0.25) for i in range(len(pred))]


    losses = [loss_mse, loss_mae, loss_sm_mae1, loss_sm_mae2, loss_logcosh, loss_quan1]
    names = ['MSE', 'MAE','Huber (5)', 'Huber (10)', 'Log-cosh', 'Quantile (0.25)']
    cmap = ['#d53e4f',
    '#fc8d59',
    '#fee08b',
    '#e6f598',
    '#99d594',
    '#3288bd']

    for lo in range(len(losses)):
        ax1.plot(pred, losses[lo], label = names[lo], color= cmap[lo])
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Loss')
    ax1.set_title("Loss with Predicted values")
    ax1.legend()
    ax1.set_ylim(bottom=0, top=40)

    fig.savefig("img/ls/all.png")

# 分类问题损失函数
# 二值交叉熵或负log似然估计
def bin_ce(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: binary cross entropy loss
    """
    loss = np.where(true==1, np.log(pred), np.log(1-pred))
    return -np.sum(loss)

# Focal loss
def focal(true, pred, gamma):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: binary cross entropy loss
    """
    loss = np.where(true==1, (1-pred)**gamma*(np.log(pred)), pred**gamma*(np.log(1-pred)))
    return -np.sum(loss)

def plot_focal():
    fig, ax1 = plt.subplots(1,1)

    # array of same target value 10000 times
    target = np.repeat(1, 10000) # considering prediction to be 1
    pred = np.arange(0,1, 0.0001) # all predictions b/w 0 and 1 for 10k values

    # calculating loss function for all predictions. 
    gammas = [0, 0.5, 1, 2, 5]
    losses_focal = [[focal(target[i], pred[i], gamma) for i in range(len(pred))] for gamma in gammas]

    # plot for binary cross entropy
    for i in range(len(gammas)):
        ax1.plot(pred, losses_focal[i], label = gammas[i])
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Focal Loss')
    ax1.set_title("Loss with Predicted values (Color: Gammas)")
    ax1.legend()

    # make right and top lines invisible
    ax1.spines['top'].set_visible(False)    # Make the top axis line for a plot invisible
    ax1.spines['right'].set_visible(False) # Make the right axis line for a plot invisible

    fig.tight_layout()
    fig.savefig("img/ls/focal.png")

# Hinge loss
def hinge(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: negative log likelihood loss
    """
    loss = np.max((0, (1 - pred*true)))
    return np.sum(loss)

# square loss
def sq_loss(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: negative log likelihood loss
    """
    loss = (1 - pred*true)**2
    return np.sum(loss)

# Logistic loss
def log_loss(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: negative log likelihood loss
    """
    loss = np.log(1 + np.exp(-(pred*true)))/np.log(2)
    return np.sum(loss)

# Exponential loss


# Kullback–Leibler divergence
def kld(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: KL divergence loss
    """
    loss = pred*(np.log(pred) - true)
    return np.sum(loss)

# Embedding loss (have 2 inputs and compare them)
# 
# * Hinge embedding criteria
# * L1 Hinge embedding
# * Cosine distance

# ### Miscelaneus losses
# 
# * Haversine distance
# * Weighted average of muliple losses

def plot_cl(pred, loss, title, path):
    fig, ax1 = plt.subplots(1,1)

    # plot for binary cross entropy
    ax1.plot(pred, loss)
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Loss')
    ax1.set_title(title)

    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    """
    target = np.repeat(100, 10000)
    pred = np.arange(-10000, 10000, 2)
    pvalues = []
    loss = []
    title = []
    path = []
    pvalues.append(pred)
    loss.append([mse(target[i], pred[i]) for i in range(len(pred))])
    title.append("MSE Loss vs. Predictions")
    path.append("img/ls/mse.png")
    
    pvalues.append(pred)
    loss.append([mae(target[i], pred[i]) for i in range(len(pred))])
    title.append("MAE Loss vs. Predictions")
    path.append("img/ls/mae.png")

    # 多变量 for 循环需使用 zip 函数
    for v, l, t, p in zip(pvalues, loss, title, path):
        plot_base(v, l, t, p)
    plot_huber()
    target = np.repeat(0, 1000) 
    pred = np.arange(-10, 10, 0.02)

    loss = [logcosh(target[i], pred[i]) for i in range(len(pred))]
    title = "Log-Cosh Loss vs. Predictions"
    path = "img/ls/logcosh.png"
    plot_base(pred, loss, title, path)
    plot_quan()
    plot_all()
    """
    # array of same target value 10000 times
    target = np.repeat(1, 10000) # considering prediction to be 1
    pred = np.arange(0,1, 0.0001) # all predictions b/w 0 and 1 for 10k values

    pvalues = []
    loss = []
    title = []
    path = []
    pvalues.append(pred)
    # calculating loss function for all predictions. 
    loss.append([bin_ce(target[i], pred[i]) for i in range(len(pred))])
    title.append("Binary Cross Entropy Loss/ Log Loss")
    path.append("img/ls/EntorLog.png")

    pvalues.append(pred)
    # calculating loss function for all predictions. 
    loss.append([hinge(target[i], pred[i]) for i in range(len(pred))])
    title.append("Hinge Loss")
    path.append("img/ls/hinge.png")

    pvalues.append(pred)
    # calculating loss function for all predictions.  
    loss.append([sq_loss(target[i], pred[i]) for i in range(len(pred))])
    title.append("Square Loss")
    path.append("img/ls/square.png")

    pvalues.append(pred)
    # calculating loss function for all predictions.  
    loss.append([sq_loss(target[i], pred[i]) for i in range(len(pred))])
    title.append("Logistic Loss")
    path.append("img/ls/log.png")
    """
    pvalues.append(pred)
    # calculating loss function for all predictions.  
    loss.append([expo(target[i], pred[i], 100) for i in range(len(pred))])
    title.append("Exponential Loss")
    path.append("img/ls/exp.png")
    """

    for v, l, t, p in zip(pvalues, loss, title, path):
        plot_base(v, l, t, p)
    plot_focal()