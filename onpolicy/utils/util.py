import numpy as np
import math
import torch
import random

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_linear_anneal(
        model, anneal_original, anneal_final, epoch, total_num_epochs,
        tau_anneal_epochs=0):
    """Update the policy temperature once per training epoch.

    ``tau_anneal_epochs=0`` retains the historical schedule exactly.  A
    positive value includes both endpoints in the requested number of epochs
    and then holds ``anneal_final``.  The latter is useful for separating an
    early exploration phase from a late stability phase without changing tau
    inside a rollout/PPO-update epoch.
    """
    tau_anneal_epochs = int(tau_anneal_epochs)
    if tau_anneal_epochs < 0:
        raise ValueError("tau_anneal_epochs must be non-negative")
    if tau_anneal_epochs == 0:
        progress = epoch / total_num_epochs
    elif tau_anneal_epochs == 1:
        progress = 1.0
    else:
        progress = min(max(epoch, 0) / (tau_anneal_epochs - 1), 1.0)
    model.tau = anneal_final + (
        (anneal_original - anneal_final) * (1 - progress)
    )

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape

def expand_slice(tensor, length, batch_size):
    return tensor.unsqueeze(0).expand(length, *tensor.shape).reshape(batch_size, *tensor.shape[1:])

def shuffle_dataset(nested_list, seed=None):
    n = len(nested_list)
    m = len(nested_list[0]) if n > 0 else 0

    flat_list = [item for sublist in nested_list for item in sublist]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(flat_list)
    else:
        random.shuffle(flat_list)

    shuffled_nested_list = [flat_list[i:i + m] for i in range(0, len(flat_list), m)]
    
    return shuffled_nested_list

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
