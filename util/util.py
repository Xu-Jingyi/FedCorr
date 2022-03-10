import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids
