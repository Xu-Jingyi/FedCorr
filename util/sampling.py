# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)]) 
    return dict_users
