import torch
import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import sem
from tqdm import tqdm

import matplotlib as mpl

linewidth = 3.
mpl.rcParams['lines.linewidth'] = linewidth

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({'font.size': 22})



class Classification:
    def __init__(self, delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, init_th = None, seed = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if init_th is None:
            self.init_th = torch.randn(d, requires_grad = True)
        else:
            self.init_th = init_th
        self.d = d
        self.R = R
        self.alpha = alpha
        self.delta0 = delta0
        self.mean_noise = mean_noise
        self.spammer_weight = spammer_weight
        self.reg = reg
        
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        
        self.Sigma_0 = 0.25 * torch.eye(d)
        self.Sigma_1 = 0.25 * torch.eye(d)
    
    ###########################################################################
    # Helper methods, spam classification
    ###########################################################################    
    def lt_mu_1(self, th):
        return self.alpha * th + self.mu_1
    
    
    def delta(self, th, mu):
        return self.delta0
    
    
    def st_mu_1(self, th, mu):
        return self.delta(th, mu) * self.lt_mu_1(th) + (1 - self.delta(th, mu)) * mu
    
    
    def approx_lt_loss(self, th, n = 1000):
        n_spam = int(self.spammer_weight * n)
        
        non_spam_X = torch.tensor(np.random.multivariate_normal(self.mu_0, self.Sigma_0, n - n_spam)).float()
        non_spam_Y = torch.zeros(n - n_spam)
        
        spam_X = torch.tensor(np.random.multivariate_normal(self.lt_mu_1(th.detach()), self.Sigma_1, n_spam)).float()
        spam_Y = torch.ones(n_spam)
        
        X = torch.cat([non_spam_X, spam_X]).float()
        Y = torch.cat([non_spam_Y, spam_Y])
        
        H  = 1. / (1. + torch.exp(-X @ th))
        Ls = -(Y * torch.log(H) + (1 - Y) * torch.log(1. - H)) + (self.reg / 2) * (torch.linalg.norm(th) ** 2)
        
        return torch.sum(Ls) / len(Ls)            
    
    
    ###########################################################################
    # Stateful PerfGD methods
    ###########################################################################
    def finite_diff_approx(self, inputs, outputs):
        # Estimate the gradient of the one-step mean update with a finite difference
        # approximation.
        Delta_inputs  = torch.column_stack([i - inputs[-1] for i in inputs])
        Delta_outputs = torch.column_stack([o - outputs[-1] for o in outputs])
        st_mu_grad    = Delta_outputs @ torch.linalg.pinv(Delta_inputs)
        return st_mu_grad
    
    
    def approx_lt_dmu(self, st_mu_grad, k = None):
        # Approximates dmu_1* / dth given estimates for the derivatives of the
        # short-term mean.
        # These derivatives should be collected in st_mu_grad where the first 
        # d columns are d(st_mu) / dth and the last d columns are d(st_mu) / dmu.
        # k = number of steps to approximate with. If k = None, then we use the
        # approximation given by k --> \infty. Note that this is actually not valid
        # when || d2 m || >= 1.
        d1 = st_mu_grad[:, :self.d]
        d2 = st_mu_grad[:, self.d:]
        if k is None:
            return torch.linalg.solve(torch.eye(self.d) - d2, d1)
        else:
            return torch.linalg.solve(torch.eye(self.d) - d2, (torch.eye(self.d) - torch.linalg.matrix_power(d1, k)) @ d1)
    
        
    def approx_grad_1(self, th, mu, Sigma, y, n = 1000):
        # Computes the standard loss gradient (i.e. not accounting for dist. shift)
        # with feature mean mu, var Sigma, and all of the same labels y = 0 or 1.
        # Does this via MC with n samples.
        X = torch.tensor(np.random.multivariate_normal(mu, Sigma, n)).float()
        Y = y * torch.ones(n)
        
        H  = 1. / (1. + torch.exp(-X @ th))
        Ls = -(Y * torch.log(H) + (1 - Y) * torch.log(1. - H))
        
        loss = torch.sum(Ls) / n + (self.reg / 2) * (torch.linalg.norm(th) ** 2)
        if th.grad is not None:
            th.grad.zero_()
        loss.backward()
        
        return th.grad.data
    
    
    def approx_grad_2(self, th, mu, Sigma, y, lt_dmu_est, n = 1000):
        # Approximates the second half of the performative loss gradient (i.e. the
        # part which accounts for the distribution shift) with feature mean mu,
        # var Sigma, and all of the same labels y = 0 or 1. Also requires an estimate
        # for dmu^*/dth (lt_dmu_est). Does this via MC with n samples.
        # (lt_dmu_est)_ij = dmu_i / dth_j
        X = torch.tensor(np.random.multivariate_normal(mu, Sigma, n)).float()
        Y = y * torch.ones(n)
        
        H  = 1. / (1. + torch.exp(-X @ th))
        Ls = -(Y * torch.log(H) + (1 - Y) * torch.log(1. - H)) + (self.reg / 2 ) * (torch.linalg.norm(th) ** 2)
        
        return ((lt_dmu_est.T @ torch.linalg.inv(Sigma) @ (X.T - torch.outer(mu, torch.ones(n)))) @ Ls) / n    
    
    
    def SPGD(self, T, lr, k, horizon, pert):
        theta_t    = copy.deepcopy(self.init_th)
        mu_tm1     = copy.deepcopy(self.mu_1)
        est_mu_tm1 = mu_tm1 + self.mean_noise * torch.randn(self.d)
        params_t   = torch.cat([theta_t, mu_tm1])
        
        thetas   = deque()
        grad2s   = deque()
        inputs   = deque(maxlen = horizon)
        outputs  = deque(maxlen = horizon)
        
        with torch.no_grad():
            for t in range(self.d):
                thetas.append(theta_t.clone().detach())
                inputs.append(params_t.clone().detach())
                
                mu_t     = self.st_mu_1(theta_t, mu_tm1)
                est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
                outputs.append(est_mu_t.detach())
                
                theta_t    = torch.clamp(theta_t + lr * torch.randn(self.d), -self.R, self.R)
                mu_tm1     = mu_t
                est_mu_tm1 = est_mu_t
                params_t   = torch.cat([theta_t, est_mu_tm1])
        
        for t in range(T - self.d):
            theta_t.requires_grad = True
            thetas.append(theta_t.clone().detach())
            inputs.append(params_t.clone().detach())
            
            mu_t     = self.st_mu_1(theta_t, mu_tm1).detach()
            est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
            outputs.append(est_mu_t)
            
            # Calculate gradient for non-spammers
            non_spam_grad_1 = self.approx_grad_1(theta_t, self.mu_0, self.Sigma_0, 0)
            
            # Calculate gradient for spammers
            # First, estimate dmu1* / dth
            try:
                st_mu_1_grad = self.finite_diff_approx(inputs, outputs)
                lt_dmu_est   = self.approx_lt_dmu(st_mu_1_grad, k)
            except RuntimeError:
                print(f'Jacobian estimation failed (lr={round(lr,4)}, H={horizon}, k={k}, pert={round(pert,4)})')
                break
            
            spam_grad_1 = self.approx_grad_1(theta_t, mu_t, self.Sigma_1, 1)
            spam_grad_2 = self.approx_grad_2(theta_t, mu_t, self.Sigma_1, 1, lt_dmu_est)
            
            grad = (1 - self.spammer_weight) * non_spam_grad_1 + self.spammer_weight * (spam_grad_1 + spam_grad_2)
            
            with torch.no_grad():
                theta_t    = torch.clamp(theta_t - lr * (grad + pert * torch.randn(self.d)), -self.R, self.R)
                mu_tm1     = mu_t
                est_mu_tm1 = est_mu_t
                params_t   = torch.cat([theta_t, est_mu_tm1])
                
                # For testing
                grad2s.append(torch.linalg.norm(spam_grad_2))
        
        while len(thetas) < T:
            thetas.append(self.R * torch.ones(self.d))
        
        return thetas#, grad2s
    
    
    ###########################################################################
    # Regular PerfGD methods
    ###########################################################################
    def PGD(self, T, lr, k, horizon):
        theta_t = copy.deepcopy(self.init_th)
        mu_t    = copy.deepcopy(self.mu_1)
        
        thetas = deque(maxlen = horizon)
        mus    = deque(maxlen = horizon)
        
        thetas_hist = deque()
        
        with torch.no_grad():
            for t in range(k * (self.d - 1)):
                thetas_hist.append(theta_t.clone())
                
                mu_t = self.st_mu_1(theta_t, mu_t)
                
                if t % k == k - 1: # Only updated theta every k steps
                    mus.append(mu_t.clone() + self.mean_noise * torch.randn(self.d))
                    thetas.append(theta_t.clone())
                    theta_t += lr * torch.randn(self.d)
        
        
        for t in range(T - k * (self.d - 1)):
            theta_t.requires_grad = True
            thetas_hist.append(theta_t.clone().detach())
            
            mu_t = self.st_mu_1(theta_t, mu_t).detach()
            
            if t % k == k - 1: # Only updated theta every k steps
                mus.append(mu_t.clone() + self.mean_noise * torch.randn(self.d))  
                thetas.append(theta_t.clone())
                
                # Estimate gradient for non-spammers
                non_spam_grad_1 = self.approx_grad_1(theta_t, self.mu_0, self.Sigma_0, 0)
                
                # Estimate gradient for spammers
                try:
                    lt_dmu_est = self.finite_diff_approx(thetas, mus)
                except RuntimeError:
                    print(f'Jacobian estimation failed (lr={round(lr,4)}, H={horizon}, wait={k}')
                    break
                
                spam_grad_1 = self.approx_grad_1(theta_t, mu_t, self.Sigma_1, 1)
                spam_grad_2 = self.approx_grad_2(theta_t, mu_t, self.Sigma_1, 1, lt_dmu_est)
                
                # Combine gradients
                grad = (1 - self.spammer_weight) * non_spam_grad_1 + self.spammer_weight * (spam_grad_1 + spam_grad_2)
                
                with torch.no_grad():
                    theta_t  = torch.clamp(theta_t - lr * grad, -self.R, self.R)
        
        while len(thetas_hist) < T:
            thetas_hist.append(self.R * torch.ones(self.d))
                
        return thetas_hist
    
    
    ###########################################################################
    # RGD methods
    ###########################################################################
    def RGD(self, T, lr, k):
        theta_t  = copy.deepcopy(self.init_th)
        mu_t     = copy.deepcopy(self.mu_1)
        
        thetas   = deque()
        
        for t in range(T):
            theta_t.requires_grad = True
            thetas.append(theta_t.clone().detach())
            
            mu_t = self.st_mu_1(theta_t, mu_t).detach()
            
            if t % k == k - 1:
                # Calculate gradient for non-spammers
                non_spam_grad_1 = self.approx_grad_1(theta_t, self.mu_0, self.Sigma_0, 0)
                
                # Calculate gradient for spammers
                spam_grad_1 = self.approx_grad_1(theta_t, mu_t, self.Sigma_1, 1)
                
                grad = (1 - self.spammer_weight) * non_spam_grad_1 + self.spammer_weight * spam_grad_1
            
                with torch.no_grad():
                    theta_t  = torch.clamp(theta_t - lr * grad, -self.R, self.R)
                
        return thetas
    
    ###########################################################################
    # DFO methods
    ###########################################################################
    def approx_st_loss(self, th, spam_mean, n = 1000):
        n_spam = int(self.spammer_weight * n)
        
        non_spam_X = torch.tensor(np.random.multivariate_normal(self.mu_0, self.Sigma_0, n - n_spam)).float()
        non_spam_Y = torch.zeros(n - n_spam)
        
        spam_X = torch.tensor(np.random.multivariate_normal(spam_mean, self.Sigma_1, n_spam)).float()
        spam_Y = torch.ones(n_spam)
        
        X = torch.cat([non_spam_X, spam_X]).float()
        Y = torch.cat([non_spam_Y, spam_Y])
        
        H  = 1. / (1. + torch.exp(-X @ th))
        Ls = -(Y * torch.log(H) + (1 - Y) * torch.log(1. - H)) + (self.reg / 2) * (torch.linalg.norm(th) ** 2)
        
        return torch.sum(Ls) / len(Ls)
    
    
    def DFO(self, T, lr, perturbation, k):
        thetas  = deque()
        queries = deque()
        
        with torch.no_grad():
            internal_t = copy.deepcopy(self.init_th)
            mu_t       = copy.deepcopy(self.mu_1)
            
            u_t      = torch.randn(self.d)
            u_t     /= torch.linalg.norm(u_t)
            deploy_t = (internal_t + perturbation * u_t).clone()
        
            for t in range(T):
                thetas.append(internal_t.clone())
                queries.append(deploy_t.clone())
                
                mu_t = self.st_mu_1(deploy_t, mu_t)
                
                if t % k == k - 1: # Only updated theta every k steps
                    loss = self.approx_st_loss(deploy_t, mu_t)
                    grad = (self.d * loss / perturbation) * u_t
                    
                    internal_t = torch.clamp(internal_t - lr * grad, -self.R, self.R).clone()
                    
                    u_t      = torch.randn(self.d)
                    u_t     /= torch.linalg.norm(u_t)
                    deploy_t = torch.clamp(internal_t + perturbation * u_t, -self.R, self.R).clone()
                    
        return thetas, queries


###############################################################################
# Experiments
###############################################################################
# Final values from grid search:
# delta0 = 0.25
# mean_noise = 0.001
# RGD: lr = 0.02, wait = 1
# DFO: lr = 0.05, wait = 1, pert = 1.
# PGD: lr = 0.1,  wait = 5, H = 2
# SPGD: lr = 0.1, pert = 0.1, H = None, k = None

# Problem instance constants
mu_0 = torch.tensor([2., 1.])
mu_1 = torch.tensor([1., 2.])
d = 2
spammer_weight = 0.5
R = 3.
alpha = -2.
reg = 1e-1
mean_noise = 0.001
delta0 = 0.25


num_trials = 50
T = 40

rgd_lr = 0.02

dfo_lr = 0.05
dfo_wait = 1
dfo_pert = 1.

pgd_lr = 0.1
pgd_wait = 5
pgd_H = 2

spgd_lr = 0.1
spgd_pert = 0.1
spgd_H = None
spgd_k = None


rgd_l  = np.zeros((num_trials, T))
dfo_i_l  = np.zeros((num_trials, T))
dfo_q_l  = np.zeros((num_trials, T))
pgd_l  = np.zeros((num_trials, T))
spgd_l = np.zeros((num_trials, T))

for r in tqdm(range(num_trials)):
    expt = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, init_th = None, seed = 67890 + r)
    
    rgd_th    = expt.RGD(T, rgd_lr, 1)
    rgd_l[r]  = np.array([expt.approx_lt_loss(th, 10000) for th in rgd_th])
    
    dfo_i_th, dfo_q_th = expt.DFO(T, dfo_lr, dfo_pert, dfo_wait)
    dfo_i_l[r] = np.array([expt.approx_lt_loss(th, 10000) for th in dfo_i_th])
    dfo_q_l[r] = np.array([expt.approx_lt_loss(th, 10000) for th in dfo_q_th])
    
    pgd_th    = np.array(expt.PGD(T, pgd_lr, pgd_wait, pgd_H))
    pgd_l[r]  = np.array([expt.approx_lt_loss(th, 10000) for th in pgd_th])
    
    spgd_th   = np.array(expt.SPGD(T, spgd_lr, spgd_k, spgd_H, spgd_pert))
    spgd_l[r] = np.array([expt.approx_lt_loss(th, 10000) for th in spgd_th])


rgd_m = np.mean(rgd_l, axis = 0)
dfo_i_m = np.mean(dfo_i_l, axis = 0)
dfo_q_m = np.mean(dfo_q_l, axis = 0)
pgd_m = np.mean(pgd_l, axis = 0)
spgd_m = np.mean(spgd_l, axis = 0)

rgd_e = sem(rgd_l, axis = 0)
dfo_i_e = sem(dfo_i_l, axis = 0)
dfo_q_e = sem(dfo_q_l, axis = 0)
pgd_e = sem(pgd_l, axis = 0)
spgd_e = sem(spgd_l, axis = 0)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()

plt.plot(range(T), rgd_m, label = 'RGD', color = colors[5])
plt.fill_between(range(T), rgd_m + rgd_e, rgd_m - rgd_e, color = colors[5], alpha = 0.3)

plt.plot(range(T), dfo_i_m, label = 'DFO (i)', color = colors[7])
plt.fill_between(range(T), dfo_i_m + dfo_i_e, dfo_i_m - dfo_i_e, color = colors[7], alpha = 0.3)

plt.plot(range(T), dfo_q_m, label = 'DFO (q)', color = colors[3])
plt.fill_between(range(T), dfo_q_m + dfo_q_e, dfo_q_m - dfo_q_e, color = colors[3], alpha = 0.3)

plt.plot(range(T), pgd_m, label = 'PGD', color = colors[4])
plt.fill_between(range(T), pgd_m + pgd_e, pgd_m - pgd_e, color = colors[4], alpha = 0.3)

plt.plot(range(T), spgd_m, label = 'SPGD', color = colors[2])
plt.fill_between(range(T), spgd_m + spgd_e, spgd_m - spgd_e, color = colors[2], alpha = 0.3)


plt.xlabel('Training iteration')
plt.ylabel('Long-term loss')
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(linewidth)
leg.set_draggable(state=True)






# ##################
# # Grid searching #
# ##################
# def experiment(T, delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, num_trials, lrs, dfo_perts, spgd_perts, Hs, waits, ks, init_th = None):    
#     # For this problem instance, we have opt = [-0.1961,  0.2205]
#     # opt = torch.tensor([-0.1961,  0.2205])
    
#     rgd_th   = np.empty((len(lrs), len(waits), num_trials), dtype=object)
#     dfo_i_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     dfo_q_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     pgd_th   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
#     spgd_th  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
#     rgd_l   = np.empty((len(lrs), len(waits), num_trials), dtype=object)
#     dfo_i_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     dfo_q_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     pgd_l   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
#     spgd_l  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
#     # rgd_d   = np.empty((len(lrs), len(waits), num_trials), dtype=object)
#     # dfo_i_d = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     # dfo_q_d = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
#     # pgd_d   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
#     # spgd_d  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
#     # dummy_expt = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1)
#     # opt    = dummy_expt.compute_opt(100, 0.1, 10000).detach()
#     # stab   = dummy_expt.compute_stab(100, 0.1, 10000).detach()
    
#     for r in tqdm(range(num_trials)):
#         # expt = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, init_th = (opt + stab) / 2, seed = r + 12345)
#         expt = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, init_th = init_th, seed = r + 12345)
        
#         # RGD experiments
#         print('Running RGD...')
#         for i in range(len(lrs)):
#             for j in range(len(waits)):
#                 lr   = lrs[i]
#                 wait = 1
#                 # wait = waits[j]
#                 # T    = int(10 / lr)
                
                
#                 traj = expt.RGD(T, lr, wait)
#                 rgd_l[i, j, r]  = np.array([expt.approx_lt_loss(th, 100) for th in traj], dtype=float)
#                 rgd_th[i, j, r] = np.array(traj)
#                 # rgd_d[i, j, r]  = np.array([torch.linalg.norm(th - opt) for th in traj], dtype=float)
        
#         # DFO experiments
#         print('Running DFO...')
#         for i in range(len(lrs)):
#             for j in range(len(dfo_perts)):
#                 for k in range(len(waits)):
#                     lr   = lrs[i]
#                     pert = dfo_perts[j]
#                     wait = waits[k]
#                     # T    = int(10 / lr)
                    
                    
#                     i_traj, q_traj = expt.DFO(T, lr, pert, wait)
#                     dfo_i_l[i, j, k, r]  = np.array([expt.approx_lt_loss(th, 100) for th in i_traj], dtype=float)
#                     dfo_q_l[i, j, k, r]  = np.array([expt.approx_lt_loss(th, 100) for th in q_traj], dtype=float)
#                     dfo_i_th[i, j, k, r] = np.array(i_traj)
#                     dfo_q_th[i, j, k, r] = np.array(q_traj)
#                     # dfo_i_d[i, j, k, r]  = np.array([torch.linalg.norm(th - opt) for th in traj], dtype=float)
#                     # dfo_q_d[i, j, k, r]  = np.array([torch.linalg.norm(th - opt) for th in traj], dtype=float)
        
#         # PGD experiments
#         print('Running PGD...')
#         for i in range(len(lrs)):
#             for j in range(len(Hs)):
#                 for k in range(len(waits)):
#                     lr   = lrs[i]
#                     H    = Hs[j]
#                     wait = waits[k]
#                     # T    = int(10 / lr)
                    
                    
#                     traj = expt.PGD(T, lr, wait, horizon = H)
#                     pgd_l[i, j, k, r]  = np.array([expt.approx_lt_loss(th, 100) for th in traj], dtype=float)
#                     pgd_th[i, j, k, r] = np.array(traj)
#                     # pgd_d[i, j, k, r]  = np.array([torch.linalg.norm(th - opt) for th in traj], dtype=float)
        
#         # SPGD experiments
#         print('Running SPGD...')
#         for i in range(len(lrs)):
#             for j in range(len(spgd_perts)):
#                 for l in range(len(Hs)):
#                     for m in range(len(ks)):
#                         lr   = lrs[i]
#                         pert = spgd_perts[j]
#                         H    = Hs[l]
#                         k    = ks[m]
#                         # T = int(10 / lr)
                        
                        
#                         traj = expt.SPGD(T, lr, k, H, pert)
#                         spgd_l[i, j, l, m, r]  = np.array([expt.approx_lt_loss(th, 100) for th in traj], dtype=float)
#                         spgd_th[i, j, l, m, r] = np.array(traj)
#                         # spgd_d[i, j, l, m, r]  = np.array([torch.linalg.norm(th - opt) for th in traj], dtype=float)
    
#     results = {'rgd_th': rgd_th, 'rgd_l': rgd_l, #'rgd_d': rgd_d,
#                'dfo_i_th': dfo_i_th, 'dfo_i_l': dfo_i_l, #'dfo_i_d': dfo_i_d,
#                'dfo_q_th': dfo_q_th, 'dfo_q_l': dfo_q_l, #'dfo_q_d': dfo_q_d,
#                'pgd_th': pgd_th, 'pgd_l': pgd_l, #'pgd_d': pgd_d,
#                'spgd_th': spgd_th, 'spgd_l': spgd_l} #, 'spgd_d': spgd_d}
#     return results


# d          = 2
# lrs        = [0.1, 0.05, 0.02]
# waits      = [1, 5, 10, 20]
# ks         = [None]
# dfo_perts  = [10 ** (-k/2) for k in range(4)]
# spgd_perts = [0] + [10 ** (-k) for k in range(2)]
# Hs         = [None] + [d + k for k in range(d + 1)]

# mu_0 = torch.tensor([2., 1.])
# mu_1 = torch.tensor([1., 2.])
# spammer_weight = 0.5
# T = 20
# R = 3.
# alpha = -2.
# reg = 1e-1
# num_trials = 10
# mean_noise = 0.001


# def make_plots(num_trials, delta0, mean_noise, mu_0, mu_1, T):
#     expt   = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1)
#     opt    = expt.compute_opt(100, 0.1, 10000).detach()
#     opt_l  = expt.approx_lt_loss(opt, 10000)
#     stab   = expt.compute_stab(100, 0.1, 10000).detach()
#     stab_l = expt.approx_lt_loss(stab, 10000)
    
#     results = experiment(T, delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1, num_trials, lrs, dfo_perts, spgd_perts, Hs, waits, ks, init_th = None)
    
    
#     ###############################################################################
#     # Find best hyperparams for each method
#     ###############################################################################
#     rgd_l   = results['rgd_l']
#     dfo_i_l = results['dfo_i_l']
#     dfo_q_l = results['dfo_q_l']
#     pgd_l   = results['pgd_l']
#     spgd_l  = results['spgd_l']
    
#     rgd_l_mean   = np.mean(rgd_l, axis = -1)
#     dfo_i_l_mean = np.mean(dfo_i_l, axis = -1)
#     dfo_q_l_mean = np.mean(dfo_q_l, axis = -1)
#     pgd_l_mean   = np.mean(pgd_l, axis = -1)
#     spgd_l_mean  = np.mean(spgd_l, axis = -1)
    
#     rgd_end   = np.zeros(rgd_l_mean.shape).flatten()
#     dfo_i_end = np.zeros(dfo_i_l_mean.shape).flatten()
#     dfo_q_end = np.zeros(dfo_q_l_mean.shape).flatten()
#     pgd_end   = np.zeros(pgd_l_mean.shape).flatten()
#     spgd_end  = np.zeros(spgd_l_mean.shape).flatten()
    
#     for i, loss_traj in enumerate(rgd_l_mean.flatten()):
#         T = len(loss_traj)
#         rgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
#     for i, loss_traj in enumerate(dfo_i_l_mean.flatten()):
#         T = len(loss_traj)
#         dfo_i_end[i] = np.mean(loss_traj[-int(T/10):])
    
#     for i, loss_traj in enumerate(pgd_l_mean.flatten()):
#         T = len(loss_traj)
#         pgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
#     for i, loss_traj in enumerate(spgd_l_mean.flatten()):
#         T = len(loss_traj)
#         spgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
#     rgd_best  = np.unravel_index(np.nanargmin(rgd_end), rgd_l_mean.shape)
#     dfo_best  = np.unravel_index(np.nanargmin(dfo_i_end), dfo_i_l_mean.shape)
#     pgd_best  = np.unravel_index(np.nanargmin(pgd_end), pgd_l_mean.shape)
#     spgd_best = np.unravel_index(np.nanargmin(spgd_end), spgd_l_mean.shape)
    
#     print('')
#     print(f'RGD:  lr={round(lrs[rgd_best[0]], 3)}, wait={waits[rgd_best[1]]}')
#     print(f'DFO:  lr={round(lrs[dfo_best[0]], 3)}, pert={round(dfo_perts[dfo_best[1]], 3)}, wait={waits[dfo_best[2]]}')
#     print(f'PGD:  lr={round(lrs[pgd_best[0]], 3)}, H={Hs[pgd_best[1]]}, wait={waits[pgd_best[2]]}')
#     print(f'SPGD: lr={round(lrs[spgd_best[0]], 3)}, pert={round(spgd_perts[spgd_best[1]], 3)}, H={Hs[spgd_best[2]]}, k={ks[spgd_best[3]]}')
    
#     ###############################################################################
#     # Make plots
#     ###############################################################################
    
#     best_rgd_l  = np.asfarray([x for x in rgd_l[rgd_best]])
#     best_dfo_l  = np.asfarray([x for x in dfo_i_l[dfo_best]])
#     best_dfo_q_l = np.asfarray([x for x in dfo_q_l[dfo_best]])
#     best_pgd_l  = np.asfarray([x for x in pgd_l[pgd_best]])
#     best_spgd_l = np.asfarray([x for x in spgd_l[spgd_best]])
    
#     rgd_l_m  = np.mean(best_rgd_l, axis=0)
#     dfo_l_m  = np.mean(best_dfo_l, axis=0)
#     dfo_q_l_m = np.mean(best_dfo_q_l, axis=0)
#     pgd_l_m  = np.mean(best_pgd_l, axis=0)
#     spgd_l_m = np.mean(best_spgd_l, axis=0)
    
#     rgd_l_sem  = sem(best_rgd_l, axis=0)
#     dfo_l_sem  = sem(best_dfo_l, axis=0)
#     dfo_q_l_sem = sem(best_dfo_q_l, axis=0)
#     pgd_l_sem  = sem(best_pgd_l, axis=0)
#     spgd_l_sem = sem(best_spgd_l, axis=0)
    
#     expt   = Classification(delta0, mean_noise, d, R, alpha, spammer_weight, reg, mu_0, mu_1)
#     opt    = expt.compute_opt(100, 0.1, 10000).detach()
#     opt_l  = expt.approx_lt_loss(opt, 10000)
#     stab   = expt.compute_stab(100, 0.1, 10000).detach()
#     stab_l = expt.approx_lt_loss(stab, 10000)
    
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
#     plt.figure()
    
#     plt.plot(rgd_l_m, label = f'RGD: lr={round(lrs[rgd_best[0]], 3)}, wait={waits[rgd_best[1]]}', color=colors[2])
#     plt.fill_between(range(T), rgd_l_m + rgd_l_sem, rgd_l_m - rgd_l_sem, color=colors[2], alpha=0.3)
    
#     plt.plot(dfo_l_m, label = f'DFO (i): lr={round(lrs[dfo_best[0]], 3)}, pert={round(dfo_perts[dfo_best[1]], 3)}, wait={waits[dfo_best[2]]}', color=colors[3])
#     plt.fill_between(range(T), dfo_l_m + dfo_l_sem, dfo_l_m - dfo_l_sem, color=colors[3], alpha=0.3)
    
#     plt.plot(dfo_q_l_m, label = f'DFO (q)', color=colors[6])
#     plt.fill_between(range(T), dfo_q_l_m + dfo_q_l_sem, dfo_q_l_m - dfo_q_l_sem, color=colors[6], alpha=0.3)
    
#     plt.plot(pgd_l_m, label = f'PGD: lr={round(lrs[pgd_best[0]], 3)}, H={Hs[pgd_best[1]]}, wait={waits[pgd_best[2]]}', color=colors[4])
#     plt.fill_between(range(T), pgd_l_m + pgd_l_sem, pgd_l_m - pgd_l_sem, color=colors[4], alpha=0.3)
    
#     plt.plot(spgd_l_m, label = f'SPGD: lr={round(lrs[spgd_best[0]], 3)}, pert={round(spgd_perts[spgd_best[1]], 3)}, H={Hs[spgd_best[2]]}, k={ks[spgd_best[3]]}', color=colors[5])
#     plt.fill_between(range(T), spgd_l_m + spgd_l_sem, spgd_l_m - spgd_l_sem, color=colors[5], alpha=0.3)
    
#     plt.title(f'δ0={round(delta0, 3)}, mn={mean_noise}')
#     plt.xlabel('Training iteration')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     if not os.path.isdir('plots/classification'):
#         os.mkdir('plots/classification')
        
#     plt.savefig(f'plots/classification/{round(delta0, 3)}_{mean_noise}.pdf')
    
#     return rgd_best, best_rgd_l, dfo_best, best_dfo_l, pgd_best, best_pgd_l, spgd_best, best_spgd_l




# delta0s = [0.25]

# rgds    = [None for _ in range(len(delta0s))]
# rgd_ls  = [None for _ in range(len(delta0s))]
# dfos    = [None for _ in range(len(delta0s))]
# dfo_ls  = [None for _ in range(len(delta0s))]
# pgds    = [None for _ in range(len(delta0s))]
# pgd_ls  = [None for _ in range(len(delta0s))]
# spgds   = [None for _ in range(len(delta0s))]
# spgd_ls = [None for _ in range(len(delta0s))]

# for i in tqdm(range(len(delta0s))):
#     delta0     = delta0s[i]    
#     rgds[i], rgd_ls[i], dfos[i], dfo_ls[i], pgds[i], pgd_ls[i], spgds[i], spgd_ls[i] = make_plots(num_trials, delta0, mean_noise, mu_0, mu_1, T)

# rgd_end = np.mean(rgd_ls[0][:, -5:])
# dfo_end = np.mean(dfo_ls[0][:, -5:])
# pgd_end = np.mean(pgd_ls[0][:, -5:])
# spgd_end = np.mean(spgd_ls[0][:, -5:])

# print(f'RGD: {rgd_end}')
# print(f'DFO: {dfo_end}')
# print(f'PGD: {pgd_end}')
# print(f'SPGD: {spgd_end}')
    







