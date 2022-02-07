import torch
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_spd_matrix
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

def L(th, mu):
    return -torch.dot(th, mu)


class ToyLinear2:
    def __init__(self, d, R, delta0, mean_noise, A = None, b = None, seed = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.d = d
        self.R = R
        self.delta0 = delta0
        self.mean_noise = mean_noise
        self.seed = seed
        
        if A is None:
            # self.A = -0.8 * torch.eye(d)
            self.A = -0.8 * torch.tensor(make_spd_matrix(d, random_state = 0)).float()
        else:
            self.A = A
        
        if b is None:
            self.b = 2 * torch.ones(d)
        else:
            self.b = b
        
        # A should be symmetric and negative definite
        self.opt_th = torch.linalg.solve(-2 * self.A, self.b)
        self.opt_l  = self.lt_loss(self.opt_th)
        
        self.stab_th = torch.linalg.solve(-self.A, self.b)
        self.stab_l  = self.lt_loss(self.stab_th)
        
        self.init_th = (self.opt_th + self.stab_th) / 2 + torch.randn(d)
        self.init_mu = torch.randn(d)


    ##############################################################################
    # Helper methods
    ##############################################################################
    def mu_star(self, th):
        return self.A @ th + self.b
    
    
    def lt_mu(self, th):
        return (self.delta0 / (2 - self.delta0)) * self.mu_star(th)
    
    
    def m(self, th, mu):
        return self.delta0 * self.mu_star(th) - (1 - self.delta0) * mu
    
    
    def lt_loss(self, th):
        return L(th, self.lt_mu(th))
    
    ##############################################################################
    
    
    ##############################################################################
    # PGD
    ##############################################################################
    def PGD(self, T, lr, k, horizon):        
        theta_t  = copy.deepcopy(self.init_th)
        mu_t     = copy.deepcopy(self.init_mu)
        est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
        
        thetas = deque(maxlen = horizon)
        mus    = deque(maxlen = horizon)
        
        thetas_hist = deque()
        
        # warmup = self.d
        warmup = 4
        with torch.no_grad():
            for t in range(k * warmup):
                thetas_hist.append(theta_t.clone())
                
                if t % k == k - 1: # Only updated theta every k steps
                    mus.append(est_mu_t.clone())
                    thetas.append(theta_t.clone())
                    
                    theta_t = torch.clamp(theta_t + lr * torch.randn(self.d), -self.R, self.R)
                    # grad = -est_mu_t
                    # theta_t -= lr * grad
                    
                mu_t = self.m(theta_t, mu_t)
                est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
            
            for t in range(T - k * warmup):
                thetas_hist.append(theta_t.clone())
                
                if t % k == k - 1: # Only updated theta every k steps
                    est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
                    mus.append(est_mu_t.clone())
                    thetas.append(theta_t.clone())
                
                    Delta_thetas = torch.column_stack([th - thetas[-1] for th in thetas])
                    Delta_mus    = torch.column_stack([mu - mus[-1] for mu in mus])
                    
                    try:
                        dmu_dth          = (Delta_mus @ torch.linalg.pinv(Delta_thetas))
                        perf_grad_approx = -(est_mu_t + dmu_dth.T @ theta_t)
                    
                    except RuntimeError:
                        # print(f'Jacobian estimation failed (lr={round(lr,4)}, H={horizon}, k={k}')
                        break
                    
                    theta_t = torch.clamp(theta_t - lr * perf_grad_approx, -self.R, self.R)
                    
                mu_t = self.m(theta_t, mu_t)
        
        while len(thetas_hist) < T:
            thetas_hist.append(self.R * torch.ones(self.d))
                
        return thetas_hist
    ##############################################################################
    
    
    ##############################################################################
    # SPGD
    ##############################################################################
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
    
    
    def SPGD(self, T, lr, k, H, pert):
        theta_t    = copy.deepcopy(self.init_th)
        mu_tm1     = copy.deepcopy(self.init_mu)
        est_mu_tm1 = mu_tm1 + self.mean_noise * torch.randn(self.d)
        params_t   = torch.cat([theta_t, mu_tm1])
        
        thetas   = deque()
        inputs   = deque(maxlen = H)
        outputs  = deque(maxlen = H)
        
        warmup = self.d
        with torch.no_grad():
            for t in range(warmup):
                thetas.append(theta_t.clone())
                inputs.append(params_t.clone())
                
                mu_t     = self.m(theta_t, mu_tm1)
                est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
                
                outputs.append(est_mu_t)
                
                theta_t    = torch.clamp(theta_t + lr * torch.randn(self.d), -self.R, self.R)
                mu_tm1     = mu_t
                est_mu_tm1 = est_mu_t
                params_t   = torch.cat([theta_t, est_mu_tm1])
            
            for t in range(T - warmup):
                thetas.append(theta_t.clone())
                inputs.append(params_t.clone())
                
                mu_t     = self.m(theta_t, mu_tm1)
                est_mu_t = mu_t + self.mean_noise * torch.randn(self.d)
                outputs.append(est_mu_t)
                
                Delta_inputs  = torch.column_stack([i - inputs[-1] for i in inputs])
                Delta_outputs = torch.column_stack([o - outputs[-1] for o in outputs])
                
                try:
                    grad_m                = (Delta_outputs @ torch.linalg.pinv(Delta_inputs))
                    long_term_grad_approx = -(est_mu_t + self.approx_lt_dmu(grad_m, k).T @ theta_t + pert * torch.randn(self.d))
                
                except RuntimeError:
                    # print(f'Jacobian estimation failed (lr={round(lr,4)}, H={H}, k={k}, pert={round(pert,4)})')
                    break
                
                theta_t    = torch.clamp(theta_t - lr * long_term_grad_approx, -self.R, self.R)
                mu_tm1     = mu_t
                est_mu_tm1 = est_mu_t
                params_t   = torch.cat([theta_t, est_mu_tm1])
        
        while len(thetas) < T:
            thetas.append(self.R * torch.ones(self.d))
                
        return thetas
    
    
    ##############################################################################
    # RGD
    ##############################################################################
    def RGD(self, T, lr, k = 1):
        theta_t = copy.deepcopy(self.init_th)
        mu_t    = copy.deepcopy(self.init_mu)
        thetas  = deque()
        
        with torch.no_grad():
            for t in range(T):
                thetas.append(theta_t.clone().detach())
                
                grad = -(mu_t + self.mean_noise * torch.randn(self.d))
                theta_t = torch.clamp(theta_t - lr * grad, -self.R, self.R)
                
                mu_t = self.m(theta_t, mu_t)
                
        return thetas
    
    
    ##############################################################################
    # Flaxman black-box DFO
    ##############################################################################
    def DFO(self, T, lr, perturbation, k = 1):
        thetas  = deque()
        queries = deque()
        
        with torch.no_grad():
            internal_t = copy.deepcopy(self.init_th)
            mu_t       = copy.deepcopy(self.init_mu)
            
            u_t      = torch.randn(self.d)
            u_t     /= torch.linalg.norm(u_t)
            deploy_t = torch.clamp(internal_t + perturbation * u_t, -self.R, self.R).clone()
        
            for t in range(T):
                thetas.append(internal_t.clone().detach())
                queries.append(deploy_t.clone().detach())
                
                if t % k == 0: # Only updated theta every k steps
                    loss = -torch.dot(deploy_t, mu_t + self.mean_noise * torch.randn(self.d))
                    grad = (self.d * loss / perturbation) * u_t
                    
                    internal_t = torch.clamp(internal_t - lr * grad, -self.R, self.R)
                    
                    u_t      = torch.randn(self.d)
                    u_t     /= torch.linalg.norm(u_t)
                    deploy_t = torch.clamp(internal_t + perturbation * u_t, -self.R, self.R).clone()
                
                mu_t = self.m(deploy_t, mu_t).clone()
        
        return thetas, queries

##############################################################################
# Run experiments
##############################################################################
def experiment(num_trials, T, lrs, dfo_perts, Hs, waits, ks, spgd_perts, d, R, delta0, mean_noise, A = None, b = None, seed = 0):    
    
    rgd_th   = np.empty((len(lrs), num_trials), dtype=object)
    dfo_i_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    dfo_q_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    pgd_th   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
    spgd_th  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
    rgd_l   = np.empty((len(lrs), num_trials), dtype=object)
    dfo_i_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    dfo_q_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    pgd_l   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
    spgd_l  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
    rgd_d   = np.empty((len(lrs), num_trials), dtype=object)
    dfo_i_d = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    dfo_q_d = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    pgd_d   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
    spgd_d  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
    
    for r in tqdm(range(num_trials)):
        expt = ToyLinear2(d, R, delta0, mean_noise, A, b, seed + r)
        
        # RGD experiments
        print('Running RGD...')
        for i in range(len(lrs)):
            # lr   = lrs[i]
            lr   = 0.4
            wait = 1
            
            traj = expt.RGD(T, lr)
            rgd_l[i, r]  = np.array([expt.lt_loss(th) for th in traj], dtype=float)
            rgd_th[i, r] = np.array(traj)
            rgd_d[i, r]  = np.array([torch.linalg.norm(th - expt.opt_th) for th in traj], dtype=float)
        
        # DFO experiments
        print('Running DFO...')
        for i in range(len(lrs)):
            for j in range(len(dfo_perts)):
                for k in range(len(waits)):
                    lr   = lrs[i]
                    pert = dfo_perts[j]
                    wait = waits[k]
                    # T    = int(10 / lr)
                    
                    
                    i_traj, q_traj = expt.DFO(T, lr, pert, wait)
                    dfo_i_l[i, j, k, r]  = np.array([expt.lt_loss(th) for th in i_traj], dtype=float)
                    dfo_q_l[i, j, k, r]  = np.array([expt.lt_loss(th) for th in q_traj], dtype=float)
                    dfo_i_th[i, j, k, r] = np.array(i_traj)
                    dfo_q_th[i, j, k, r] = np.array(q_traj)
                    dfo_i_d[i, j, k, r]  = np.array([torch.linalg.norm(th - expt.opt_th) for th in traj], dtype=float)
                    dfo_q_d[i, j, k, r]  = np.array([torch.linalg.norm(th - expt.opt_th) for th in traj], dtype=float)
        
        # PGD experiments
        print('Running PGD...')
        for i in range(len(lrs)):
            for j in range(len(Hs)):
                for k in range(len(waits)):
                    lr   = lrs[i]
                    H    = Hs[j]
                    wait = waits[k]
                    # T    = int(10 / lr)
                    
                    
                    traj = expt.PGD(T, lr, wait, horizon = H)
                    pgd_l[i, j, k, r]  = np.array([expt.lt_loss(th) for th in traj], dtype=float)
                    pgd_th[i, j, k, r] = np.array(traj)
                    pgd_d[i, j, k, r]  = np.array([torch.linalg.norm(th - expt.opt_th) for th in traj], dtype=float)
        
        # SPGD experiments
        print('Running SPGD...')
        for i in range(len(lrs)):
            for j in range(len(spgd_perts)):
                for l in range(len(Hs)):
                    for m in range(len(ks)):
                        lr   = lrs[i]
                        pert = spgd_perts[j]
                        H    = Hs[l]
                        k    = ks[m]
                        # T = int(10 / lr)
                        
                        traj = expt.SPGD(T, lr, k, H, pert)
                        spgd_l[i, j, l, m, r]  = np.array([expt.lt_loss(th) for th in traj], dtype=float)
                        spgd_th[i, j, l, m, r] = np.array(traj)
                        spgd_d[i, j, l, m, r]  = np.array([torch.linalg.norm(th - expt.opt_th) for th in traj], dtype=float)
    
    results = {'rgd_th': rgd_th, 'rgd_l': rgd_l, 'rgd_d': rgd_d,
               'dfo_i_th': dfo_i_th, 'dfo_i_l': dfo_i_l, 'dfo_i_d': dfo_i_d,
               'dfo_q_th': dfo_q_th, 'dfo_q_l': dfo_q_l, 'dfo_q_d': dfo_q_d,
               'pgd_th': pgd_th, 'pgd_l': pgd_l, 'pgd_d': pgd_d,
               'spgd_th': spgd_th, 'spgd_l': spgd_l, 'spgd_d': spgd_d}
    return results


# d          = 5
d = 2
R          = 5.
lrs        = [10 ** (-k/2) for k in range(1, 7)]
waits      = [1, 5, 10, 20]
# ks         = [1, 10, None]
ks = [None]
dfo_perts  = [10 ** (-k/2) for k in range(4)]
spgd_perts = [0] + [10 ** (-k/2) for k in range(4)]
Hs         = [None] + [d + k for k in range(d + 1)]
spgd_Hs = [None] + [2 * d + k for k in range(d + 1)]


def make_plots(delta0, mean_noise, T):
    d = 5
    R = 10.
    num_trials = 5
    
    results = experiment(num_trials, T, lrs, dfo_perts, Hs, waits, ks, spgd_perts, d, R, delta0, mean_noise)
    
    ###############################################################################
    # Find best hyperparams for each method
    ###############################################################################
    rgd_l   = results['rgd_l']
    dfo_i_l = results['dfo_i_l']
    dfo_q_l = results['dfo_q_l']
    pgd_l   = results['pgd_l']
    spgd_l  = results['spgd_l']
    
    rgd_l_mean   = np.mean(rgd_l, axis = -1)
    dfo_i_l_mean = np.mean(dfo_i_l, axis = -1)
    dfo_q_l_mean = np.mean(dfo_q_l, axis = -1)
    pgd_l_mean   = np.mean(pgd_l, axis = -1)
    spgd_l_mean  = np.mean(spgd_l, axis = -1)
    
    rgd_end   = np.zeros(rgd_l_mean.shape).flatten()
    dfo_i_end = np.zeros(dfo_i_l_mean.shape).flatten()
    dfo_q_end = np.zeros(dfo_q_l_mean.shape).flatten()
    pgd_end   = np.zeros(pgd_l_mean.shape).flatten()
    spgd_end  = np.zeros(spgd_l_mean.shape).flatten()
    
    for i, loss_traj in enumerate(rgd_l_mean.flatten()):
        T = len(loss_traj)
        rgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
    for i, loss_traj in enumerate(dfo_i_l_mean.flatten()):
        T = len(loss_traj)
        dfo_i_end[i] = np.mean(loss_traj[-int(T/10):])
    
    for i, loss_traj in enumerate(pgd_l_mean.flatten()):
        T = len(loss_traj)
        pgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
    for i, loss_traj in enumerate(spgd_l_mean.flatten()):
        T = len(loss_traj)
        spgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
    rgd_best  = np.unravel_index(np.nanargmin(rgd_end), rgd_l_mean.shape)
    dfo_best  = np.unravel_index(np.nanargmin(dfo_i_end), dfo_i_l_mean.shape)
    pgd_best  = np.unravel_index(np.nanargmin(pgd_end), pgd_l_mean.shape)
    spgd_best = np.unravel_index(np.nanargmin(spgd_end), spgd_l_mean.shape)
    
    print('')
    print(f'RGD:  lr={round(lrs[rgd_best[0]], 3)}')
    print(f'DFO:  lr={round(lrs[dfo_best[0]], 3)}, pert={round(dfo_perts[dfo_best[1]], 3)}, wait={waits[dfo_best[2]]}')
    print(f'PGD:  lr={round(lrs[pgd_best[0]], 3)}, H={Hs[pgd_best[1]]}, wait={waits[pgd_best[2]]}')
    print(f'SPGD: lr={round(lrs[spgd_best[0]], 3)}, pert={round(spgd_perts[spgd_best[1]], 3)}, H={Hs[spgd_best[2]]}, k={ks[spgd_best[3]]}')
    
    ###############################################################################
    # Make plots
    ###############################################################################
    expt   = ToyLinear2(d, R, delta0, mean_noise)
    opt_l  = expt.opt_l
    stab_l = expt.stab_l
    
    best_rgd_l  = np.asfarray([x for x in rgd_l[rgd_best]])
    best_dfo_l  = np.asfarray([x for x in dfo_i_l[dfo_best]])
    best_pgd_l  = np.asfarray([x for x in pgd_l[pgd_best]])
    best_spgd_l = np.asfarray([x for x in spgd_l[spgd_best]])
    
    rgd_l_m  = np.mean(best_rgd_l, axis=0)
    dfo_l_m  = np.mean(best_dfo_l, axis=0)
    pgd_l_m  = np.mean(best_pgd_l, axis=0)
    spgd_l_m = np.mean(best_spgd_l, axis=0)
    
    rgd_l_sem  = sem(best_rgd_l, axis=0, nan_policy='propagate')
    dfo_l_sem  = sem(best_dfo_l, axis=0)
    pgd_l_sem  = sem(best_pgd_l, axis=0)
    spgd_l_sem = sem(best_spgd_l, axis=0)
    
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    plt.plot(opt_l * np.ones(T), label = 'OPT', color = colors[0])
    plt.plot(stab_l * np.ones(T), label = 'STAB', color = colors[1])
    
    plt.plot(rgd_l_m, label = 'RGD', color=colors[5])
    plt.fill_between(range(T), rgd_l_m + rgd_l_sem, rgd_l_m - rgd_l_sem, color=colors[5], alpha=0.3)
    
    plt.plot(dfo_l_m, label = 'DFO', color=colors[7])
    plt.fill_between(range(T), dfo_l_m + dfo_l_sem, dfo_l_m - dfo_l_sem, color=colors[7], alpha=0.3)
    
    plt.plot(pgd_l_m, label = 'PGD', color=colors[4])
    plt.fill_between(range(T), pgd_l_m + pgd_l_sem, pgd_l_m - pgd_l_sem, color=colors[4], alpha=0.3)
    
    plt.plot(spgd_l_m, label = 'SPGD', color=colors[2])
    plt.fill_between(range(T), spgd_l_m + spgd_l_sem, spgd_l_m - spgd_l_sem, color=colors[2], alpha=0.3)
    
    # plt.title(f'δ0={round(delta0, 3)}, mn={mean_noise}')
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    # plt.ylim((opt_l - 0.5, stab_l + 0.5))
    plt.legend()
        
    # plt.savefig(f'plots/toy_linear/{round(delta0, 3)}_{mean_noise}_{T}.pdf')
    # plt.savefig(f'plots/toy_linear/final/{round(delta0, 3)}_{mean_noise}.pdf')
    
    return rgd_best, best_rgd_l, dfo_best, best_dfo_l, pgd_best, best_pgd_l, spgd_best, best_spgd_l
    # return spgd_best



# delta0s = [1 - (0.01) ** (1 / k) for k in range(1, 17)]
# delta0s = [1 - (0.01) ** (2 ** -k) for k in range(7)]
# settle_steps = [2 ** k for k in range(7)]
# settle_steps = [128, 512, 2048]
settle_steps = [32]
delta0s = [1 - (0.01) ** (1 / k) for k in settle_steps]


rgds    = [None for _ in range(len(delta0s))]
rgd_ls  = [None for _ in range(len(delta0s))]
dfos    = [None for _ in range(len(delta0s))]
dfo_ls  = [None for _ in range(len(delta0s))]
pgds    = [None for _ in range(len(delta0s))]
pgd_ls  = [None for _ in range(len(delta0s))]
spgds   = [None for _ in range(len(delta0s))]
spgd_ls = [None for _ in range(len(delta0s))]

for i in tqdm(range(len(delta0s))):
    delta0     = delta0s[i]
    T          = 50
    mean_noise = 0.001
    # mean_noise = 0.
    
    rgds[i], rgd_ls[i], dfos[i], dfo_ls[i], pgds[i], pgd_ls[i], spgds[i], spgd_ls[i] = make_plots(delta0, mean_noise, T)
    # spgds[i] = make_plots(delta0, mean_noise, T)

rgd_end_m    = np.zeros(len(delta0s))
rgd_end_sem  = np.zeros(len(delta0s))
dfo_end_m    = np.zeros(len(delta0s))
dfo_end_sem  = np.zeros(len(delta0s))
pgd_end_m    = np.zeros(len(delta0s))
pgd_end_sem  = np.zeros(len(delta0s))
spgd_end_m   = np.zeros(len(delta0s))
spgd_end_sem = np.zeros(len(delta0s))
for i in range(len(delta0s)):
    rgd_end_m[i]   = np.mean(rgd_ls[i][:, -5:])
    rgd_end_sem[i] = sem(rgd_ls[i][:, -5:], axis=None)
    
    dfo_end_m[i]   = np.mean(dfo_ls[i][:, -5:])
    dfo_end_sem[i] = sem(dfo_ls[i][:, -5:], axis=None)
    
    pgd_end_m[i]   = np.mean(pgd_ls[i][:, -5:])
    pgd_end_sem[i] = sem(pgd_ls[i][:, -5:], axis=None)
    
    spgd_end_m[i]   = np.mean(spgd_ls[i][:, -5:])
    spgd_end_sem[i] = sem(spgd_ls[i][:, -5:], axis=None)
    

expt   = ToyLinear2(d, R, delta0, mean_noise)
opt_l  = expt.opt_l
stab_l = expt.stab_l

# xs = [2 ** k for k in range(7)]
xs = settle_steps
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure()
plt.plot(xs, [1. for _ in range(len(delta0s))], label = 'OPT', color = colors[0])
plt.plot(xs, [0. for _ in range(len(delta0s))], label = 'STAB', color = colors[1])


plt.plot(xs, rgd_end_m / opt_l, label = 'RGD', color = colors[5])
plt.fill_between(xs, (rgd_end_m + rgd_end_sem) / opt_l, (rgd_end_m - rgd_end_sem) / opt_l, color = colors[5], alpha = 0.3)

plt.plot(xs, dfo_end_m / opt_l, label = 'DFO', color = colors[7])
plt.fill_between(xs, (dfo_end_m + dfo_end_sem) / opt_l, (dfo_end_m - dfo_end_sem) / opt_l, color = colors[7], alpha = 0.3)

plt.plot(xs, pgd_end_m / opt_l, label = 'PGD', color = colors[4])
plt.fill_between(xs, (pgd_end_m + pgd_end_sem) / opt_l, (pgd_end_m - pgd_end_sem) / opt_l, color = colors[4], alpha = 0.3)

plt.plot(xs, spgd_end_m / opt_l, label = 'SPGD', color = colors[2])
plt.fill_between(xs, (spgd_end_m + spgd_end_sem) / opt_l, (spgd_end_m - spgd_end_sem) / opt_l, color = colors[2], alpha = 0.3)


plt.xlabel('Steps for mean to settle')
plt.ylabel('% of optimal revenue')
plt.xscale('log')
# plt.xticks([2**k for k in range(7)], [2**k for k in range(7)])
plt.xticks(xs, xs)
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(linewidth)
leg.set_draggable(state=True)


mus = np.zeros((32, 2))
mus[0] = np.random.randn(2)
# mus[0] = -2 * np.ones(2)
# th = np.zeros(2)
th = np.random.randn(2)
for t in range(1, 32):
    mus[t] = expt.m(th, mus[t-1])

lt = expt.lt_mu(th)

plt.figure()
plt.scatter(mus[0,0], mus[0,1], c='c', marker='x', label='Initial mean')
plt.scatter(mus[1:, 0], mus[1:, 1], c = range(31), cmap = 'viridis', label = 'Mean updates')
plt.scatter(lt[0], lt[1], c = 'r', marker='*', label = 'Long-term mean')
plt.legend()

