import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
from sklearn.datasets import make_spd_matrix
from tqdm import tqdm
import time
import csv
import sys
import os

import matplotlib as mpl

linewidth = 3.
mpl.rcParams['lines.linewidth'] = linewidth

plt.rcParams.update({'font.size': 22})


class Bottleneck:
    def __init__(self, d, eps, reg, mean_noise, seed = 0, R = None, sigma = None, mu0 = None):
        # Problem instance constants.
        np.random.seed(seed)
        
        self.d = d
        self.eps = eps
        self.reg = reg
        self.mean_noise = mean_noise
        
        if R is None:
            self.R = 1. / np.sqrt(d)
        else:
            self.R = R
        
        if sigma is None:
            self.Sigma = make_spd_matrix(d)
        else:
            self.Sigma = (sigma ** 2) * np.eye(d)
        
        if mu0 is None:
            self.mu0 = np.random.rand(d)
            self.mu0 /= np.linalg.norm(self.mu0)
        
        self.theta0 = np.clip(np.random.randn(d), -self.R, self.R)
    
    
    # Simulation helper methods
    def score(self, theta, mu):
        return np.dot(theta, mu)
    
    
    def m(self, s):
        # Does changing 1 - s --> C - s for large enough C fix the blow up problem?
        return (1. - s) * self.mu0
    
    
    def M(self, theta, mu):
        return self.m(self.score(theta, mu))
    
    
    
    # Bottleneck PerfGD helper methods
    def ds_dtheta(self, theta, mu):
        return mu.reshape((1, -1))
    
    
    def ds_dmu(self, theta, mu):
        return theta.reshape((1, -1))
    
    
    def dM_dtheta(self, theta, mu, dm_ds):
        return dm_ds @ self.ds_dtheta(theta, mu)
    
    
    def dM_dmu(self, theta, mu, dm_ds):
        return dm_ds @ self.ds_dmu(theta, mu)
    
    
    def approx_Mk_jacobian(self, k, theta, mu, dm_ds):
        d1M = self.dM_dtheta(theta, mu, dm_ds)
        d2M = self.dM_dtheta(theta, mu, dm_ds)
        
        if k is None:
            return np.linalg.solve(np.eye(self.d) - d2M, d1M)
        else:
            return np.linalg.solve(np.eye(self.d) - d2M, (np.eye(self.d) - np.linalg.matrix_power(d1M, k)) @ d1M)
        
        # if k == 0:
        #     return np.zeros((self.d, self.d))
        # else:
        #     return self.dM_dtheta(theta, mu, dm_ds) + self.dM_dmu(theta, mu, dm_ds) @ self.approx_Mk_jacobian(k - 1, theta, mu, dm_ds)
    
    
    def approx_long_term_perf_grad(self, k, theta, mu, dm_ds):
        return -mu - (self.approx_Mk_jacobian(k, theta, mu, dm_ds)) @ theta + self.reg * theta
    
    
    def finite_diff_approx(self, out_hist, in_hist):
        # Error for 1D bottleneck case: interprets as a 1D array instead of matrix
        Delta_out = np.array([o - out_hist[-1] for o in out_hist]).T.reshape((-1, len(out_hist)))
        Delta_in  = np.array([i - in_hist[-1] for i in in_hist]).T.reshape((-1, len(in_hist)))
        return Delta_out @ np.linalg.pinv(Delta_in, rcond = 1e-8)
    
    
    
    # Vanilla stateful PerfGD helper methods
    # def exact_dM_dmu(self, theta, mu):
    #     return -np.outer(self.mu0, theta)
    
    
    # def vanilla_approx_Mk_jacobian(self, k, theta, mu, dM_dth):
    #     if k == 0:
    #         return np.zeros((d, d))
    #     else:
    #         return dM_dth + self.exact_dM_dmu(theta, mu) @ self.vanilla_approx_Mk_jacobian(k - 1, theta, mu, dM_dth)    
    
    
    def vanilla_approx_Mk_jacobian(self, k, theta, mu, M_grad):
        d1M = M_grad[:, :self.d]
        d2M = M_grad[:, self.d:]
        if k is None:
            return np.linalg.solve(np.eye(self.d) - d2M, d1M)
        else:
            return np.linalg.solve(np.eye(self.d) - d2M, (np.eye(self.d) - np.linalg.matrix_power(d1M, k)) @ d1M)
    
    
    def vanilla_approx_long_term_perf_grad(self, k, theta, mu, dM_dth):
        return -mu - (self.vanilla_approx_Mk_jacobian(k, theta, mu, dM_dth)) @ theta + self.reg * theta
    
    
    
    # Exact GD helper methods
    def long_term_mu(self, theta):
        return self.mu0 / (1. + np.dot(theta, self.mu0))
    
    
    def long_term_loss(self, theta):
        return -np.dot(theta, self.long_term_mu(theta)) + (self.reg / 2) * (np.linalg.norm(theta) ** 2)
    
    
    def long_term_grad(self, theta):
        return self.reg * theta - self.mu0 / ((1 + self.score(theta, self.mu0)) ** 2)
    
    
    
    # Define the four optimization algs
    def SPGD(self, T, lr, k, H = None, pert = 0.):
        init = self.d
        # m_hist       = deque(maxlen = H)
        # th_hist      = deque(maxlen = H)
        full_th_hist = deque()
        
        inputs = deque(maxlen = H)
        outputs = deque(maxlen = H)
        
        prev_mu = self.mu0.copy()
        prev_est_mu = prev_mu + self.mean_noise * np.random.randn(self.d)
        cur_th  = self.theta0.copy()
        cur_in  = np.concatenate([cur_th, prev_est_mu])
        
        for t in range(init):
            cur_mu = self.M(cur_th, prev_mu).copy()
            est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            outputs.append(est_mu.copy())
            inputs.append(cur_in.copy())
            full_th_hist.append(cur_th.copy())
            
            cur_th = np.clip(cur_th + 0.1 * np.random.randn(d), -self.R, self.R)
            prev_mu = cur_mu
            prev_est_mu = est_mu
            cur_in = np.concatenate([cur_th, prev_est_mu])
        
        for t in range(T - init):
            cur_mu = self.M(cur_th, prev_mu).copy()
            est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            outputs.append(est_mu.copy())
            inputs.append(cur_in.copy())
            full_th_hist.append(cur_th.copy())
            
            M_grad = self.finite_diff_approx(outputs, inputs)
            grad   = self.vanilla_approx_long_term_perf_grad(k, cur_th, est_mu, M_grad)
            
            cur_th = np.clip(cur_th - lr * (grad + pert * np.random.randn(self.d)), -self.R, self.R)
            prev_mu = cur_mu
            prev_est_mu = est_mu
            cur_in = np.concatenate([cur_th, prev_est_mu])
        
        return full_th_hist
            
    
    
    def BSPGD(self, T, lr, k, H = None, pert = 0.):
        init = self.d
        
        m_hist  = deque(maxlen = H)
        s_hist  = deque(maxlen = H)
        th_hist = deque()
        
        prev_mu = self.mu0.copy()
        prev_est_mu = prev_mu + self.mean_noise * np.random.randn(self.d)
        cur_th  = self.theta0.copy()
        
        for t in range(init):
            est_s   = self.score(cur_th, prev_est_mu)
            cur_mu  = self.M(cur_th, prev_mu).copy()
            est_mu  = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            s_hist.append(est_s)
            m_hist.append(est_mu)
            th_hist.append(cur_th.copy())
            
            cur_th = np.clip(cur_th + 0.1 * np.random.randn(self.d), -self.R, self.R)
            prev_mu = cur_mu
            prev_est_mu = est_mu
        
        
        for t in range(T - init):
            est_s   = self.score(cur_th, prev_est_mu)
            cur_mu  = self.M(cur_th, prev_mu).copy()
            est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            s_hist.append(est_s)
            m_hist.append(est_mu)
            th_hist.append(cur_th.copy())
            
            dm_ds = self.finite_diff_approx(m_hist, s_hist)
            grad  = self.approx_long_term_perf_grad(k, cur_th, est_mu, dm_ds)
            
            cur_th = np.clip(cur_th - lr * (grad + pert * np.random.randn(self.d)), -self.R, self.R)
            prev_mu = cur_mu
            prev_est_mu = est_mu
        
        return th_hist
    
    
    
    def GD(self, T, lr):
        th_hist = deque()
        
        cur_th = self.theta0.copy()
        
        for t in range(T):
            th_hist.append(cur_th.copy())
            
            grad = self.long_term_grad(cur_th)
            
            cur_th = np.clip(cur_th - lr * grad, -self.R, self.R)
        
        return th_hist
    
    
    
    def DFO(self, T, lr, perturbation, wait):
        th_hist = deque()
        q_hist = deque()
        
        internal_th = self.theta0.copy()
        
        u = np.random.randn(self.d)
        u /= np.linalg.norm(u)
        deploy_th = internal_th + perturbation * u
        
        prev_mu = self.mu0.copy()
        
        for t in range(T):            
            th_hist.append(internal_th.copy())
            q_hist.append(deploy_th.copy())
            
            cur_mu = self.M(deploy_th, prev_mu).copy()
            est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            if t % wait == wait - 1:
                u = np.random.randn(self.d)
                u /= np.linalg.norm(u)
                
                loss = -np.dot(deploy_th, est_mu) + (self.reg / 2) * np.linalg.norm(deploy_th) ** 2
                grad = (self.d * loss / perturbation) * u
                
                internal_th = np.clip(internal_th - lr * grad, -self.R, self.R)
                deploy_th = np.clip(internal_th + perturbation * u, -self.R, self.R).copy()
                
            prev_mu = cur_mu
        
        return th_hist, q_hist
    
    
    def PGD(self, T, lr, wait, H = None):
        init = self.d
        
        mus  = deque(maxlen = H)
        ths  = deque(maxlen = H)
        th_hist = deque()
        
        prev_mu = self.mu0.copy()
        cur_th  = self.theta0.copy()
        
        for t in range(wait * init):
            th_hist.append(cur_th.copy())
            cur_mu  = self.M(cur_th, prev_mu).copy()
            
            if t % wait == wait - 1:
                est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
                mus.append(est_mu.copy())
                ths.append(cur_th.copy())
            
                cur_th = np.clip(cur_th + 0.1 * np.random.randn(self.d), -self.R, self.R)
                
            prev_mu = cur_mu
        
        
        for t in range(T - wait * init):
            th_hist.append(cur_th.copy())
            cur_mu  = self.M(cur_th, prev_mu).copy()
            
            if t % wait == wait - 1:
                est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
                mus.append(est_mu.copy())
                ths.append(cur_th.copy())
                
                dmu_dth = self.finite_diff_approx(mus, ths)
                grad = -(est_mu + dmu_dth.T @ cur_th)
                cur_th = np.clip(cur_th - lr * grad, -self.R, self.R)
            
            prev_mu = cur_mu
        
        return th_hist
    
    
    def RGD(self, T, lr):
        th_hist = deque()
        cur_th = self.theta0.copy()
        cur_mu = self.mu0.copy()
        
        for t in range(T):
            th_hist.append(cur_th.copy())
            cur_mu = self.M(cur_th, cur_mu)
            est_mu = cur_mu + self.mean_noise * np.random.randn(self.d)
            
            grad = -est_mu
            cur_th = np.clip(cur_th - lr * grad, -self.R, self.R)
        
        return th_hist



def experiment(num_trials, T, d, eps, reg, mean_noise, seed, lrs, dfo_perts, spgd_perts, Hs, spgd_Hs, waits, ks = [None], init_th = None):        
    rgd_th   = np.empty((len(lrs), num_trials), dtype=object)
    dfo_i_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    dfo_q_th = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    pgd_th   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
    spgd_th  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    bspgd_th  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
    rgd_l   = np.empty((len(lrs), num_trials), dtype=object)
    dfo_i_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    dfo_q_l = np.empty((len(lrs), len(dfo_perts), len(waits), num_trials), dtype=object)
    pgd_l   = np.empty((len(lrs), len(Hs), len(waits), num_trials), dtype=object)
    spgd_l  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    bspgd_l  = np.empty((len(lrs), len(spgd_perts), len(Hs), len(ks), num_trials), dtype=object)
    
    
    for r in tqdm(range(num_trials)):
        expt = Bottleneck(d, eps, reg, mean_noise, seed + r)
        
        # RGD experiments
        print('Running RGD...')
        for i in range(len(lrs)):
            lr   = lrs[i]
            
            traj = expt.RGD(T, lr)
            rgd_l[i, r]  = np.array([expt.long_term_loss(th) for th in traj], dtype=float)
            rgd_th[i, r] = np.array(traj)
        
        # DFO experiments
        print('Running DFO...')
        for i in range(len(lrs)):
            for j in range(len(dfo_perts)):
                for k in range(len(waits)):
                    lr   = lrs[i]
                    pert = dfo_perts[j]
                    wait = waits[k]
                    
                    i_traj, q_traj = expt.DFO(T, lr, pert, wait)
                    dfo_i_l[i, j, k, r]  = np.array([expt.long_term_loss(th) for th in i_traj], dtype=float)
                    dfo_q_l[i, j, k, r]  = np.array([expt.long_term_loss(th) for th in q_traj], dtype=float)
                    dfo_i_th[i, j, k, r] = np.array(i_traj)
                    dfo_q_th[i, j, k, r] = np.array(q_traj)
        
        # PGD experiments
        print('Running PGD...')
        for i in range(len(lrs)):
            for j in range(len(Hs)):
                for k in range(len(waits)):
                    lr   = lrs[i]
                    H    = Hs[j]
                    wait = waits[k]                    
                    
                    traj = expt.PGD(T, lr, wait, H)
                    pgd_l[i, j, k, r]  = np.array([expt.long_term_loss(th) for th in traj], dtype=float)
                    pgd_th[i, j, k, r] = np.array(traj)
        
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
                        
                        traj = expt.SPGD(T, lr, k, H, pert)
                        spgd_l[i, j, l, m, r]  = np.array([expt.long_term_loss(th) for th in traj], dtype=float)
                        spgd_th[i, j, l, m, r] = np.array(traj)
        
        # BSPGD experiments
        print('Running BSPGD...')
        for i in range(len(lrs)):
            for j in range(len(spgd_perts)):
                for l in range(len(Hs)):
                    for m in range(len(ks)):
                        lr   = lrs[i]
                        pert = spgd_perts[j]
                        H    = Hs[l]
                        k    = ks[m]
                        
                        traj = expt.SPGD(T, lr, k, H, pert)
                        bspgd_l[i, j, l, m, r]  = np.array([expt.long_term_loss(th) for th in traj], dtype=float)
                        bspgd_th[i, j, l, m, r] = np.array(traj)
                        
    results = {'rgd_th': rgd_th, 'rgd_l': rgd_l,
                'dfo_i_th': dfo_i_th, 'dfo_i_l': dfo_i_l,
                'dfo_q_th': dfo_q_th, 'dfo_q_l': dfo_q_l,
                'pgd_th': pgd_th, 'pgd_l': pgd_l,
                'spgd_th': spgd_th, 'spgd_l': spgd_l,
                'bspgd_th': bspgd_th, 'bspgd_l': bspgd_l}
    return results


def make_plots(num_trials, T, d, eps, reg, mean_noise, seed, lrs, dfo_perts, spgd_perts, Hs, waits, ks = [None], init_th = None):
    
    results = experiment(num_trials, T, d, eps, reg, mean_noise, seed, lrs, dfo_perts, spgd_perts, Hs, waits, ks = [None], init_th = None)
         
    ###############################################################################
    # Find best hyperparams for each method
    ###############################################################################
    rgd_l   = results['rgd_l']
    dfo_i_l = results['dfo_i_l']
    dfo_q_l = results['dfo_q_l']
    pgd_l   = results['pgd_l']
    spgd_l  = results['spgd_l']
    bspgd_l = results['bspgd_l']
    
    rgd_l_mean   = np.mean(rgd_l, axis = -1)
    dfo_i_l_mean = np.mean(dfo_i_l, axis = -1)
    dfo_q_l_mean = np.mean(dfo_q_l, axis = -1)
    pgd_l_mean   = np.mean(pgd_l, axis = -1)
    spgd_l_mean  = np.mean(spgd_l, axis = -1)
    bspgd_l_mean = np.mean(bspgd_l, axis = -1)
    
    rgd_end   = np.zeros(rgd_l_mean.shape).flatten()
    dfo_i_end = np.zeros(dfo_i_l_mean.shape).flatten()
    dfo_q_end = np.zeros(dfo_q_l_mean.shape).flatten()
    pgd_end   = np.zeros(pgd_l_mean.shape).flatten()
    spgd_end  = np.zeros(spgd_l_mean.shape).flatten()
    bspgd_end = np.zeros(bspgd_l_mean.shape).flatten()
    
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
    
    for i, loss_traj in enumerate(bspgd_l_mean.flatten()):
        T = len(loss_traj)
        bspgd_end[i] = np.mean(loss_traj[-int(T/10):])
    
    rgd_best  = np.unravel_index(np.nanargmin(rgd_end), rgd_l_mean.shape)
    dfo_best  = np.unravel_index(np.nanargmin(dfo_i_end), dfo_i_l_mean.shape)
    pgd_best  = np.unravel_index(np.nanargmin(pgd_end), pgd_l_mean.shape)
    spgd_best = np.unravel_index(np.nanargmin(spgd_end), spgd_l_mean.shape)
    bspgd_best = np.unravel_index(np.nanargmin(bspgd_end), bspgd_l_mean.shape)
    
    
    print('')
    print(f'RGD:   lr={round(lrs[rgd_best[0]], 3)}')
    print(f'DFO:   lr={round(lrs[dfo_best[0]], 3)}, pert={round(dfo_perts[dfo_best[1]], 3)}, wait={waits[dfo_best[2]]}')
    print(f'PGD:   lr={round(lrs[pgd_best[0]], 3)}, H={Hs[pgd_best[1]]}, wait={waits[pgd_best[2]]}')
    print(f'SPGD:  lr={round(lrs[spgd_best[0]], 3)}, pert={round(spgd_perts[spgd_best[1]], 3)}, H={Hs[spgd_best[2]]}, k={ks[spgd_best[3]]}')
    print(f'BSPGD: lr={round(lrs[bspgd_best[0]], 3)}, pert={round(spgd_perts[bspgd_best[1]], 3)}, H={Hs[bspgd_best[2]]}, k={ks[bspgd_best[3]]}')
    
    ###############################################################################
    # Make plots
    ###############################################################################
    
    best_rgd_l  = np.asfarray([x for x in rgd_l[rgd_best]])
    best_dfo_l  = np.asfarray([x for x in dfo_i_l[dfo_best]])
    best_dfo_q_l = np.asfarray([x for x in dfo_q_l[dfo_best]])
    best_pgd_l  = np.asfarray([x for x in pgd_l[pgd_best]])
    best_spgd_l = np.asfarray([x for x in spgd_l[spgd_best]])
    best_bspgd_l = np.asfarray([x for x in bspgd_l[bspgd_best]])
    
    rgd_l_m  = np.mean(best_rgd_l, axis=0)
    dfo_l_m  = np.mean(best_dfo_l, axis=0)
    dfo_q_l_m = np.mean(best_dfo_q_l, axis=0)
    pgd_l_m  = np.mean(best_pgd_l, axis=0)
    spgd_l_m = np.mean(best_spgd_l, axis=0)
    bspgd_l_m = np.mean(best_bspgd_l, axis=0)
    
    rgd_l_sem  = sem(best_rgd_l, axis=0)
    dfo_l_sem  = sem(best_dfo_l, axis=0)
    dfo_q_l_sem = sem(best_dfo_q_l, axis=0)
    pgd_l_sem  = sem(best_pgd_l, axis=0)
    spgd_l_sem = sem(best_spgd_l, axis=0)
    bspgd_l_sem = sem(best_bspgd_l, axis=0)
    
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    
    plt.plot(rgd_l_m, label = f'RGD: lr={round(lrs[rgd_best[0]], 3)}', color=colors[5])
    plt.fill_between(range(T), rgd_l_m + rgd_l_sem, rgd_l_m - rgd_l_sem, color=colors[5], alpha=0.3)
    
    plt.plot(dfo_l_m, label = f'DFO (i): lr={round(lrs[dfo_best[0]], 3)}, pert={round(dfo_perts[dfo_best[1]], 3)}, wait={waits[dfo_best[2]]}', color=colors[7])
    plt.fill_between(range(T), dfo_l_m + dfo_l_sem, dfo_l_m - dfo_l_sem, color=colors[7], alpha=0.3)
    
    # plt.plot(dfo_q_l_m, label = f'DFO (q)', color=colors[3])
    # plt.fill_between(range(T), dfo_q_l_m + dfo_q_l_sem, dfo_q_l_m - dfo_q_l_sem, color=colors[3], alpha=0.3)
    
    plt.plot(pgd_l_m, label = f'PGD: lr={round(lrs[pgd_best[0]], 3)}, H={Hs[pgd_best[1]]}, wait={waits[pgd_best[2]]}', color=colors[4])
    plt.fill_between(range(T), pgd_l_m + pgd_l_sem, pgd_l_m - pgd_l_sem, color=colors[4], alpha=0.3)
    
    plt.plot(spgd_l_m, label = f'SPGD: lr={round(lrs[spgd_best[0]], 3)}, pert={round(spgd_perts[spgd_best[1]], 3)}, H={Hs[spgd_best[2]]}, k={ks[spgd_best[3]]}', color=colors[2])
    plt.fill_between(range(T), spgd_l_m + spgd_l_sem, spgd_l_m - spgd_l_sem, color=colors[2], alpha=0.3)
    
    plt.plot(bspgd_l_m, label = f'BSPGD: lr={round(lrs[bspgd_best[0]], 3)}, pert={round(spgd_perts[bspgd_best[1]], 3)}, H={Hs[bspgd_best[2]]}, k={ks[bspgd_best[3]]}', color=colors[8])
    plt.fill_between(range(T), bspgd_l_m + bspgd_l_sem, bspgd_l_m - bspgd_l_sem, color=colors[8], alpha=0.3)
    
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    if not os.path.isdir('plots/bottleneck'):
        os.mkdir('plots/bottleneck')
        
    plt.savefig(f'plots/classification/{mean_noise}.pdf')
    
    return rgd_best, best_rgd_l, dfo_best, best_dfo_l, pgd_best, best_pgd_l, spgd_best, best_spgd_l





# Problem instance constants.
d = 5
R = 1 / np.sqrt(d)
reg = 1.
sigma = 1
eps = 1.5
mean_noise = 0.1


# # Perform grid search for each method.
# lrs        = [10 ** (-k/2) for k in range(1, 7)]
# waits      = [1, 5, 10, 20]
# ks         = [None]
# dfo_perts  = [10 ** (-k/2) for k in range(4)]
# spgd_perts = [0] + [10 ** (-k/2) for k in range(4)]
# Hs         = [None] + [d + k for k in range(d + 1)]
# spgd_Hs = [None] + [2 * d + k for k in range(d + 1)]


# # Run experiments
# T = 40
# num_trials = 10
# seed = 0


# rgd_best, best_rgd_l, dfo_best, best_dfo_l, pgd_best, best_pgd_l, spgd_best, best_spgd_l = make_plots(num_trials, T, d, eps, reg, mean_noise, seed, lrs, dfo_perts, spgd_perts, Hs, waits, ks = [None], init_th = None)

# Results:
# RGD:   lr=0.1
# DFO:   lr=0.01, pert=0.032, wait=1
# PGD:   lr=0.316, H=9, wait=1
# SPGD:  lr=0.1, pert=0.032, H=5, k=None
# BSPGD: lr=0.316, pert=0.1, H=None, k=None


# Plot results of best hyperparams for each method.
gd_lr = 0.1

rgd_lr = 0.1

dfo_lr = 0.01
dfo_pert = 0.032
dfo_wait = 1

pgd_lr = 0.316
pgd_H = 9
pgd_wait = 1

spgd_lr = 0.1
spgd_pert = 0.032
spgd_H = 5
spgd_k = None

bspgd_lr = 0.316
bspgd_pert = 0.1
bspgd_H = None
bspgd_k = None

num_trials = 100
T = 40

rgd_l  = np.zeros((num_trials, T))
dfo_i_l  = np.zeros((num_trials, T))
dfo_q_l  = np.zeros((num_trials, T))
pgd_l  = np.zeros((num_trials, T))
spgd_l = np.zeros((num_trials, T))
bspgd_l = np.zeros((num_trials, T))

for r in tqdm(range(num_trials)):
    expt = Bottleneck(d, eps, reg, mean_noise, seed = 12345 + r, R = None, sigma = None, mu0 = None)
    
    bspgd_th = expt.BSPGD(T, bspgd_lr, bspgd_k, bspgd_H, bspgd_pert)
    spgd_th = expt.SPGD(T, spgd_lr, spgd_k, spgd_H, spgd_pert)
    pgd_th = expt.PGD(T, pgd_lr, pgd_wait, pgd_H)
    rgd_th = expt.RGD(T, rgd_lr)
    gd_th   = expt.GD(T, gd_lr)
    dfo_i_th, dfo_q_th = expt.DFO(T, dfo_lr, dfo_pert, dfo_wait)
    
    
    rgd_l[r]  = np.array([expt.long_term_loss(th) for th in rgd_th])
    
    dfo_i_l[r] = np.array([expt.long_term_loss(th) for th in dfo_i_th])
    dfo_q_l[r] = np.array([expt.long_term_loss(th) for th in dfo_q_th])
    
    pgd_l[r]  = np.array([expt.long_term_loss(th) for th in pgd_th])
    
    spgd_l[r] = np.array([expt.long_term_loss(th) for th in spgd_th])
    
    bspgd_l[r] = np.array([expt.long_term_loss(th) for th in bspgd_th])


rgd_m = np.mean(rgd_l, axis = 0)
dfo_i_m = np.mean(dfo_i_l, axis = 0)
dfo_q_m = np.mean(dfo_q_l, axis = 0)
pgd_m = np.mean(pgd_l, axis = 0)
spgd_m = np.mean(spgd_l, axis = 0)
bspgd_m = np.mean(bspgd_l, axis = 0)

rgd_e = sem(rgd_l, axis = 0)
dfo_i_e = sem(dfo_i_l, axis = 0)
dfo_q_e = sem(dfo_q_l, axis = 0)
pgd_e = sem(pgd_l, axis = 0)
spgd_e = sem(spgd_l, axis = 0)
bspgd_e = sem(bspgd_l, axis = 0)



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()

plt.plot(range(T), rgd_m, label = 'RGD', color = colors[5])
plt.fill_between(range(T), rgd_m + rgd_e, rgd_m - rgd_e, color = colors[5], alpha = 0.3)

plt.plot(range(T), dfo_i_m, label = 'DFO', color = colors[7])
plt.fill_between(range(T), dfo_i_m + dfo_i_e, dfo_i_m - dfo_i_e, color = colors[7], alpha = 0.3)

# plt.plot(range(T), dfo_q_m, label = 'DFO (q)', color = colors[3])
# plt.fill_between(range(T), dfo_q_m + dfo_q_e, dfo_q_m - dfo_q_e, color = colors[3], alpha = 0.3)

plt.plot(range(T), pgd_m, label = 'PGD', color = colors[4])
plt.fill_between(range(T), pgd_m + pgd_e, pgd_m - pgd_e, color = colors[4], alpha = 0.3)

plt.plot(range(T), spgd_m, label = 'SPGD', color = colors[2])
plt.fill_between(range(T), spgd_m + spgd_e, spgd_m - spgd_e, color = colors[2], alpha = 0.3)

plt.plot(range(T), bspgd_m, label = 'BSPGD', color = colors[9])
plt.fill_between(range(T), bspgd_m + bspgd_e, bspgd_m - bspgd_e, color = colors[9], alpha = 0.3)


plt.xlabel('Training iteration')
plt.ylabel('Long-term loss')
leg = plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(linewidth)
leg.set_draggable(state=True)



