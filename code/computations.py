#%%

import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt

fig_path =  '../figures/'

#%%

def run_test(ns, delta_mu, sigma_1, sigma_2, alpha, min_pow, plot=False):
    mean_scales_1 = sigma_1 / np.sqrt(ns)
    mean_scales_2 = sigma_2 / np.sqrt(ns)
    test_scales = np.sqrt( mean_scales_1**2 + mean_scales_2**2 )

    t_alpha = test_scales * stats.norm.ppf(1-alpha)
    pow = 1 - stats.norm.cdf( (t_alpha-delta_mu) / test_scales )
    n_min_pow = np.min( ns[ pow > min_pow ] )

    if plot:
        plt.figure()
        plt.scatter( ns, pow )
        plt.axhline( min_pow, color='red', linestyle='--', label='Wanted power' )
        plt.axvline( n_min_pow, color='green', linestyle='--', label=f'Minimal group size = {n_min_pow}' )
        plt.legend()
        plt.xlabel('Group size')
        plt.ylabel('Test power')
        plt.savefig(f'{fig_path}power-vs-group-size.png')
    return n_min_pow

# %%

n_min = 10
n_max = 100
ns = np.arange(n_min, n_max, dtype=int)
alpha = 0.05
min_pow = 0.8
delta_mu = 10
sigma_1 = 15.0

run_test(ns, delta_mu, sigma_1, sigma_1, alpha, min_pow, plot=True)

#%%

param_range = 0.2
n_deltas = 200
delta_mus = delta_mu * ( 1 +  param_range * np.linspace( -1.0, 1.0, n_deltas ) )
delta_sigmas = sigma_1 * param_range * np.linspace( -1.0, 1.0, n_deltas )

ns_min_beta = np.zeros( (len(delta_sigmas), len(delta_mus)) )
for i,delta_sigma in enumerate(delta_sigmas):
    for j,delta_mu in enumerate(delta_mus):
        ns_min_beta[i,j] = run_test(ns, delta_mu, sigma_1, sigma_1+delta_sigma, alpha, min_pow, plot=False)

plt.figure()
grid_delta_mus, grid_sigmas_2 = np.meshgrid(delta_mus, sigma_1 + delta_sigmas)
plt.contourf(grid_delta_mus, grid_sigmas_2, ns_min_beta, levels=20)
#plt.contourf(grid_delta_mus, grid_sigmas_2/sigma_1, ns_min_beta, levels=20)
# try contour lines 
plt.colorbar( label='Minimal group size' )
plt.xlabel(r'$\Delta \mu$')
#plt.ylabel(r'$\sigma_2 / \sigma_1$')
plt.ylabel(r'$\sigma_A$')
plt.savefig(f'{fig_path}minimal-group-size-simulations.png')

#%%

# print the value associated to delta_mu=9 and sigma_2=16.5
i = np.argmin( np.abs( delta_mus - 9 ) )
j = np.argmin( np.abs( delta_sigmas - 1.5 ) )
print( f'Minimal group size for delta_mu=9 and sigma_2=16.5: {ns_min_beta[j,i]}' )


#%%

def plot_stuff( delta_mu, delta_sigma, ax, n=40, plot_std=False ):
    sigma_0 = 15
    mu_1 = 100
    sigma_1 = sigma_0 / np.sqrt(n)
    sigma_2 = (sigma_0+delta_sigma) / np.sqrt(n)
    mu2 = mu_1 + delta_mu
    lab_mu = r'\Delta\mu'
    lab_sigma = r'\Delta\sigma'
    lab_1 = 'Control'
    lab_2 = f'${lab_mu}={delta_mu:.1f},{lab_sigma}={delta_sigma:.1f}$'
    
    if plot_std:
        x_1 = np.linspace( mu_1-5*sigma_1, mu_1+5*sigma_1, 100 )
        ax.plot( x_1, stats.norm.pdf( x_1, mu_1, sigma_1 ), label=lab_1, color='black' )
        ax.set_xlabel(r'$\overline{Y}$')    
        ax.set_ylabel(r'$f(\overline{Y})$')    
    x_2 = np.linspace( mu2-5*sigma_2, mu2+5*sigma_2, 100 ) 
    ax.plot( x_2, stats.norm.pdf( x_2, mu2, sigma_2 ), label=lab_2 )

    return

fig, ax = plt.subplots( 1, 1 )
plot_stuff( delta_mus[-1], delta_sigmas[0], ax, plot_std=True )
plot_stuff( 10, 0, ax )
plot_stuff( delta_mus[0], delta_sigmas[-1], ax )

ax.legend(loc='upper left')
plt.savefig(f'{fig_path}scenarios.png')

# %%
