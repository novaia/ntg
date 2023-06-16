# Import numpy library
import numpy as np

# Load the two numpy files
file1 = np.load('../data/dataset_info/second_pass_fid_stats.npz')
file2 = np.load('../data/dataset_info/stats.npz')

# Extract the matrices 'mu' and 'sigma' from each file
mu1 = file1['mu']
sigma1 = file1['sigma']
mu2 = file2['mu']
sigma2 = file2['sigma']

# Compare the mu and sigma from each file to see if they are the same
if np.array_equal(mu1, mu2) and np.array_equal(sigma1, sigma2):
    print("The mu and sigma from both files are the same.")
else:
    print("The mu and sigma from both files are different.")