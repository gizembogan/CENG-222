# %% Imports
import random
import numpy as np
from matplotlib import pyplot as plt


# %% Functions

# Function to generate a population with given parameter and size using the
# inverse transformation method.
def gen_inverse(k, M):
    # inverse of the f(x) is => F(u) = u^ (1/ k+1)

    population = np.zeros(M)
    for i in range(M):
        u = random.random()
        population[i] = u**( 1 / (k+1) )
    return population


# Function to generate a population with given parameter and size using the
# rejection method.
def gen_rejection(k, M):
    

    population = np.zeros(M)
    count = 0
    while count < M:
        x = random.random()
        y = random.random() * (k+1)
        if y <= ((k+1) * (x**k)):     # f(x) = (k+1) * (x^k)
            population[count] = x
            count += 1 
    return population
    


# Function to calculate the population mean using k.
def calc_population_mean(k):
    # E[x] = integral of x * f(x) on [0,1] => (k+1) / (k+2)
    mean = ((k+1) / (k+2))
    return mean

# Function to calculate the population variance using k.
def calc_population_variance(k):
    # E[x^2] = integral of (x^2) * f(x) on [0,1] => (k+1) / (k+3)
    ex2 = ((k+1) / (k+3))
    ex = ((k+1) / (k+2))
    variance = (ex2 - (ex**2))
    return variance


# Function to randomly take samples of size N from a population.
def random_sample(population, N):
    sample = np.zeros(N)

    for i in range(N):
        index = int(random.random() * (len(population)))
        sample[i] = population[index]
    return sample


# Function to calculate the sample mean.
def calc_sample_mean(sample):
    n = len(sample)
    mean = (np.sum(sample) / n)
    return mean


# Function to calculate the sample variance (biased/unbiased).
def calc_sample_variance(sample, unbiased=True):
    mean = calc_sample_mean(sample)
    value = 0
    for i in range (len(sample)):
        value += ((sample[i] - mean)**2)

    if unbiased:
        variance = (value / (len(sample) - 1))
    else:
        variance = (value / len(sample))

    return variance


# Function to estimate the parameter k using method of moments
def estimate_k_mom(sample):     #equation is calculated by hand
    mean = calc_sample_mean(sample)
    est_k = (((2 * mean) - 1) / (1 - mean))
    return est_k
    
# Function to estimate the parameter k using maximum likelihood
def estimate_k_mle(sample):      #equation is calculated by hand
    sample = np.array(sample)
    ln_of_x = np.log(sample)
    sum_of_lns = np.sum(ln_of_x)
    est_k = ((-len(sample)) / sum_of_lns) - 1
    return est_k
  

# Function to calculate the confidence interval for population mean given the
# sample and the required confidence level. If population standard deviation is
# not provided, use sample standard deviation as its estimator. As confidence
# level, it should only accept 95, 96, 97, 98 and 99 for which the z values are
# hard-coded in the function.
def calc_conf_int_mean(sample, confidence_lvl, pop_std=0):
    z_value = 0
    if confidence_lvl == 95:
        z_value = 1.645
    elif confidence_lvl == 96:
        z_value = 1.750
    elif confidence_lvl == 97:
        z_value = 1.880
    elif confidence_lvl == 98:
        z_value = 2.054
    elif confidence_lvl == 99:
        z_value = 2.326
    
    n = len(sample)
    s_mean = calc_sample_mean(sample)

    if pop_std == 0:
        s_var = calc_sample_variance(sample)
        std = s_var**(1/2)
    else:
        std = pop_std


    e = z_value * (std / (n**(1/2)))
    
    l_bound = s_mean - e
    u_bound = s_mean + e
    
    return (l_bound, u_bound)

# %% Experiments

# Generate the two populations of size 1000000, calculate and print their means
# and variances and plot the population histograms.
M = 1000000
k_1 = 2.1
k_2 = 3.7
conf_lvl = 97

population_1 = gen_inverse(k_1, M)
population_2 = gen_rejection (k_2, M)

p_mean_1 = calc_population_mean(k_1)
p_mean_2 = calc_population_mean(k_2)
print("-- Means of the Populations --")
print("")
print("Population 1: ", p_mean_1)
print("Population 2: ", p_mean_2)
print("")

p_var_1 = calc_population_variance(k_1)
p_var_2 = calc_population_variance(k_2)
print("-- Variances of the Populations --")
print("Population 1: ", p_var_1)
print("Population 2: ", p_var_2)
print("")

pop_std1 = (p_var_1**(1/2))
pop_std2 = (p_var_2**(1/2))

plt.figure()

plt.subplot(1, 2, 1)
plt.hist(population_1, bins=50, alpha=0.7, color='blue')
plt.title(f'Population 1 (with Inverse Method)')

plt.subplot(1, 2, 2)
plt.hist(population_2, bins=50, alpha=0.7, color='red')
plt.title(f'Population 2 (with Rejection Method)')

plt.tight_layout()

# Collect 100000 random samples of size 25 from both populations, calculate
# sample means, biased and unbiased sample variances, MoM and MLE estimates of
# the parameter k and population mean intervals with 97% confidence with and
# without the population standard deviation for each sample of each population.
N = 25
R = 100000

samples1 = np.zeros([R, N])
samples2 = np.zeros([R, N])


s_means1 = np.zeros(R)
s_means2 = np.zeros(R)


s_biased_variances1 = np.zeros(R) 
s_unbiased_variances1 = np.zeros(R)
s_biased_variances2 = np.zeros(R) 
s_unbiased_variances2 = np.zeros(R) 

s_mom1 = np.zeros(R)
s_mom2 = np.zeros(R)

s_mle1 = np.zeros(R)
s_mle2 = np.zeros(R)

s_conf_int_w_std_1 = np.zeros([R,2])
s_conf_int_wo_std_1 = np.zeros([R,2])
s_conf_int_w_std_2 = np.zeros([R,2])
s_conf_int_wo_std_2 = np.zeros([R,2])


for i in range(R):
    samples1[i, :] = random_sample(population_1, N)
    samples2[i, :] = random_sample(population_2, N)

    s_means1[i] = calc_sample_mean(samples1[i])
    s_means2[i] = calc_sample_mean(samples2[i])

    s_biased_variances1 = calc_sample_variance(samples1[i], unbiased=False)
    s_unbiased_variances1 = calc_sample_variance(samples1[i])
    s_biased_variances2 = calc_sample_variance(samples2[i], unbiased=False)
    s_unbiased_variances2 = calc_sample_variance(samples2[i])

    s_mom1[i] = estimate_k_mom(samples1[i])
    s_mom2[i] = estimate_k_mom(samples2[i])

    s_mle1[i] = estimate_k_mle(samples1[i])
    s_mle2[i] = estimate_k_mle(samples2[i])

    s_conf_int_w_std_1[i, :] = calc_conf_int_mean(samples1[i], 97, pop_std1)
    s_conf_int_wo_std_1[i, :] = calc_conf_int_mean(samples1[i], 97)
    s_conf_int_w_std_2[i, :] = calc_conf_int_mean(samples2[i], 97, pop_std2)
    s_conf_int_wo_std_2[i, :] = calc_conf_int_mean(samples2[i], 97)

# Calculate and print means of sample means, biased and unbiased sample
# variances, MoM and MLE estimates of parameter k and plot the histograms of
# sample means, k estimates using MoM and MLE for both populations.


mean_means1 = np.mean(s_means1)
mean_means2 = np.mean(s_means2)
print("The Mean of the Population 1's Samples Means: ", mean_means1)
print("The Mean of the Population 2's Samples Means: ", mean_means2)
print("")

mean_biasedvars1 = np.mean(s_biased_variances1)
mean_unbiasedvars1 = np.mean(s_unbiased_variances1)
mean_biasedvars2 = np.mean(s_biased_variances2)
mean_unbiasedvars2 = np.mean(s_unbiased_variances2)
print("The Mean of the Population 1's Biased Samples Variances: ", mean_biasedvars1)
print("The Mean of the Population 1's Unbiased Samples Variances: ", mean_unbiasedvars1)
print("The Mean of the Population 2's Biased Samples Variances: ", mean_biasedvars2)
print("The Mean of the Population 2's Unbiased Samples Variances: ", mean_unbiasedvars2)
print("")

mean_mom1 = np.mean(s_mom1)
mean_mom2 = np.mean(s_mom2)
mean_mle1 = np.mean(s_mle1)
mean_mle2 = np.mean(s_mle2)

print("The Mean of the Population 1's 'k' Estimates with MoM: ", mean_mom1)
print("The Mean of the Population 2's 'k' Estimates with MoM: ", mean_mom2)
print("The Mean of the Population 1's 'k' Estimates with MLE: ", mean_mle1)
print("The Mean of the Population 2's 'k' Estimates with MLE: ", mean_mle2)
print("")

plt.figure()
plt.hist(s_means1, bins=50, alpha=0.5, label='Sample Means of Population 1')
plt.hist(s_means2, bins=50, alpha=0.5, label='Sample Means of Population 2')
plt.legend()

plt.figure()
plt.hist(s_mom1, bins=50, alpha=0.5, label="MoM Estimates of Population 1's Samples")
plt.hist(s_mom2, bins=50, alpha=0.5, label="MoM Estimates of Population 2's Samples")
plt.legend()

plt.figure()
plt.hist(s_mle1, bins=50, alpha=0.5, label="MLE Estimates of Population 1's Samples")
plt.hist(s_mle2, bins=50, alpha=0.5, label="MLE Estimates of Population 2's Samples")
plt.legend()

plt.show()

# Calculate and print the ratio of confidence intervals computed with and
# without using the population standard deviation that contains the population
# mean for both populations.

def is_mean_in_interval(mean,lower_bounds, upper_bounds):
   return (np.logical_and((lower_bounds <= mean), (mean <= upper_bounds )))

def find_bounds(interval):
    lowers = interval[:,0]
    uppers = interval[:,1]
    return lowers, uppers

lower_w1 , upper_w1 = find_bounds(s_conf_int_w_std_1)
lower_wo1 , upper_wo1 = find_bounds(s_conf_int_wo_std_1)
lower_w2 , upper_w2 = find_bounds(s_conf_int_w_std_2)
lower_wo2 , upper_wo2 = find_bounds(s_conf_int_wo_std_2)

count_w1 = count_wo1 = count_w2 = count_wo2 = 0
for i in range(R):
    if (is_mean_in_interval(p_mean_1 ,lower_w1[i], upper_w1[i])):
        count_w1 += 1
    if (is_mean_in_interval(p_mean_1 ,lower_wo1[i], upper_wo1[i])):
        count_wo1 += 1
    if (is_mean_in_interval(p_mean_2 ,lower_w2[i], upper_w2[i])):
        count_w2 += 1
    if (is_mean_in_interval(p_mean_2 ,lower_wo2[i], upper_wo2[i])):
        count_wo2 += 1

ratio_w1 = ( count_w1 / R)
ratio_wo1 = ( count_wo1 / R)
ratio_w2 = ( count_w2 / R)
ratio_wo2 = ( count_wo2 / R)

print("The Ratio of Confidence Intervals that contain Population Mean WITH using Population_Std (Population 1) : ", ratio_w1)
print("The Ratio of Confidence Intervals that contain Population Mean WITHOUT using Population_Std (Population 1) : ", ratio_wo1)
print("The Ratio of Confidence Intervals that contain Population Mean WITH using Population_Std (Population 2) : ", ratio_w2)
print("The Ratio of Confidence Intervals that contain Population Mean WITHOUT using Population_Std (Population 2) : ", ratio_wo2)

print('*'*50)
# Collect a sample of length 100000*25 from both populations, calculate and
# print their sample means, biased and unbiased sample variances, MoM and MLE
# estimates of parameter k and confidence intervals with and without using the
# population standard deviation.
print("-- NEW LARGE SAMPLES --")
print("")
new_size =  R * N
new_sample1 = random_sample(population_1, new_size)
new_sample2 = random_sample(population_2, new_size)


new_mean1 = calc_sample_mean(new_sample1)
new_mean2 = calc_sample_mean(new_sample2)

print("Sample of Population 1's Mean :", new_mean1)
print("Sample of Population 2's Mean :", new_mean2)
print("")


new_b_var1 = calc_sample_variance(new_sample1, unbiased=False)
new_unb_var1 = calc_sample_variance(new_sample1)
new_b_var2 = calc_sample_variance(new_sample2, unbiased=False)
new_unb_var2 = calc_sample_variance(new_sample2)

print("Biased Variance of Population 1's Sample :", new_b_var1)
print("Unbiased Variance of Population 1's Sample :", new_unb_var1)
print("Biased Variance of Population 2's Sample :", new_b_var2)
print("Unbiased Variance of Population 2's Sample :", new_unb_var2)
print("")

new_mom1 = estimate_k_mom(new_sample1)
new_mom2 = estimate_k_mom(new_sample2)

new_mle1 = estimate_k_mle(new_sample1)
new_mle2 = estimate_k_mle(new_sample2)

print("MoM Estimate of 'k' (Population 1's Sample):", new_mom1)
print("MoM Estimate of 'k' (Population 2's Sample):", new_mom2)
print("MLE Estimate of 'k' (Population 1's Sample):", new_mle1)
print("MLE Estimate of 'k' (Population 2's Sample):", new_mle2)
print("")

# Calculate confidence intervals for new samples
new_confint_w_std1 = calc_conf_int_mean(new_sample1, conf_lvl, pop_std1)
new_confint_wo_std1 = calc_conf_int_mean(new_sample1, conf_lvl)
new_confint_w_std2 = calc_conf_int_mean(new_sample2, conf_lvl, pop_std2)
new_confint_wo_std2 = calc_conf_int_mean(new_sample2, conf_lvl)

print("Confidence Interval with Pop_Std of Population 1's Sample :", new_confint_w_std1)
print("Confidence Interval without Pop_Std of Population 1's Sample :", new_confint_wo_std1)
print("Confidence Interval with Pop_Std of Population 2's Sample:", new_confint_w_std2)
print("Confidence Interval without Pop_Std of Population 2's Sample :", new_confint_wo_std2)

