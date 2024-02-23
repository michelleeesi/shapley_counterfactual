# %%
import numpy as np
import time
import pandas as pd
from collections.abc import Iterable
import itertools
from itertools import chain, combinations
from array import array
import copy
import random
import scipy.stats as st
from scipy.stats import norm
import math
import statistics
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import csv
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler



# %%
# Each data point will be cast into a data item to save properties such as power in the coalition
class DataItemMC:
    def __init__(self, ind):
        self.index = ind
        self.ev = []
        self.std = 5000
        self.mean = -1000
        self.ran_mean = math.inf
        self.power = []
        self.coa_dict = {}

    def __str__(self):
        return f"x_{self.index} has ev {self.mean} with stdev {self.std}"

# %%
# Brute force algorithm to find the optimal subset to shift from A to B
def BFSubsetCF(A, B, all_train_data, target_data, owners, test, util):
    subsets_dict = {}
    A_data = owners[A]
    B_data = owners[B]
    if (len(A_data)==1): return A_data

    # checking all subsets in increasing order of size
    for i in range(1, len(A_data)):
        allSub = list(itertools.combinations(A_data, i))
        for subset in allSub:
            print(subset)
            if subset not in subsets_dict:
                subsets_dict.update({str(subset): 1})
                newA = A_data.copy()
                for el in subset:
                    newA.remove(el)

                # updating dictionaries of owned data per owner
                owners.update({A: newA})
                owners.update({B: list(np.append(B_data, list(subset)))})
                print("A: " + str(owners[A]))
                print("B: " + str(owners[B]))

                # checking difference
                difference = diffShapleyCF(all_train_data, target_data, test, owners, A, B, util)
                print("difference: " +str(difference))
                owners.update({A: A_data})
                owners.update({B: B_data})

                # stop if difference is negative
                if difference < 0:
                    return subset

# %%
# Monte Carlo algorithm to find the optimal subset to shift from A to B
def MCSubsetCF(A, B, all_train_data, target_data, owners, ss_size, delta, test,util):

    # getting initial MC difference between A and B, stop if already negative
    diff = diffShapleyMCCF({}, all_train_data, target_data, owners, A, B, delta, 50, test, util)
    if diff[0] < 0:
        return [[], 0, True, diff[0]]
    
    t1 = time.time()
    if (len(owners[A])==1): return [owners[A], 0, True, -1]
    flipped = False

    A_data = owners[A]
    B_data = owners[B]
    perms = 0
    permutations_dict = {}

    # checking subsets in increasing order of size
    for i in range(1, len(A_data)):
        allSub = list(itertools.combinations(A_data, i))
        for subset in allSub:
            print(subset)
            newA = A_data.copy()
            for el in subset:
                newA.remove(el)
            owners.update({A: newA})
            owners.update({B: list(np.append(B_data, list(subset)))})
#           
            # sample coalitions by generating permutations until Shapley difference is negative and has stopped changing within a tolerance bound
            est_mean, added_perms, permutations_dict = diffShapleyMCCF(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
            perms += added_perms
            owners.update({A: A_data})
            owners.update({B: B_data})
            print(est_mean)
            t2 = time.time()

            # stop if time limit is reached
            if (t2-t1 > 7200):
                return subset, perms, False, 1
            
            # verify if shifted subset actually has negative difference
            if est_mean < 0: 
                owners.update({A: newA})
                owners.update({B: list(np.append(B_data, list(subset)))})
                diff = diffShapleyMCCF(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
                if (diff[0] < 0):
                    flipped = True
                else:
                    flipped = False
                owners.update({A: A_data})
                owners.update({B: B_data})
                return subset, perms, flipped, diff[0]
    return [None,perms, False, 1]

# %%
# draw permutations for the Monte Carlo estimation of Shapley
def monteCarloCF(curr_est, permutations_dict, owners, all_train_data, target_data, A, B, step_size, test, util):
    if (len(owners[A])==1): return [[-1], permutations_dict]
    A_data = owners[A]
    B_data = owners[B]

    # get step_size permutations every round
    for i in range(step_size):

        # draw a permutation, get coalition from permutation
        permutation = np.random.permutation(list(owners.keys()))
        S = findCoalition(permutation, A, B)
        permString = listToString(S)

        P_data = []

        # get data from coalition
        if permString not in permutations_dict:
            P_data = [item for key in S for item in owners.get(key, [])]
            permutations_dict.update({permString: P_data})
        else:
            P_data = permutations_dict[permString]

        # transfer data ownership between A and B
        owners.update({A: list(np.append(owners[A], P_data))})
        owners.update({B: list(np.append(owners[B], P_data))})

        # get utility of the data transfer
        curr_est.append((1/(len(owners)-len(S)-1))*0.5*len(owners.keys())*diffUtilityCF(all_train_data, target_data, test, owners, A, B, util))

        owners.update({A: A_data})
        owners.update({B: B_data})
    return curr_est, permutations_dict
        

# %%
# SV-Exp algorithm to find the optimal subset to shift from A to B
def greedyCF(A, B, all_train_data, target_data, owners, ss_size, delta, test, util):
   
   # get initial difference between A and B, stop if already negative
    diff = diffShapleyMCCF({}, all_train_data, target_data, owners, A, B, delta, 50, test, util)
    if diff[0] < 0:
        return [[], 0, True, diff[0]]
    tbeg = time.time()
    if (len(owners[A])==1): return [owners[A], 0, True, -1]
    flipped = False
    A_data = owners[A]
    B_data = owners[B]
    perms = 0
    subset = []
    
    t0=time.time()
    
    # preprocess the data and cast all single items into Data Items
    allTest, other_owner_data = processGreedy(owners, A, B)
    
    # get top k items to shift from A to B
    topk = topKBaselineCF(diff[2], A, B, owners, all_train_data, target_data, other_owner_data, allTest, ss_size, 1, delta, test, util)
    
    topk_items = [i.index for i in topk[0]]
    
    perms += topk[1]

    permutations_dict = topk[2]
    
    t1 = time.time()
    print(f"The time it took for first iteration was: {t1-t0}")
    
    # for each item in top k
    for i in range(1, len(topk_items)+1):
        
        # add the top 1 item to the subset, update A and B with data transfer
        subset.append(topk_items[0]) # try with 1, then 2, etc.
        print(subset)
        newA = A_data.copy()
        for el in subset:
            newA.remove(el)
        owners.update({A: newA})
        owners.update({B: list(np.append(B_data, list(subset)))})
        
        t2=time.time()

        # estimate the differential Shapley until tolerance is reached
        est_mean, added_perms, permutations_dict = diffShapleyMCCF(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
        perms += added_perms

        owners.update({A: A_data})
        owners.update({B: B_data})
        
        t3=time.time()
        print(f"The time it took for moving subset was: {t3-t2}")
        tend = time.time()

        # time out if it takes too long
        if(tend-tbeg > 7200):
            return subset, perms, False, 1
        
        # if the differential Shapley is negative, verify if subset transfer actually flipped Shapleys and return subset
        if est_mean < 0: 
            owners.update({A: newA})
            owners.update({B: list(np.append(B_data, list(subset)))})
            diff = diffShapleyMCCF(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
            if (diff[0] < 0):
                flipped = True
            else:
                flipped = False
            owners.update({A: A_data})
            owners.update({B: B_data})
            return subset, perms, flipped, diff[0]
        
        t4=time.time()

        # run the top k algorithm again with the top 1 item removed to get the next best item to shift
        topk = topKBaselineCF(permutations_dict, A, B, owners, all_train_data, target_data, other_owner_data, topk[0][1:], ss_size, 1, delta, test, util)
        
        topk_items = [i.index for i in topk[0]]
        
        perms += topk[1]

        permutations_dict = topk[2]
        
        t5=time.time()
        print(f"The time it took for ranking next item was: {t5-t4}")
        
    return [0,0]

# %%
# Thompson sampling implementation for the top k algorithm
def topKBaselineCF(permutations_dict, A, B, owners, all_train_data, target_data, other_owner_data, items, step_size, k, delta, test, util):
    if (len(owners[A])==1): return [items, 0, {}]

    A_data = owners[A].copy()
    B_data = owners[B].copy()

    total_perms = 0

    # first round of Thompson sampling
    if len(items[0].power)==0:
        # sample twice to get mean
        for i in range(2):
            permutation = np.random.permutation(list(owners.keys()))
            S = findCoalition(permutation, A, B)
            permString = listToString(S)

            P_data = []

            if permString not in permutations_dict:
                P_data = [item for key in S for item in owners.get(key, [])]
                permutations_dict.update({permString: P_data})
            else:
                P_data = permutations_dict[permString]
            coaString = listToString(P_data)

            # for each item, sample the differential utility post transferring that one item and add to list of sample powers of that data item
            for i in items:
                owners.update({A: list(set(np.append(owners[A], P_data)))})
                owners.update({B: list(set(np.append(owners[B], P_data)))})
                owners[A].remove(i.index)
                owners[B].append(i.index)
                if coaString not in i.coa_dict:
                    i.coa_dict.update({coaString: 0.5*len(owners)*(1/(len(owners)-len(S)-1))*diffUtilityCF(all_train_data, target_data, test, owners, A, B, util)})
                i.power.append(i.coa_dict[coaString])
                owners.update({A: A_data})
                owners.update({B: B_data})

        # take the mean and stdev of the two samples for all items
        for i in items:
            i.mean = np.mean(i.power)
            i.std = np.std(i.power, ddof=1)
    
    # sort the items by mean power
    sorted_items = sorted(items, key=lambda x:x.mean)

    # while the first and second item are not significantly different, sample more
    while not findSig(sorted_items, 1, delta):

        # for each sample, sample randomly from the gaussian distribution of the power of that item
        for i in sorted_items:
            ran_mean = np.random.normal(loc=np.mean(i.power), scale=np.std(i.power, ddof=1)/math.sqrt(len(i.power)))
            i.ran_mean = ran_mean
        
        # sort the items by sampled power
        thompson_sorted = sorted(items, key=lambda x:x.ran_mean)
        i = thompson_sorted[0]

        # for the item with the best mean, sample more
        for r in range(step_size):
            total_perms += 1
            permutation = np.random.permutation(list(owners.keys()))

            S = findCoalition(permutation, A, B)
            permString = listToString(S)

            P_data = []

            if permString not in permutations_dict:
                P_data = [item for key in S for item in owners.get(key, [])]
                permutations_dict.update({permString: P_data})
            else:
                P_data = permutations_dict[permString]
            coaString = listToString(P_data)
            
            owners.update({A: list(set(np.append(owners[A], P_data)))})
            owners.update({B: list(set(np.append(owners[B], P_data)))})
            owners[A].remove(i.index)
            owners[B].append(i.index)

            if coaString not in i.coa_dict:
                i.coa_dict.update({coaString: 0.5*len(owners)*(1/(len(owners)-len(S)-1))*diffUtilityCF(all_train_data, target_data, test, owners, A, B, util)})
            i.power.append(i.coa_dict[coaString])
            # samples.append(i.coa_dict[coaString])
            owners.update({A: A_data})
            owners.update({B: B_data})
            
        i.mean = np.mean(i.power)
        i.std = np.std(i.power, ddof=1)

        # update gaussian for best item, resort
        sorted_items = sorted(sorted_items, key=lambda x:x.mean)

        # print("1st place mean: ", sorted_items[0].mean)
        # print("2nd place mean: ", sorted_items[1].mean)
    return [sorted_items, total_perms, permutations_dict]

# %%
# Monte Carlo estimation of Shapley difference, stop when tolerance is reached
def diffShapleyMCCF(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util):
    total_samples = []
    perms = 0

    change = math.inf
    old_mean = 0.0000000001

    while (change > 0.01):
        perms += ss_size
        total_samples, permutations_dict = monteCarloCF(total_samples, permutations_dict, owners, all_train_data, target_data, A, B, ss_size, test, util)
        est_mean = np.mean(total_samples)
        est_std = np.std(total_samples, ddof=1)
        print("mean: ", est_mean)
        change = abs((est_mean-old_mean)/old_mean)
        old_mean = est_mean
        print("change: ",change)

    return est_mean, perms, permutations_dict

# %%
# assign data items to owners based on parameters
def preprocessCF(n_owners, ds_size, total_size, train, test, owner_data_dist, owner_size_dist, starting_diff, util):

    train = train.sort_values(by=random.choice(train.columns[:-1].tolist()))
    num_rows, num_cols = train.shape
    column = train.columns[1]

    # Compute the index range and generate k uniformly distributed values
    uniform_indices = np.linspace(0, num_rows-1, num=n_owners, dtype=int)
    dataset = set()
    owners = {}
        
    for i in range(n_owners):
        
        # initializing this owner's data
        owner_data = []
        
        if (owner_size_dist == 'uniform'):
            # every data owner's size gets sampled from uniform distribution
            size = random.randint(1, num_rows-1)

            if (owner_data_dist == 'uniform'):
                owner_data = random.sample(range(0, num_rows), size)
                
            # if (owner_data_dist == 'clustered'):
            #     gmm = estimate_gaussian_mixture(train, column, num_components=4)
            #     sampled_rows = sample_rows_from_gaussian_component(train, gmm, column, component_index=random.randint(0,3), num_samples=size)
            #     owner_data = list(sampled_rows)
                
            
        if (owner_size_dist == 'zipfian'):
            # every data owner's size get sampled from zipfian distribution (lots will have very few, some have very large amt)
            size = int(zipfian(1.5, 1, num_rows-1, size=1))

            if (owner_data_dist == 'uniform'):
                owner_data = random.sample(range(0, num_rows), size)
                    
            # if (owner_data_dist == 'clustered'):
            #     gmm = estimate_gaussian_mixture(train, column, num_components=4)
            #     sampled_rows = sample_rows_from_gaussian_component(train, gmm, column, component_index=random.randint(0,3), num_samples=size)
            #     owner_data = list(sampled_rows)
        
        owners.update({i: owner_data})
    
    for o in owners.values():
        dataset = dataset.union(set(o))
    
    dataset = list(dataset)
    
    
    lenA = 0

    # choose A and B such that A has a higher estimed Shapley
    while (lenA<=1):
        twoOwners = random.sample(list(owners.keys()), 2)
        difference = diffShapleyMCCF({},train, train.loc[dataset], owners, twoOwners[0], twoOwners[1], 0.05, 50, test, util)

        print(difference)
        
        if (difference[0] > 0):
            A = twoOwners[0]
            B = twoOwners[1]
        else:
            A = twoOwners[1]
            B = twoOwners[0]
        
        lenA = len(owners[A])


    return dataset, owners, A, B
        

# %%
# assign data owner sizes and data according to a Zipf distribution
def zipfian_assignment(n_owners, A_size, B_size, train):
    owners = {}
    dataset = set()
    num_rows, num_cols = train.shape
    
    for i in range(n_owners-2):
        size = int(zipfian(1.5, 1, num_rows-1, size=1))
        owners.update({i: random.sample(range(0, num_rows), size)})
    
    owners.update({n_owners-2: random.sample(range(0, num_rows), A_size)})
    owners.update({n_owners-1: random.sample(range(0, num_rows), B_size)})

    for o in owners.values():
        dataset = dataset.union(set(o))
    
    dataset = list(dataset)

    return dataset, owners, n_owners-2, n_owners-1          
            

# # %%
# # assign data ownership according to a gaussian mixture model
# def estimate_gaussian_mixture(dataframe, column_name, num_components=4, random_state=42):
#     # Extract the target column as a 1D array
#     target_data = dataframe[column_name].values.reshape(-1, 1)

#     # Create and fit the Gaussian Mixture Model
#     gmm = GaussianMixture(n_components=num_components, random_state=random_state)
#     gmm.fit(target_data)
#     return gmm

# # %%
# # sample rows from a gaussian mixture model according to component
# def sample_rows_from_gaussian_component(dataframe, gmm, column, component_index, num_samples, random_state=42):
#     np.random.seed(random_state)
    
#     # Assign each row to a GMM component
#     component_assignments = gmm.predict(dataframe[column].values.reshape(-1, 1))
#     print(component_assignments)

#     # Get the indices of rows assigned to the specified component
#     component_indices = np.where(component_assignments == component_index)[0]

#     # Randomly sample rows from the specified component
#     sampled_indices = np.random.choice(component_indices, size=num_samples, replace=False)

#     return sampled_indices

# %%
# brute force method of calculating a Differential Shapley without any Monte Carlo estimation
def diffShapleyCF(all_train_data, target_data, test, owners, A, B, util):
    A_data = owners[A]
    B_data = owners[B]
    list_owners = list(owners.keys())
    list_owners.remove(A)
    list_owners.remove(B)
    coalitions = powerset(list_owners)
    total = 0

    # go through all the powersets in order of size
    for S in coalitions:
        P_data = []
        for o in S:
            P_data.append(owners[o])
        P_data = list(flatten(P_data))
        owners.update({A: list(np.append(owners[A], P_data))})
        owners.update({B: list(np.append(owners[B], P_data))})
        total += 1/((len(S)+1)*math.comb(len(list(owners.keys()))-1,len(S)+1)) * diffUtilityCF(all_train_data, target_data, test, owners, A, B, util)
        owners.update({A: A_data})
        owners.update({B: B_data})
    return total

# %%
# see if the first item in the Thompson sampling sorted list is statistically different from the second item in the Thompson sampling sorted list
def findSig(tups, k, delta):
    if len(tups)==1: return True
    first = tups[0]
    second = tups[1]
    z = norm.ppf(1-(delta/2))
    n = len(tups[0].power)
    if n==0: return False

    mean_diff = second.mean-first.mean
    lb = mean_diff-z*np.sqrt(first.std**2+second.std**2)/np.sqrt(n)
    ub = mean_diff+z*np.sqrt(first.std**2+second.std**2)/np.sqrt(n)
    if (lb < 0 < ub):
        return False
    return True

# Generate zipfian random variables
def zipfian(a, minimum, maximum, size=None):
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(minimum, maximum+1) # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.random.choice(v, size=size, replace=True, p=p)

# Find a coalition given a permutation and two owners A and B
def findCoalition(permutation, o1, o2):
    coa = []
    for data_owner in permutation:
        if (data_owner!=o1 and data_owner!=o2):
            coa.append(data_owner)
        else:
            return sorted(coa)
    return sorted(coa)

def listToString(perm):
    return ' '.join(str(x) for x in perm)

# helper function for flattening array of arrays
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# used for creating all possible subsets of A data to shift over to B data
def powerset(s):
    powerset = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
    return powerset

def wrap_testBatchCF(args):
    return testBatchCF(*args)

# %%
# utility of a dataset is difference in error when trained with vs without that dataset
def utilityKDE(all_train_data, target_data, test_data, data_owner_data):

    data_owner_data = [int(x) for x in data_owner_data]

    feature = 11
    data = target_data.loc[data_owner_data]
    data_column = data.iloc[:, feature].values
    data_column = [x for x in data_column]
    
    target_column = target_data.iloc[:,feature].values
    target_column = [x for x in target_column]
    
    test_column = test_data.iloc[:,feature].values

    ground_truth_grid, ground_truth_density = kernel_density_estimation(target_column)

    sample_estimation_grid, sample_estimation_density = kernel_density_estimation(data_column)

    errors = []
    for data_item in test_column:
        gt_density = np.interp(data_item, ground_truth_grid, ground_truth_density)
        errors.append(abs(gt_density-np.interp(data_item, sample_estimation_grid, sample_estimation_density))/gt_density)

    return -np.mean(errors)

# %%
def diffUtilityCF(all_train_data, target_data, test_data, owners, o1, o2, util='decision_tree'):
    return utilityFunc(all_train_data, target_data, test_data, owners[o1], util) - utilityFunc(all_train_data, target_data, test_data, owners[o2],util)

# %%
def utilityFunc(all_train_data, target_data, test_data, data_owner_data, util):
    if util == 'decision_tree':
        return utilityDecisionTree(all_train_data, target_data, test_data, data_owner_data)
    elif util == 'kde':
        return utilityKDE(all_train_data, target_data, test_data, data_owner_data)
    elif util == 'knn':
        return utilityKNN(all_train_data, target_data, test_data, data_owner_data)
    elif util == 'log_reg':
        return utilityLR(all_train_data, target_data, test_data, data_owner_data)

# %%
# utility of a dataset is difference in error when trained with vs without that dataset
def utilityKNN(all_train_data, target_data, test_data, data_owner_data):
    
    data_owner_data = [int(x) for x in data_owner_data]

    without_owner_data = target_data.drop(data_owner_data)
    
    target_data_features = target_data.drop("target", axis=1).values
    without_owner_data_features = without_owner_data.drop("target",axis=1).values
    test_data_features = test_data.drop("target",axis=1).values
    
    target_data_labels = target_data["target"].values
    without_owner_data_labels = without_owner_data["target"].values
    test_data_labels = test_data["target"].values
    
    
    print(len(target_data))
    print(len(without_owner_data))
    

    # Initialize KNN models
    large_knn = KNeighborsClassifier(n_neighbors=30)
    small_knn = KNeighborsClassifier(n_neighbors=30)

    # Fit the models
    large_knn.fit(target_data_features, target_data_labels)
    small_knn.fit(without_owner_data_features, without_owner_data_labels)

    # Predict labels
    large_pred_labels = large_knn.predict(test_data_features)
    small_pred_labels = small_knn.predict(test_data_features)

    # Calculate loss (Mean Squared Error) for both models
    large_loss = mean_squared_error(test_data_labels, large_pred_labels)
    small_loss = mean_squared_error(test_data_labels, small_pred_labels)

    return large_loss - small_loss

# %%
# decision tree utility function
def utilityDecisionTree(all_train_data, target_data, test_data, data_owner_data):
    data_owner_data = [int(x) for x in data_owner_data]

    data = target_data.loc[data_owner_data]
    data_features = data.drop("target", axis=1).values
    data_labels = data["target"].values

    test_data_features = test_data.drop("target", axis=1).values
    test_data_labels = test_data["target"].values

    if (len(set(data_labels))==1):
        small_pred_probs = [data_labels[0]]*len(test_data)
    else:
        small_logistic_regression = DecisionTreeClassifier(random_state=19)
        small_logistic_regression.fit(data_features, data_labels)
        small_pred_probs = small_logistic_regression.predict_proba(test_data_features)
    
    small_loss = -log_loss(test_data_labels, small_pred_probs)
    return small_loss



# %%
# logistic regression utility function
def utilityLR(all_train_data, target_data, test_data, data_owner_data):
    data_owner_data = [int(x) for x in data_owner_data]

    data = target_data.loc[data_owner_data]
    #print(data)

    data_features = data.drop("target", axis=1).values

    data_labels = data["target"].values
    test_data_features = test_data.drop("target", axis=1).values 

    test_data_labels = test_data["target"].values
    

    if (len(set(data_labels))==1):
        if (data_labels[0]==1):
            small_pred_probs = np.array([[0,1] for i in range(len(test_data_labels))])
        else:
            small_pred_probs = np.array([[1,0] for i in range(len(test_data_labels))])
    else: 
        small_logistic_regression = LogisticRegression(max_iter=1000,solver="liblinear",random_state=19)
        small_logistic_regression.fit(data_features, data_labels)
        small_pred_probs = small_logistic_regression.predict_proba(test_data_features)

    small_loss = -1*log_loss(test_data_labels, small_pred_probs)
    return small_loss

# %%
def kernel_density_estimation(data, bandwidth=0.2, kernel='gaussian', x_grid=None):
    # Fit the CF model
    data = np.array(data)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(data[:,np.newaxis])

    # If x_grid is not provided, create a default grid of values
    if x_grid is None:
        x_grid = np.linspace(data.min(), data.max(), 1000)

    # Compute the log-density values for the grid of values
    log_dens = kde.score_samples(x_grid[:,np.newaxis])

    # Exponentiate the log-densities to obtain the density values
    density = np.exp(log_dens)

    return x_grid, density
    

# %%
# shuffle and split the data into training and test sets
def shuffle_and_split(data):    
    X_train, X_test = np.split(data.sample(frac=1, random_state=1729), [int(0.8 * len(data))])
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, X_test

# %%
# master function for comparing all counterfactuals for a given distribution of data owners
# compares MC and SV-Exp
def compareAllCF(all_train_data, test_data, target_data_size, num_owners, num_tries, start_diff, data_dist, size_dist, util):

    dataset, owners, A, B = preprocessCF(num_owners, target_data_size, len(all_train_data), all_train_data, test_data, owner_data_dist= data_dist, owner_size_dist = size_dist, starting_diff=start_diff, util=util)
    while (len(owners[A])==1 and len(owners[B])==1):
        dataset, owners, A, B = preprocessCF(num_owners, target_data_size, len(all_train_data), all_train_data, test_data, owner_data_dist=data_dist, owner_size_dist = size_dist, starting_diff=start_diff, util=util)
    print(dataset)
    print(owners)
    print(A)
    print(owners[A])
    print(B)
    print(owners[B])

    BFSol = None
#     BFSol = BFSubset(A, B, data = dataset, owners = owners, mat = mat)
    MCSamples = []
    greedyBaselineSamples = []
    MCAnswers = []
    greedyBaselineAnswers = []
    MCCheck = []
    greedyBaselineCheck = []
    MCTimes = []
    GRBSTimes = []
    MCDiff = []
    GRBSDiff = []
    
    for i in range(num_tries):

        # SV-Exp Greedy Algorithm
        grbstimestart = time.time()
        greedySolBaseline = greedyCF(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
        grbstimeend = time.time()
        grbsTime = grbstimeend - grbstimestart
        GRBSTimes.append(grbsTime)
        greedyBaselineCheck.append(greedySolBaseline[2])
        greedyBaselineSamples.append(greedySolBaseline[1])
        greedyBaselineAnswers.append(len(greedySolBaseline[0]))
        GRBSDiff.append(greedySolBaseline[3])
        
        # MC Algorithm
        mctimestart = time.time()
        MCSol = MCSubsetCF(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
        mctimeend = time.time()
        mcTime = mctimeend - mctimestart
        MCTimes.append(mcTime)
        MCCheck.append(MCSol[2])
        MCSamples.append(MCSol[1])
        MCAnswers.append(len(MCSol[0]))
        MCDiff.append(MCSol[3])

    return len(dataset), len(set(owners[A])), len(set(owners[B])), BFSol, MCSamples, greedyBaselineSamples, MCAnswers, greedyBaselineAnswers, MCTimes, GRBSTimes, MCCheck, greedyBaselineCheck, MCDiff, GRBSDiff

# %%
# compares SV-Exp and MC for a given size distribution between A and B for zipfian-distributed data owners
# %%
def compareZipfian(all_train_data, test_data, A_size, B_size, num_owners, data_dist, size_dist, util,ds_name, csv_file_path):
    
    BFSol = None
    dataset_len = []
    #     BFSol = BFSubset(A, B, data = dataset, owners = owners, mat = mat)
    MCSamples = []
    greedyBaselineSamples = []
    MCAnswers = []
    greedyBaselineAnswers = []
    MCCheck = []
    greedyBaselineCheck = []
    MCTimes = []
    GRBSTimes = []
    
    # for i in range(10):
    dataset, owners, A, B = zipfian_assignment(num_owners, A_size, B_size, all_train_data)
    dataset_len.append(len(dataset))
    print(dataset)
    print(owners)
    print(A)
    print(owners[A])
    print(B)
    print(owners[B])

    grbstimestart = time.time()
    greedySolBaseline = greedyCF(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
    grbstimeend = time.time()
    grbsTime = grbstimeend - grbstimestart
    GRBSTimes.append(grbsTime)
    greedyBaselineCheck.append(greedySolBaseline[2])
    greedyBaselineSamples.append(greedySolBaseline[1])
    greedyBaselineAnswers.append(len(greedySolBaseline[0]))
            

    mctimestart = time.time()
    MCSol = MCSubsetCF(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
    mctimeend = time.time()
    mcTime = mctimeend - mctimestart
    MCTimes.append(mcTime)
    MCCheck.append(MCSol[2])
    MCSamples.append(MCSol[1])
    MCAnswers.append(len(MCSol[0]))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([len(dataset), A_size, B_size, MCSol[1], greedySolBaseline[1], len(MCSol[0]), len(greedySolBaseline[0]), mcTime, grbsTime,MCSol[2],greedySolBaseline[2],MCSol[3],greedySolBaseline[3],num_owners,data_dist, size_dist,util,ds_name])

    return np.mean(dataset_len), A_size, B_size, BFSol, np.mean(MCSamples), np.mean(greedyBaselineSamples), np.mean(MCAnswers), np.mean(greedyBaselineAnswers), np.mean(MCTimes), np.mean(GRBSTimes), np.mean(MCCheck)*100, np.mean(greedyBaselineCheck)*100

def wrap_compareZipfian(args):
    return compareZipfian(*args)       
    

# %%
# assign clusters for the natural data distribution experiment
def assignClusters(df, cat):
    unique_categories = df[cat].unique()

    # Create a dictionary to store the lists of indices
    owner_data = {}
    category_indices = {}

    # Iterate through unique values and extract the indices
    for i in range(len(unique_categories)):
        category = unique_categories[i]
        owner_data[i] = df.index[df[cat] == category].tolist()
        category_indices[i] = category

    return owner_data, category_indices

# compare SV-Exp and MC for a given natural data distribution where data owners own different categories of data
def compareNatural(all_train_data, test_data, cat, A, B, data_dist, size_dist, util, ds_name, csv_file_path, trial):
    
    BFSol = None
    #     BFSol = BFSubset(A, B, data = dataset, owners = owners, mat = mat)
    MCSamples = []
    greedyBaselineSamples = []
    MCAnswers = []
    greedyBaselineAnswers = []
    MCCheck = []
    greedyBaselineCheck = []
    MCTimes = []
    GRBSTimes = []
    
    owners, categories = assignClusters(all_train_data, cat)

    all_train_data = all_train_data.drop(cat, axis=1)
    test_data = test_data.drop(cat, axis=1)

    grbstimestart = time.time()
    greedySolBaseline = greedyCF(A, B, all_train_data, all_train_data, owners, 20, 0.05, test_data, util)
    grbstimeend = time.time()
    grbsTime = grbstimeend - grbstimestart
    GRBSTimes.append(grbsTime)
    greedyBaselineCheck.append(greedySolBaseline[2])
    greedyBaselineSamples.append(greedySolBaseline[1])
    greedyBaselineAnswers.append(len(greedySolBaseline[0]))
            

    mctimestart = time.time()
    MCSol = MCSubsetCF(A, B, all_train_data, all_train_data, owners, 20, 0.05, test_data, util)
    mctimeend = time.time()
    mcTime = mctimeend - mctimestart
    MCTimes.append(mcTime)
    MCCheck.append(MCSol[2])
    MCSamples.append(MCSol[1])
    MCAnswers.append(len(MCSol[0]))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([categories[A], categories[B],len(owners[A]), len(owners[B]), MCSol[1], greedySolBaseline[1], len(MCSol[0]), len(greedySolBaseline[0]), mcTime, grbsTime,MCSol[2],greedySolBaseline[2],MCSol[3],greedySolBaseline[3],len(owners),data_dist, size_dist, util, ds_name, trial])

    return BFSol, np.mean(MCSamples), np.mean(greedyBaselineSamples), np.mean(MCAnswers), np.mean(greedyBaselineAnswers), np.mean(MCTimes), np.mean(GRBSTimes), np.mean(MCCheck)*100, np.mean(greedyBaselineCheck)*100
    

def wrap_compareNatural(args):
    return compareNatural(*args)   
            

# %%
# cast individual data items into Data Item objects so that Thompson sampling can keep track of means and stdevs
def processGreedy(owners, A, B):
    other_owner_data = []
    for k in list(owners.keys()):
        if (k != A and k != B):
            other_owner_data.append(owners[k])
    other_owner_data = set(flatten(other_owner_data))
    allTest = []
    for i in list(set(owners[A])):
        allTest.append(DataItemMC(i))
    return allTest, other_owner_data

# %%
# Master comparison: Test SV-Exp vs MC for all uniform distributions of data over different data assignments
# Record times, samples, and answers for each trial
def testBatchCF(all_train_data, test_data, target_data_size, num_owners, num_iterations, num_trials, start_diff, data_dist, size_dist, csv_file_path,util, ds_name):
    totalMCSamples = []
    trueAnswers = []
    finalMCAnswers = []

    finalGreedyBaselineAnswers = []

    totalGreedyBaselineSamples = []
    totalMCTimes = []

    totalGRBSTimes = []

    datasetSizes = []

    ownerASizes = []
    ownerBSizes = []
    
    totalMCCheck = []
    totalGRBSCheck = []

    for i in range(num_trials):
        datasetSize, ownerASize, ownerBSize, correctSol, MCSamples, greedyBaselineSamples, mcSub, greedyBaselineSubset, MCTimes, GRBSTimes, MCCheck, GRBSCheck, MCDiff, GRBSDiff = compareAllCF(all_train_data, test_data, target_data_size, num_owners, num_iterations, start_diff, data_dist, size_dist, util)
        datasetSizes.append(datasetSize)
        ownerASizes.append(ownerASize)
        ownerBSizes.append(ownerBSize)
        totalMCSamples.append(np.mean(MCSamples))
        totalGreedyBaselineSamples.append(np.mean(greedyBaselineSamples))
#         trueAnswers.append(len(correctSol))
        finalMCAnswers.append(np.mean(mcSub))
        finalGreedyBaselineAnswers.append(np.mean(greedyBaselineSubset))
        totalMCCheck.append(MCCheck)
        totalGRBSCheck.append(GRBSCheck)
        totalMCTimes.append(np.mean(MCTimes))
        totalGRBSTimes.append(np.mean(GRBSTimes))
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([datasetSize, ownerASize, ownerBSize, MCSamples[0], greedyBaselineSamples[0], mcSub[0], greedyBaselineSubset[0], MCTimes[0], GRBSTimes[0],MCCheck[0],GRBSCheck[0],MCDiff[0], GRBSDiff[0],num_owners,data_dist, size_dist,util,ds_name])
    
    return datasetSizes, ownerASizes, ownerBSizes, trueAnswers, totalMCSamples, totalGreedyBaselineSamples, finalMCAnswers, finalGreedyBaselineAnswers, totalMCTimes, totalGRBSTimes, totalMCCheck, totalGRBSCheck, MCDiff, GRBSDiff

