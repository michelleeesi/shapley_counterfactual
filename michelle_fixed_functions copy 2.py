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
def BFSubsetKDE(A, B, all_train_data, target_data, owners, test, util):
    subsets_dict = {}
    A_data = owners[A]
    B_data = owners[B]
    if (len(A_data)==1): return A_data
    for i in range(1, len(A_data)):
        allSub = list(itertools.combinations(A_data, i))
        for subset in allSub:
            print(subset)
            if subset not in subsets_dict:
                subsets_dict.update({str(subset): 1})
                newA = A_data.copy()
                for el in subset:
                    newA.remove(el)
                owners.update({A: newA})
                owners.update({B: list(np.append(B_data, list(subset)))})
                print("A: " + str(owners[A]))
                print("B: " + str(owners[B]))
                difference = diffShapleyKDE(all_train_data, target_data, test, owners, A, B, util)
                print("difference: " +str(difference))
                owners.update({A: A_data})
                owners.update({B: B_data})
                if difference < 0:
                    return subset

# %%
def MCSubsetKDE(A, B, all_train_data, target_data, owners, ss_size, delta, test,util):
    diff = diffShapleyMCKDE({}, all_train_data, target_data, owners, A, B, delta, 50, test, util)
    if diff[0] < 0:
        return [[], 0, True, diff[0]]
    t1 = time.time()
    if (len(owners[A])==1): return [owners[A], 0, True, -1]
    flipped = False
    A_data = owners[A]
    B_data = owners[B]
    perms = 0
    permutations_dict = {}
    for i in range(1, len(A_data)):
        allSub = list(itertools.combinations(A_data, i))
        for subset in allSub:
            print(subset)
            total_samples = []
            newA = A_data.copy()
            for el in subset:
                newA.remove(el)
            owners.update({A: newA})
            owners.update({B: list(np.append(B_data, list(subset)))})
#           
            est_mean, added_perms, permutations_dict = diffShapleyMCKDE(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
            perms += added_perms

            owners.update({A: A_data})
            owners.update({B: B_data})
            print(est_mean)
            t2 = time.time()
            if (t2-t1 > 7200):
                return subset, perms, False, 1
            if est_mean < 0: 
                owners.update({A: newA})
                owners.update({B: list(np.append(B_data, list(subset)))})
                diff = diffShapleyMCKDE(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
                if (diff[0] < 0):
                    flipped = True
                else:
                    flipped = False
                # flipped = verifyShapley(A, B, all_train_data, target_data, owners, ss_size, delta, test,util)
                owners.update({A: A_data})
                owners.update({B: B_data})
                return subset, perms, flipped, diff[0]
    return [None,perms, False, 1]

# %%
def monteCarloKDE(curr_est, permutations_dict, owners, all_train_data, target_data, A, B, step_size, test, util):
    if (len(owners[A])==1): return [[-1], permutations_dict]
    A_data = owners[A]
    B_data = owners[B]
    for i in range(step_size):
        permutation = np.random.permutation(list(owners.keys()))
        S = findCoalition(permutation, A, B)
        permString = listToString(S)

        P_data = []

        if permString not in permutations_dict:
            P_data = [item for key in S for item in owners.get(key, [])]
            permutations_dict.update({permString: P_data})
        else:
            P_data = permutations_dict[permString]

        owners.update({A: list(np.append(owners[A], P_data))})
        owners.update({B: list(np.append(owners[B], P_data))})

        curr_est.append((1/(len(owners)-len(S)-1))*0.5*len(owners.keys())*diffUtilityKDE(all_train_data, target_data, test, owners, A, B, util))

        owners.update({A: A_data})
        owners.update({B: B_data})
    return curr_est, permutations_dict
        

# %%
def greedyKDE(A, B, all_train_data, target_data, owners, ss_size, delta, test, util):
    diff = diffShapleyMCKDE({}, all_train_data, target_data, owners, A, B, delta, 50, test, util)
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
    
    allTest, other_owner_data = processGreedy(owners, A, B)
    
    topk = topKBaselineKDE(diff[2], A, B, owners, all_train_data, target_data, other_owner_data, allTest, ss_size, 1, delta, test, util)
    
    topk_items = [i.index for i in topk[0]]
    
    perms += topk[1]

    permutations_dict = topk[2]
    
    t1 = time.time()
    print(f"The time it took for first iteration was: {t1-t0}")
    
    for i in range(1, len(topk_items)+1):
        subset.append(topk_items[0]) # try with 1, then 2, etc.
        print(subset)
        newA = A_data.copy()
        for el in subset:
            newA.remove(el)
        owners.update({A: newA})
        owners.update({B: list(np.append(B_data, list(subset)))})
        probability = math.inf
        
        t2=time.time()

        est_mean, added_perms, permutations_dict = diffShapleyMCKDE(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
        perms += added_perms

        owners.update({A: A_data})
        owners.update({B: B_data})
        
        t3=time.time()
        print(f"The time it took for moving subset was: {t3-t2}")
        tend = time.time()
        if(tend-tbeg > 7200):
            return subset, perms, False, 1
        if est_mean < 0: 
            owners.update({A: newA})
            owners.update({B: list(np.append(B_data, list(subset)))})
            diff = diffShapleyMCKDE(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util)
            if (diff[0] < 0):
                flipped = True
            else:
                flipped = False
            # flipped = verifyShapley(A, B, all_train_data, target_data, owners, ss_size, delta, test,util)
            owners.update({A: A_data})
            owners.update({B: B_data})
            return subset, perms, flipped, diff[0]
        
        t4=time.time()
        topk = topKBaselineKDE(permutations_dict, A, B, owners, all_train_data, target_data, other_owner_data, topk[0][1:], ss_size, 1, delta, test, util)
        
        topk_items = [i.index for i in topk[0]]
        
        perms += topk[1]

        permutations_dict = topk[2]
        
        t5=time.time()
        print(f"The time it took for ranking next item was: {t5-t4}")
        
    return [0,0]

# %%
def topKBaselineKDE(permutations_dict, A, B, owners, all_train_data, target_data, other_owner_data, items, step_size, k, delta, test, util):
    if (len(owners[A])==1): return [items, 0, {}]

    A_data = owners[A].copy()
    B_data = owners[B].copy()

    total_perms = 0

    if len(items[0].power)==0:
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
    #                 print("datatups")
    #                 for l in O: print(l)

            for i in items:
                # samples = []
                owners.update({A: list(set(np.append(owners[A], P_data)))})
                owners.update({B: list(set(np.append(owners[B], P_data)))})
                owners[A].remove(i.index)
                owners[B].append(i.index)
                if coaString not in i.coa_dict:
                    i.coa_dict.update({coaString: 0.5*len(owners)*(1/(len(owners)-len(S)-1))*diffUtilityKDE(all_train_data, target_data, test, owners, A, B, util)})
                i.power.append(i.coa_dict[coaString])
                owners.update({A: A_data})
                owners.update({B: B_data})

        for i in items:
            i.mean = np.mean(i.power)
            i.std = np.std(i.power, ddof=1)
    
    sorted_items = sorted(items, key=lambda x:x.mean)

    while total_perms < 30000 and not findSig(sorted_items, 1, delta):

        for i in sorted_items:
            ran_mean = np.random.normal(loc=np.mean(i.power), scale=np.std(i.power, ddof=1)/math.sqrt(len(i.power)))
            i.ran_mean = ran_mean
        
        thompson_sorted = sorted(items, key=lambda x:x.ran_mean)
        i = thompson_sorted[0]

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
                i.coa_dict.update({coaString: 0.5*len(owners)*(1/(len(owners)-len(S)-1))*diffUtilityKDE(all_train_data, target_data, test, owners, A, B, util)})
            i.power.append(i.coa_dict[coaString])
            # samples.append(i.coa_dict[coaString])
            owners.update({A: A_data})
            owners.update({B: B_data})
            
        i.mean = np.mean(i.power)
        i.std = np.std(i.power, ddof=1)
        sorted_items = sorted(sorted_items, key=lambda x:x.mean)
        # print("1st place mean: ", sorted_items[0].mean)
        # print("2nd place mean: ", sorted_items[1].mean)
    return [sorted_items, total_perms, permutations_dict]

# %%
def diffShapleyMCKDE(permutations_dict, all_train_data, target_data, owners, A, B, delta, ss_size, test, util):
    total_samples = []
    perms = 0

    change = math.inf
    old_mean = 0.0000000001

    while (change > 0.01):
        perms += ss_size
        total_samples, permutations_dict = monteCarloKDE(total_samples, permutations_dict, owners, all_train_data, target_data, A, B, ss_size, test, util)
        est_mean = np.mean(total_samples)
        est_std = np.std(total_samples, ddof=1)
        print("mean: ", est_mean)
        change = abs((est_mean-old_mean)/old_mean)
        old_mean = est_mean
        print("change: ",change)

    return est_mean, perms, permutations_dict

# %%
def preprocessKDE(n_owners, ds_size, total_size, train, test, owner_data_dist, owner_size_dist, starting_diff, util):

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
                
            if (owner_data_dist == 'clustered'):
                gmm = estimate_gaussian_mixture(train, column, num_components=4)
                sampled_rows = sample_rows_from_gaussian_component(train, gmm, column, component_index=random.randint(0,3), num_samples=size)
                owner_data = list(sampled_rows)
                
            
        if (owner_size_dist == 'zipfian'):
            # every data owner's size get sampled from zipfian distribution (lots will have very few, some have very large amt)
            size = int(zipfian(1.5, 1, num_rows-1, size=1))

            if (owner_data_dist == 'uniform'):
                owner_data = random.sample(range(0, num_rows), size)
                    
            if (owner_data_dist == 'clustered'):
                gmm = estimate_gaussian_mixture(train, column, num_components=4)
                sampled_rows = sample_rows_from_gaussian_component(train, gmm, column, component_index=random.randint(0,3), num_samples=size)
                owner_data = list(sampled_rows)
        
        owners.update({i: owner_data})
    
    for o in owners.values():
        dataset = dataset.union(set(o))
    
    dataset = list(dataset)
    
    
    lenA = 0
    while (lenA<=1):
        twoOwners = random.sample(list(owners.keys()), 2)
        difference = diffShapleyMCKDE({},train, train.loc[dataset], owners, twoOwners[0], twoOwners[1], 0.05, 50, test, util)

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
            

# %%
def estimate_gaussian_mixture(dataframe, column_name, num_components=4, random_state=42):
    """
    Estimate a Gaussian Mixture Model (GMM) for a specified column in a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to estimate the GMM for.
        num_components (int): The number of Gaussian components in the mixture (default is 2).
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        GaussianMixture: The fitted GMM model.
    """
    # Extract the target column as a 1D array
    target_data = dataframe[column_name].values.reshape(-1, 1)

    # Create and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, random_state=random_state)
    gmm.fit(target_data)
    return gmm

# %%
def sample_rows_from_gaussian_component(dataframe, gmm, column, component_index, num_samples, random_state=42):
    np.random.seed(random_state)
    
    # Assign each row to a GMM component
    component_assignments = gmm.predict(dataframe[column].values.reshape(-1, 1))
    print(component_assignments)

    # Get the indices of rows assigned to the specified component
    component_indices = np.where(component_assignments == component_index)[0]

    # Randomly sample rows from the specified component
    sampled_indices = np.random.choice(component_indices, size=num_samples, replace=False)

    return sampled_indices

# %%
def diffShapleyKDE(all_train_data, target_data, test, owners, A, B, util):
    A_data = owners[A]
    B_data = owners[B]
    list_owners = list(owners.keys())
    list_owners.remove(A)
    list_owners.remove(B)
    coalitions = powerset(list_owners)
    total = 0
    for S in coalitions:
        P_data = []
        for o in S:
            P_data.append(owners[o])
        P_data = list(flatten(P_data))
        owners.update({A: list(np.append(owners[A], P_data))})
        owners.update({B: list(np.append(owners[B], P_data))})
        total += 1/((len(S)+1)*math.comb(len(list(owners.keys()))-1,len(S)+1)) * diffUtilityKDE(all_train_data, target_data, test, owners, A, B, util)
        owners.update({A: A_data})
        owners.update({B: B_data})
    return total

# %%
def findSig(tups, k, delta):
    if len(tups)==1: return True
    first = tups[0]
    second = tups[1]
    z = norm.ppf(1-(delta/2))
    n = len(tups[0].power)
    if n==0: return False
    # if (tups[0].mean+z*(tups[0].std/np.sqrt(n))<=tups[1].mean and tups[1].mean-z*(tups[1].std/np.sqrt(n))>=tups[0].mean):
    #     return True
    mean_diff = second.mean-first.mean
    lb = mean_diff-z*np.sqrt(first.std**2+second.std**2)/np.sqrt(n)
    ub = mean_diff+z*np.sqrt(first.std**2+second.std**2)/np.sqrt(n)
    if (lb < 0 < ub):
        return False
    return True


def zipfian(a, minimum, maximum, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(minimum, maximum+1) # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.random.choice(v, size=size, replace=True, p=p)

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

def wrap_testBatchKDE(args):
    return testBatchKDE(*args)

# %%
# data_owner_data is a list of indices
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
def diffUtilityKDE(all_train_data, target_data, test_data, owners, o1, o2, util='decision_tree'):
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
# data_owner_data is a list of indices
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
    # Fit the KDE model
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
def preComputeSingleton(items, owners, target_data, test, A, B, util, iter=10000):
    for r in range(iter):
#             print(total_perms)
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
        
        for i in items:
            owners.update({A: list(set(np.append(owners[A], P_data)))})
            owners.update({B: list(set(np.append(owners[B], P_data)))})
            owners[A].remove(i.index)
            owners[B].append(i.index)

            if coaString not in i.coa_dict:
                i.coa_dict.update({coaString: 0.5*len(owners)*(1/(len(owners)-len(S)-1))*diffUtilityKDE(all_train_data, target_data, test, owners, A, B, util)})
            # samples.append(i.coa_dict[coaString])
            owners.update({A: A_data})
            owners.update({B: B_data})
    

# %%
def shuffle_and_split(data):
    '''Takes an array x, shuffles, and splits into training and data set'''
    
#     np.split(original_data.sample(frac=1, random_state=1729), 
#                                [int(0.8 * len(original_data)), int(0.2*len(original_data))])
#     shuffled = data.sample(frac=1)
#     folds = np.array_split(shuffled, n_folds)  
    
#     np.random.shuffle(x)
#     X_train, X_test = x[:int((len(x)*0.8)), :], x[int(len(x)*0.8):, :]
#     X_train = pd.DataFrame(data=X_train)
#     X_test = pd.DataFrame(data=X_test)
    
    X_train, X_test = np.split(data.sample(frac=1, random_state=1729), [int(0.8 * len(data))])
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, X_test

# %%
def compareAllKDE(all_train_data, test_data, target_data_size, num_owners, num_tries, start_diff, data_dist, size_dist, util):

    dataset, owners, A, B = preprocessKDE(num_owners, target_data_size, len(all_train_data), all_train_data, test_data, owner_data_dist= data_dist, owner_size_dist = size_dist, starting_diff=start_diff, util=util)
    while (len(owners[A])==1 and len(owners[B])==1):
        dataset, owners, A, B = preprocessKDE(num_owners, target_data_size, len(all_train_data), all_train_data, test_data, owner_data_dist=data_dist, owner_size_dist = size_dist, starting_diff=start_diff, util=util)
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

        grbstimestart = time.time()
        greedySolBaseline = greedyKDE(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
        grbstimeend = time.time()
        grbsTime = grbstimeend - grbstimestart
        GRBSTimes.append(grbsTime)
        greedyBaselineCheck.append(greedySolBaseline[2])
        greedyBaselineSamples.append(greedySolBaseline[1])
        greedyBaselineAnswers.append(len(greedySolBaseline[0]))
        GRBSDiff.append(greedySolBaseline[3])
        

        mctimestart = time.time()
        MCSol = MCSubsetKDE(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
        mctimeend = time.time()
        mcTime = mctimeend - mctimestart
        MCTimes.append(mcTime)
        MCCheck.append(MCSol[2])
        MCSamples.append(MCSol[1])
        MCAnswers.append(len(MCSol[0]))
        MCDiff.append(MCSol[3])

    return len(dataset), len(set(owners[A])), len(set(owners[B])), BFSol, MCSamples, greedyBaselineSamples, MCAnswers, greedyBaselineAnswers, MCTimes, GRBSTimes, MCCheck, greedyBaselineCheck, MCDiff, GRBSDiff

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
    greedySolBaseline = greedyKDE(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
    grbstimeend = time.time()
    grbsTime = grbstimeend - grbstimestart
    GRBSTimes.append(grbsTime)
    greedyBaselineCheck.append(greedySolBaseline[2])
    greedyBaselineSamples.append(greedySolBaseline[1])
    greedyBaselineAnswers.append(len(greedySolBaseline[0]))
            

    mctimestart = time.time()
    MCSol = MCSubsetKDE(A, B, all_train_data, all_train_data.loc[dataset], owners, 10, 0.05, test_data, util)
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

    # categorical_columns = all_train_data.select_dtypes(include=['object']).columns

    all_train_data = all_train_data.drop(cat, axis=1)
    test_data = test_data.drop(cat, axis=1)

    # for i in range(len(owners)):
    #     for j in range(len(owners)):
    #         if (i != j):
    #             A = i
    #             B = j
    grbstimestart = time.time()
    greedySolBaseline = greedyKDE(A, B, all_train_data, all_train_data, owners, 20, 0.05, test_data, util)
    grbstimeend = time.time()
    grbsTime = grbstimeend - grbstimestart
    GRBSTimes.append(grbsTime)
    greedyBaselineCheck.append(greedySolBaseline[2])
    greedyBaselineSamples.append(greedySolBaseline[1])
    greedyBaselineAnswers.append(len(greedySolBaseline[0]))
            

    mctimestart = time.time()
    MCSol = MCSubsetKDE(A, B, all_train_data, all_train_data, owners, 20, 0.05, test_data, util)
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
def verifyShapley(A, B, all_train_data, target_data, owners, ss_size, delta, test,util="decision_tree"):
    owner_A_samples = []
    owner_B_samples = []
    permutations_dict = {}

    A_data = owners[A]
    B_data = owners[B]
    value2 = -1000
    value1 = 1000
    while value2<0<value1:
        for r in range(ss_size):
            permutation = np.random.permutation(list(owners.keys()))
            S = findCoalition(permutation, A, B)
            permString = listToString(S)

            P_data = []

            if permString not in permutations_dict:
                P_data = [item for key in S for item in owners.get(key, [])]
                permutations_dict.update({permString: P_data})
            else:
                P_data = permutations_dict[permString]
            
            owners.update({A: list(np.append(owners[A], P_data))})
            owners.update({B: list(np.append(owners[B], P_data))})

            owner_A_samples.append(utilityFunc(all_train_data, target_data, test, owners[A], util)-utilityFunc(all_train_data, target_data, test, P_data, util))
            owner_B_samples.append(utilityFunc(all_train_data, target_data, test, owners[B], util)-utilityFunc(all_train_data, target_data, test, P_data, util))
            owners.update({A: A_data})
            owners.update({B: B_data})
        owner_A_mean = np.mean(owner_A_samples)
        owner_A_std = np.std(owner_A_samples,ddof=1)
        owner_B_mean = np.mean(owner_B_samples)
        owner_B_std = np.std(owner_B_samples,ddof=1)
        n = len(owner_A_samples)
        print("n: ", n)
        value1 = owner_A_mean-owner_B_mean + 1.96*np.sqrt(owner_A_std**2+owner_B_std**2)/np.sqrt(n)
        value2 = owner_A_mean-owner_B_mean -1.96*np.sqrt(owner_A_std**2+owner_B_std**2)/np.sqrt(n)
        
    print("difference in means: ",owner_A_mean-owner_B_mean)
    if(owner_A_mean < owner_B_mean):
        return True
    else:
        return False
            


# %%
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
def testBatchKDE(all_train_data, test_data, target_data_size, num_owners, num_iterations, num_trials, start_diff, data_dist, size_dist, csv_file_path,util, ds_name):
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
        datasetSize, ownerASize, ownerBSize, correctSol, MCSamples, greedyBaselineSamples, mcSub, greedyBaselineSubset, MCTimes, GRBSTimes, MCCheck, GRBSCheck, MCDiff, GRBSDiff = compareAllKDE(all_train_data, test_data, target_data_size, num_owners, num_iterations, start_diff, data_dist, size_dist, util)
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

# %% [markdown]
# 

# %%
# adult_data = pd.read_csv("adult.csv")
# X_train, X_test = shuffle_and_split(adult_data)
# X_train.to_csv("adult_train.csv", index=False)
# X_test.head(200).to_csv("adult_test.csv", index=False)

# %%
# X_train.head()

# %%


# %%
# cat = "marital-status"
# X_train = pd.read_csv("adult_train.csv")
# X_test = pd.read_csv("adult_test.csv")
# sample = X_train.sample(n=800)
# owners, categories = assignClusters(sample, cat)
# categorical_columns = sample.select_dtypes(include=['object']).columns
# sample = sample.drop(categorical_columns, axis=1)
# X_test = X_test.drop(categorical_columns, axis=1)
# for key in owners:
#     print(len(owners[key]))
# print(categories)

# X_train_cancer = pd.read_csv("breast_cancer_train_copy.csv")
# X_test_cancer = pd.read_csv("breast_cancer_test_copy.csv")
# scaler = MinMaxScaler()

# X_train = pd.read_csv("reservation_train.csv")
# X_test = pd.read_csv("reservation_test.csv")
# owners, categories = assignClusters(X_train, "arrival_month")
# print(categories)
# A=7
# B=6
# print(owners[A])
# print(owners[B])
# subset=[38,44,62,65]
# newA = owners[A].copy()
# B_data = owners[B]
# for el in subset:
#     newA.remove(el)
# owners.update({A: newA})
# owners.update({B: list(np.append(B_data, list(subset)))})
# print(owners[7])
# print(owners[6])
# res=diffShapleyMCKDE({}, X_train, X_train, owners, A, B, 0.05, 10, X_test, "log_reg")
# # results = compareNatural("arrival_month", X_train_adult, X_test_adult, "clustered", "zipfian", "log_reg","adult", "PRACTICE_NATURAL_PART2.csv", 1)
# # # columns_to_scale = X_train_spam.columns.drop("target")
# # X_train_spam[columns_to_scale] = scaler.fit_transform(X_train_spam[columns_to_scale])
# X_test_spam[columns_to_scale] = scaler.fit_transform(X_test_spam[columns_to_scale])

# # columns_to_scale_c = X_train_cancer.columns.drop("target")
# # X_train_cancer[columns_to_scale_c] = scaler.fit_transform(X_train_cancer[columns_to_scale_c])
# # X_test_cancer[columns_to_scale_c] = scaler.fit_transform(X_test_cancer[columns_to_scale_c])


# # # %%
# # # utilityFunc(sample, sample, X_test, owners[6], "log_reg")

# # # %%
# datasetSizes, ownerASizes, ownerBSizes, trueAnswers, totalMCSamples, totalGreedyBaselineSamples, finalMCAnswers, finalGreedyBaselineAnswers, MCTimesAvg, GRBSTimesAvg, totalMCCheck, totalGRBSCheck, MCDiff, GRBSDiff = testBatchKDE(X_train_cancer, X_test_cancer, 100, 15, 1, num_trials=1, start_diff=0, data_dist="uniform", size_dist="uniform",csv_file_path="GAHHHHHHHHEXPERIMENTS.csv",util="log_reg",ds_name="cancer")

# %%
# blah = compareNatural("workclass", X_train.sample(n=800), X_test, "clustered", "zipfian", "log_reg", "adult", "PRACTICE_NATURAL.csv")

# %%
# test1 = compareZipfian(3**4, 3**2, X_train, X_test, 10, "uniform", "zipfian", "log_reg","cancer", "ZIPFIAN_TABLE_DATA.csv")

# %%
# diffShapleyMCKDE(X_train_cancer, X_train_cancer.loc[dataset], owners, A, B, 0.05, 10, X_test_cancer, "log_reg")


