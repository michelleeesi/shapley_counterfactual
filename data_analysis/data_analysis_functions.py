# %%
import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal, wasserstein_distance
warnings.filterwarnings("ignore")

# %%
# assigning clusters for the natural distribution
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

# %%
def add_log_column(df, cols):
    for column_name in cols:
        if column_name in df.columns:
            log_column_name = f"Log_{column_name}"
            df[log_column_name] = np.log(df[column_name])
        else:
            print(f"Column '{column_name}' not found in the dataframe.")

    return df


# %%
def plot_sorted_columns(dataframes, sort_column, column1, column2):
    # Create subplots with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for i, df in enumerate(dataframes):
        # Sort the DataFrame by the specified column
        sorted_df = df.sort_values(by=sort_column)

        # Reset the index of the sorted DataFrame
        sorted_df = sorted_df.reset_index(drop=True)

        # Plot the two columns against their indices
        row = i // 5
        col = i % 5
        axes[row, col].plot(sorted_df.index, sorted_df[column1], label=column1)
        axes[row, col].plot(sorted_df.index, sorted_df[column2], label=column2)

        # Set labels and legend for each subplot
        axes[row, col].set_title(f"{df['util'][0]}, {df['num_owners'][0]}, {df['data_dist'][0]}, {df['ds_name'][0]}")
        axes[row, col].set_xlabel('Trial')
        axes[row, col].set_ylabel('Runtime (in seconds)')
        axes[row, col].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()



# %%
def prepareData(csv_list, plot_col):
    for df in csv_list:
        df.loc[df['MC_time'] > 7200, 'MC_time'] = float('nan')
        df.loc[df['greedy_time'] > 7200, 'greedy_time'] = float('nan')
        df = add_log_column(df, ["MC_time", "greedy_time"])
    plot_sorted_columns(csv_list, plot_col, "MC_time","greedy_time")
# %%

def process_csv_files(csv_files, columns_to_process):
    processed_df = pd.DataFrame()

    for df in csv_files:
        df.loc[df['MC_diff'] == 1, 'MC_answer'] = np.nan
        df.loc[df['MC_diff'] == 1, 'MC_time'] = np.nan
        df.loc[df['greedy_diff'] == 1, 'greedy_answer'] = np.nan
        df.loc[df['greedy_diff'] == 1, 'greedy_time'] = np.nan
        df.loc[df['MC_diff'] == 1, 'MC_accuracy'] = np.nan
        df.loc[df['greedy_diff'] == 1, 'greedy_accuracy'] = np.nan
        df.loc[df['MC_answer'] == 0, 'MC_answer'] = np.nan
        df.loc[df['MC_time'] == 0, 'MC_time'] = np.nan

        # Calculate statistics for each specified column
        stats = {}
        for column in columns_to_process:
            col_mean = df[column].mean(skipna=True)
            col_stdev = df[column].std(skipna=True)
            stats[f"{column}_mean"] = col_mean
            stats[f"{column}_std"] = col_stdev

        stats["MC_accuracy"] = (df['MC_accuracy'].mean(skipna=True))*100
        stats["greedy_accuracy"] = (df['greedy_accuracy'].mean(skipna=True))*100
        stats["num_owners"] = df['num_owners'][0]
        stats["util"] = df['util'][0]
        stats["data_dist"] = df['data_dist'][0]
        stats["size_dist"] = df['size_dist'][0]
        stats["ds_name"] = df['ds_name'][0]
        # Append calculated statistics to the result DataFrame
        processed_df = processed_df.append(stats, ignore_index=True)
    return processed_df

# %%

def plot_sorted_columns(dataframes, xvar, column1, column2):
    """
    Sort DataFrames by a specified column and plot two other columns against their indices.

    Parameters:
    - dataframes: list of pandas DataFrames
      The input list containing the DataFrames.
    - sort_column: str
      The name of the column by which to sort the DataFrames.
    - column1: str
      The name of the first column to plot.
    - column2: str
      The name of the second column to plot.
    """
    # Create subplots with 2 rows and 5 columns
    # fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for i, df in enumerate(dataframes):

        
        df = df.sort_values(by=xvar)

        # plt.scatter(df[xvar], df[column1], label=f"MC, n={df['num_owners'][0]}", color = cblue)

        plt.scatter(df[xvar], df[column2], label=f"n={df['num_owners'][0]}")

        plt.yscale('log')

        # # Reset the index of the sorted DataFrame
        # sorted_df = sorted_df.reset_index(drop=True)

        # # Plot the two columns against their indices
        # row = i // 3
        # col = 1
        # axes[row, col].plot(sorted_df.index, sorted_df[column1], label=column1)
        # axes[row, col].plot(sorted_df.index, sorted_df[column2], label=column2)

        # # Set labels and legend for each subplot
        # axes[row, col].set_title(f"{df['util'][0]}, {df['num_owners'][0]}, {df['ds_name'][0]}")
        # axes[row, col].set_xlabel('Trial')
        # axes[row, col].set_ylabel('Runtime (in seconds)')
        # axes[row, col].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.legend()
    # plt.xlabel('Size of A Dataset - Size of B Dataset', fontsize=16)
    # plt.xlabel('Size of B Dataset', fontsize=16)
    plt.xlabel('Size of SV-Exp Counterfactual', fontsize=16)
    plt.ylabel('Runtime (in seconds)', fontsize=16)
    # plt.title('Runtime of SV-Exp vs. A-B Dataset Size Difference', fontsize=19, pad=10)
    plt.title('Runtime of SV-Exp vs. \n Size of SV-Exp Counterfactual', fontsize=19, pad=10)
    # plt.title('Runtime of SV-Exp vs. Size of B Dataset', fontsize=19, pad=10)
    # plt.figure(figsize=(200, 20))

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(0, 90)
    # Show the plot
    plt.show()

def plot_runtimes(summary_stats, mc_var, greedy_var):
    # Sample data (replace this with your actual data)
    mc_time_means = summary_stats[f'{mc_var}_mean']
    mc_time_stds = summary_stats[f'{mc_var}_std']

    greedy_time_means = summary_stats[f'{greedy_var}_mean']
    greedy_time_stds = summary_stats[f'{greedy_var}_std']

    num_owners = summary_stats['num_owners']
    util = summary_stats['util']

    # Number of data points
    num_points = len(mc_time_means)

    # Plotting
    plt.figure(figsize=(10, 6))  # Set figure size

    bar_width = 0.35
    index = np.arange(num_points)

    # Plot bars for MC
    plt.bar(index, mc_time_means, bar_width, label='MC', yerr=mc_time_stds, capsize=5)
    # Plot bars for SV-Exp
    plt.bar(index + bar_width, greedy_time_means, bar_width, label='SV-Exp', yerr=greedy_time_stds, capsize=5)

    plt.xlabel('Parameter Set', fontsize=19)
    plt.ylabel('Mean Runtime (seconds)', fontsize=19)
    plt.title(f'Runtime Performance for {util[0]} Utility', fontsize=25)

    # Use custom labels for x-axis
    parameter_labels = [f"{util[i]}, \n n={num_owners[i]}" for i in range(0, num_points)]
    plt.xticks(index + bar_width / 2, parameter_labels, fontsize=17)

    # Customize the y-axis tick labels to e^y
    plt.yscale('log')
    plt.yticks(fontsize=17)

    plt.ylim(0, 10000)

    plt.legend()
    
    plt.show()


# %%
def shuffle_and_split(data):
    
    X_train, X_test = np.split(data.sample(frac=1, random_state=1729), [int(0.8 * len(data))])
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, X_test

# %%

def makeTableNat(df, x, y, metrics):

    # Group by all possible combinations of two columns and calculate the mean runtime
    for m in metrics:

        title = ""
        
        if m == "MC_answer":
            title = "MC Counterfactual Size"
        
        if m == "greedy_answer":
            title = "SV-Exp Counterfactual Size"

        if m == "MC_accuracy":
            title = "MC Counterfactual Accuracy"

        if m == "greedy_accuracy":
            title = "SV-Exp Counterfactual Accuracy"
        
        # df['MC_answer'] = df['MC_answer']/df['A_size']


        grouped_mean = df.groupby([x, y])[m].apply(lambda group: group.mean(skipna=True)).reset_index()
        grouped_std = df.groupby([x, y])[m].apply(lambda group: group.std(skipna=True)).reset_index()

        # Create a pivot table
        pivot_table_mean = grouped_mean.pivot(index=y, columns=x, values=m)[::-1]
        pivot_table_mean.index = pivot_table_mean.index.astype(int)
        pivot_table_mean.columns = pivot_table_mean.columns.astype(int)

        pivot_table_std = grouped_std.pivot(index=y, columns=x, values=m)[::-1]
        pivot_table_std.index = pivot_table_std.index.astype(int)
        pivot_table_std.columns = pivot_table_std.columns.astype(int)

        # # Custom function to remove leading zeros from the format
        # def format_no_leading_zero(x):
        #     return str(x).lstrip('0')

        # # Apply the custom formatting to the pivot tables
        # pivot_table_mean = pivot_table_mean.applymap(format_no_leading_zero)
        # pivot_table_std = pivot_table_std.applymap(format_no_leading_zero)

        months = ['Jan (25)', 'Feb (37)', 'Mar (52)', 'Apr (45)', 'May (59)', 'Jun (61)', 'Jul (65)', 'Aug (79)', 'Sep (105)', 'Oct (117)', 'Nov (73)', 'Dec (82)']

        # print(pivot_table_mean)
        # print(pivot_table_std)

        # # Create a heatmap
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # sns.heatmap(pivot_table_std/pivot_table_mean, xticklabels=months, yticklabels=months[::-1], annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 8})
        # # plt.title(f'{m} Mean Heatmap')
        # # plt.title('MC Counterfactual Accuracy Mean Heatmap', fontsize=15, pad=10)
        # plt.title('SV-Exp CF Size Coefficient of Variation Heatmap', fontsize=15, pad=10)
        # plt.xlabel("Month (A)", fontsize=13)
        # plt.ylabel("Month (B)", fontsize=13)
        # plt.tick_params(labelsize=11)

        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # sns.heatmap(pivot_table_mean, xticklabels=months, yticklabels=months[::-1], annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 8})
        # # plt.title(f'{m} Mean Heatmap')
        # # plt.title('MC Counterfactual Accuracy Mean Heatmap', fontsize=15, pad=10)
        # plt.title('Initial Differential Shapley Heatmap', fontsize=15, pad=10)
        # plt.xlabel("Month (A)", fontsize=13)
        # plt.ylabel("Month (B)", fontsize=13)
        # plt.tick_params(labelsize=11)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_table_mean, xticklabels=months, yticklabels=months[::-1], annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 8.5})
        # plt.title(f'{m} Mean Heatmap')
        # plt.title('MC Counterfactual Accuracy Mean Heatmap', fontsize=15, pad=10)
        plt.title('SV-Exp CF Size Mean Heatmap', fontsize=18, pad=10)
        plt.xlabel("Month (A)", fontsize=16)
        plt.ylabel("Month (B)", fontsize=16)
        plt.tick_params(labelsize=14)

        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_table_std, xticklabels=months, yticklabels=months[::-1], annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 8.5})
        # plt.title(f'{m} Std. Dev. Heatmap')
        # plt.title('MC Counterfactual Accuracy Std. Dev. Heatmap', fontsize=15, pad=10)
        plt.title('SV-Exp CF Size Std. Dev. Heatmap', fontsize=18, pad=10)
        plt.xlabel("Month (A)", fontsize=16)
        plt.ylabel("Month (B)", fontsize=16)

        plt.tick_params(labelsize=15)
        plt.tight_layout()
        plt.show()


# %%

def makeTableZipf(df, x, y, metrics):
    # Group by all possible combinations of two columns and calculate the mean runtime
    for m in metrics:

        title = ""
        
        if m == "MC_answer":
            title = "MC Counterfactual Size"
        
        if m == "greedy_answer":
            title = "SV-Exp Counterfactual Size"

        if m == "MC_accuracy":
            title = "MC CF Success Rate"

        if m == "greedy_accuracy":
            title = "SV-Exp CF Success Rate"

        grouped_mean = df.groupby([x, y])[m].apply(lambda group: group.mean(skipna=True)).reset_index()
        grouped_std = df.groupby([x, y])[m].apply(lambda group: group.std(skipna=True)).reset_index()

        # Create a pivot table
        pivot_table_mean = grouped_mean.pivot(index=y, columns=x, values=m)[::-1]
        pivot_table_std = grouped_std.pivot(index=y, columns=x, values=m)[::-1]
        # print(pivot_table_mean)
        # print(pivot_table_std)

        # Create a heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_table_mean, annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 13})
        # plt.title(f'{m} Mean Heatmap')
        plt.title(f'{title} Mean Heatmap', fontsize=17, pad=10)
        # plt.title('MC Counterfactual Accuracy Mean Heatmap')
        plt.xlabel("Size of Owned Data (A)", fontsize=16)
        plt.ylabel("Size of Owned Data (B)", fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_table_std, annot=True, cmap='YlGnBu', fmt=".2f",annot_kws={"size": 13})
        # plt.title(f'{m} Std. Dev. Heatmap')
        plt.title(f'{title} Std. Dev. Heatmap', fontsize=17, pad=10)
        # plt.title('MC Counterfactual Accuracy Std. Dev. Heatmap')
        plt.xlabel("Size of Owned Data (A)", fontsize=16)
        plt.ylabel("Size of Owned Data (B)", fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()
    

# %%
def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

# %%
def genWassDist(train_data, cat):

    owners, categories = assignClusters(train_data, cat)
    for k in owners:
        print(len(owners[k]))
    # print(categories)

    n = len(owners)

    # Calculate Wasserstein distances for each pair of subsets
    wasserstein_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            month1 = categories[i]
            month2 = categories[j]
            subset1_indices = owners[i]
            subset2_indices = owners[j]

            subset1 = train_data.iloc[subset1_indices, :]
            subset2 = train_data.iloc[subset2_indices, :]
            # print(month1, len(subset1))
            # print(month2, len(subset2))

            wasserstein_dist = wasserstein_distance(subset1.values.flatten(), subset2.values.flatten())
            wasserstein_distances[month1-1, month2-1] = wasserstein_dist
            wasserstein_distances[month2-1, month1-1] = wasserstein_dist  # Symmetric matrix

    # Display the Wasserstein distance heatmap

    wasserstein_df = pd.DataFrame(wasserstein_distances[::-1, :], index=range(n, 0, -1), columns=range(1, n + 1))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(wasserstein_df, xticklabels=months, yticklabels=months[::-1], cmap='viridis', annot=True, fmt=".3f", cbar_kws={'label': 'Wasserstein Distance'})

    plt.title('Wasserstein Distance Heatmap Between Data Subsets')
    plt.xlabel(cat)
    plt.ylabel(cat)
    plt.show()

    # Assuming you have two subsets of rows
    # subset1_indices = owners[1]
    # subset2_indices = owners[8]

    # subset1 = X_train.iloc[subset1_indices, :]
    # subset2 = X_train.iloc[subset2_indices, :]

    # # Calculate means and covariance matrices for each subset
    # mean_cov_subset1 = (subset1.mean().values, np.cov(subset1.values, rowvar=False))
    # mean_cov_subset2 = (subset2.mean().values, np.cov(subset2.values, rowvar=False))

    # # Calculate KL divergence
    # kl_divergence = kl_mvn(subset1.mean().values, np.cov(subset1.values, rowvar=False), subset2.mean().values,np.cov(subset2.values, rowvar=False))


