
# coding: utf-8

# This is an implementation of NIPS 2017 paper, titled "*Thy Friend is My Friend: Iterative Collaborative Filtering for Sparse Matrix Estimation*" ( [link](https://papers.nips.cc/paper/7057-thy-friend-is-my-friend-iterative-collaborative-filtering-for-sparse-matrix-estimation.pdf) and [arxiv](https://arxiv.org/pdf/1712.00710v1.pdf) ) for Global NIPS Paper implementation [Challenge](https://www.nurture.ai/nips-challenge). The authors have also created a short [video](https://www.youtube.com/watch?v=qxfDK44YuQE) for it.<br>
# For any questions related to this notebook, feel free to contact me at ssahoo.infinity@gmail.com .

# ## How to use this Notebook:
# Following are some important points/guidelines/assumptions to keep in mind while navigating through this notebook:
# - This notebook consists of sections numbered by Roman numerals
# - Brief description about all sections:
#   - I: Model preparation: Changes to be made to rating matrix(dataset) before we can apply the algorithms
#   - II: Algorithm Details: All the algorithms described in the paper are implemented here.
#   - III: Other important functions:
#     - Data Handling: for data manipulation
#     - Substitutes: methods which can be used in place of (algorithm) methods in paper
#     - Evaluation: for evaulation of recommender system
#   - IV: Test script/Experiments: testing using a dataset
# - If a function/variable has suffix as "_csr", it refers to CSR (Compressed Sparse Row) data. Else, if it has a suffix as "_matrix", it refers to 2D (numpy) matrix data.
# - The dataset ratings are assumed to be integers; to be modified in future
# - data_csr ensures that user_id and item_id start from 0 by taking in FIRST_INDEX as a global variable
# - All the datasets are symmetricized and normalized before we begin applying the algorithm
# - There are some dataset parameters like FIRST_INDEX, USERS, etc which will be automatically detected from dataset. Hence, they have been set as -1 for default condition. To overload automatic detection, provide a value (but we recommend you not to overload).
# - Brief description about hyperparameters:
#   - RADIUS : this controls the size of neighborhood. It is the distance from vertex u to vertices i (at neighborhood boundary). Setting large/small values for RADIUS might reduce the average number of neighbors per vertex, resulting suboptimal overlap of vertices. To set optimal RADIUS, you can use output from 'describe_neighbor_count' function to understand how varying RADIUS affects neighborhood size.
#   - THRESHOLD : this controls the final set of vertices which are considered for individual rating estimation. Setting this too high will increase the size of the set of vertices being considered and vice versa. To set optimal threshold, you can use output from 'describe_distance_matrix' function to understand how THRESHOLD affects the size of this set.
#   - UNPRED_RATING : average rating for user-item pair, for which the algorithm could not make an estimate for the rating.
#   - TRAIN_TEST_SPLIT : %age of test dataset.
#   - C1 : %age of edges in train dataset going to $E1$ for expanding the neighborhood (step 2).
#   - C2 : %age of edges in train dataset going to $E2$ for distance computation between the vertices (step 3). Also, please note that 1 - C1 - C2 is %age of edges in train dataset going to $E3$ for rating estimation (step 4).
# - As noted in paper, the most computationally expensive step in the algorithm is Step 3: Distance computation. You can reduce the time for this step by reducing the dataset size by setting SIZE_REDUCTION parameter and appropriate user and item limits. Try to ensure user_limit < item_limit. Please note, we are yet to account for items not rated by any users, as a result of size reduction. Look out for output from check_and_set_dataset for further info.

# Importing required modules required for complete notebook

# In[3]:


# Built and tested on python2
import numpy as np
from tqdm import *
import sys
from datetime import datetime
## Make sure the following imports work too (used in later cells)
# import random
# import pandas as pd
# import math
# from sklearn.metrics import mean_squared_error
# from math import sqrt


# In[ ]:

FIRST_INDEX = -1
USERS = -1
ITEMS = -1
SPARSITY = -1                  # 'p' in the equations
UNOBSERVED = 0                 # default value in matrix for unobserved ratings
N_RATINGS = 7
C1 = 0                         # only to account for scale_factor in step 3
C2 = 1                         # only to account for scale_factor in step 3

RADIUS = 3                              # radius of neighborhood, radius = # edges between start and end vertex
UNPRED_RATING = -1                      # rating (normalized) for which we dont have predicted rating

#datetime.now().time()     # (hour, min, sec, microsec)


# # 0. Model Notations
# Following are the symbols/notations used in the paper. The variables/notations used in this notebook have been discussed in Experiments section.
#
# $u$ = user <br>
# $i$ = item <br>
# $M$ = symmetric rating matrix of size $n \times n$ (usually the dataset) <br>
# $E$ = set of $(u,i)$ where each user $u$ has rated an item $i$ also seen in the matrix $M$ (intuitively $E$ is edge set(matrix) between user and items. <br>
# $p$ = sparsity of $M$ i.e. (= #observed ratings in $M$ / total # ratings in $M$)<br>
# $r$ = radius, distance (in no of edges) between user $u$ and item $i$ at neighborhood boundary (look in step 2) <br>

# # I. Model preparation
# We first look at function which converts our asymmetric rating matrix to a symmetric matrix and another function that normalizes the ratings between [0,1].

# This function is used to normalize the ratings:

# In[ ]:


'''Function to normalize all ratings of a CSR (compressed sparse row) matrix'''
def normalize_ratings_csr(data_csr):
    #TODO: assuming non negative ratings, make it generic
    data_csr[:,2] = data_csr[:,2] / float(max(data_csr[:,2]))
    print('Normalize ratings: done')
    return data_csr


# These functions are used to make the given CSR matrix symmetric:

# In[ ]:


''' Function to get data in matrix format for given data in CSR format '''
def csr_to_matrix(data_csr, symmetry=False):

    data_matrix = np.full(((USERS+ITEMS), (USERS+ITEMS)), UNOBSERVED, dtype='float16')
    for line in data_csr:
        data_matrix[int(line[0])][int(line[1])] = line[2]
        if symmetry:
            data_matrix[int(line[1])][int(line[0])] = line[2]

    return data_matrix

'''Function get matrix from csr such that no two item_ids and user_ids are same'''
def get_csr_with_offset(data_csr, offset):
    new_data_csr = np.copy(data_csr)
    new_data_csr[:,1] = new_data_csr[:,1] + offset            # so that user_ids != item_ids
    #new_data_matrix = csr_to_matrix(new_data_csr, symmetry=True)
    return new_data_csr

'''MAIN Function to convert asymmetrix CSR matrix to symmetrix matrix
   the returned CSR doesnt contain repitions for any user-item pair.
   Repetitions can be generated for a 2D matrix by calling csr_to_matrix(data_csr, symmetry=True)'''
def csr_to_symmetric_csr(data_csr):
    # Assuming the rating matrix to be non symmetric
    # Even if it is symmetric, the user_id and item_id need to be different for graph
    data_csr = get_csr_with_offset(data_csr, offset=USERS)
    print('CSR to symmetric CSR matrix: done')
    return data_csr


# # II. Algorithm Details
# As per paper: *We present and discuss details of each step of the algorithm, which primarily involves computing pairwise distances (or similarities) between vertices.*

# ### Step 1: Sample Splitting
# Partition the rating matrix into three different parts. Following are the exerpts from paper:
# - *Each edge in $E$ is independently placed into $E_1, E_2,$ or $E_3$, with probabilities $c_1, c_2,$ and $1 - c_1 - c_2$ respectively. Matrices $M_1, M_2$, and $M_3$ contain information from the subset of the data in $M$ associated to $E_1, E_2$, and $E_3$ respectively.*
# - *$M_1$ is used to define local neighborhoods of each vertex (in step 2), $M_2$ is used to compute similarities of these neighborhoods (in step 3), and $M_3$ is used to average over datapoints for the final estimate (in step 4)*

# In[ ]:


'''Function to split matrix(here CSR) into three different parts with probabilities c1, c2 and 1-c1-c2'''
def sample_splitting_csr(data_csr, c1=0.33, c2=0.33, shuffle=True):
    if shuffle:
        np.random.shuffle(data_csr) # inplace shuffle

    m1_sz = int(c1 * data_csr.shape[0])
    m2_sz = int(c2 * data_csr.shape[0])

    m1_csr = data_csr[              : m1_sz         ,:]
    m2_csr = data_csr[        m1_sz : m1_sz + m2_sz ,:]
    m3_csr = data_csr[m1_sz + m2_sz :               ,:]

    if m1_csr.shape[0]+m2_csr.shape[0]+m3_csr.shape[0] == data_csr.shape[0]:
        print('Sample splitting: done')
    else:
        print('Sample splitting: FAILED')

    return [m1_csr, m2_csr, m3_csr]


# ### Step 2: Expanding the Neighborhood
# We do the following in this step:
# - radius $r$ to be tuned using cross validation. We can use its default value as $r = \frac{6\ln(1/p)}{8\ln(c_1pn)}$ as per paper.
# - use matrix $M_1$ to build neighborhood based on radius $r$
# - Build BFS tree rooted at each vertex to get product of the path from user to item, such that
#   - each vertex (user or item) in a path from user to boundary item is unique
#   - the path chosen is the shortest path (#path edges) between the user and the boundary item
#   - in case of multiple paths (or trees), choose any one path (i.e. any one tree) at random
# - Normalize the product of ratings by total no of final items at the boundary
#
# $N_{u,r}$ obtained is a vector for user $u$ for $r$-hop, where each element is product of path from user to item or zero. $\tilde{N_{u,r}}$ is normalized vector.
#

# Functions useful for getting products along paths:

# In[ ]:


import random

'''Function to create a graph as adjacency list: a dictionary of sets'''
def create_dict_graph_from_csr(data_csr):
    data_matrix = csr_to_matrix(data_csr, symmetry=True)
    # Create an (unweighted) adjacency list for the graph
    ## we still have the 2D matrix for the weights
    graph = dict()
    print('Creating graph as dictionary:')
    sys.stdout.flush()
    for i in tqdm(range(len(data_matrix))):
        temp_set = set()
        for j in range(len(data_matrix[i])):
            if data_matrix[i,j] > 0:
                temp_set.add(j)
        graph[i] = temp_set
    return graph

'''Function gives all possible path from 'start' vertex at r-hop distance '''
# help from:
# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
# radius = # edges between start and end vertex
def bfs_paths(graph, start, radius):
    queue = [(start, [start])]
    visited = [start]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next in visited:
                continue
            depth = len(path + [next]) - 1
            if depth == radius:
                # We do not append next to visited because
                # we want all shorted paths to next and then
                # choose one path at random in get_product()
                yield path + [next]
            else:
                queue.append((next, path + [next]))
                visited.append(next)

'''Function which returns a dictionary for a given user
   where each item represents the key in the dictionary
   and it returns a list of lists(paths) from user to item r-hop distance apart'''
def create_item_dict(all_path):
    dict_path = dict()
    for path in all_path:
        r_hop_item = path[-1]
        dict_path.setdefault(r_hop_item,[]).append(path)
    return dict_path

'''Function to get product from user to item in the path
   It chooses any path at random, if #paths > 1'''
def get_product(data_matrix, path):
    if len(path) < 1:
        return UNOBSERVED
    idx = random.randint(0, len(path)-1)    # in case of multiple paths to same item
    p = path[idx]                           # choose any one path at random

    product = 1
    for i in range(len(p)-1):
        product = product * data_matrix[p[i],p[i+1]]
    return product


# Functions useful for creating and operating on neighbor boundary (vector) matrices:

# In[ ]:


'''Function to generate product matrix from user to items
   (items which are at r-hop boundary from user)'''
def generate_product_matrix(graph, data_matrix, radius):

    # each u'th row in product_matrix represents a neighbor boundary vector for vertex u
    # therefore it is a (USERS+ITEMS) x (USERS+ITEMS) dimensional matrix
    product_matrix = np.full(((USERS+ITEMS), (USERS+ITEMS)), UNOBSERVED, dtype='float16')

    for user_vertex in tqdm(range(USERS+ITEMS)):               #a user_vertex may also be of an item
        all_path = list(bfs_paths(graph, user_vertex, radius)) # 1. get a list of all r-hop paths from given user
        dict_path = create_item_dict(all_path)                 # 2. create dict of paths from user to individual items
        for item_vertex in dict_path:                          #an item_vertex may also be of a user
            paths = dict_path[item_vertex]                     # 3. get the set of user-item paths
            product = get_product(data_matrix, paths)          # 4. get product for a unique user-item path (at random)
            product_matrix[user_vertex, item_vertex] = product # 5. store the product in the matrix
    return product_matrix

'''Function to normalize the product of paths in neighbor boundary vector for every u'th rowed user
   normalized along the same row'''
#TODO: implement it in efficient manner
def row_wise_normalize_matrix(data_matrix):
    n_neighbors_per_row = np.full((USERS+ITEMS), 0, dtype=float)
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            if data_matrix[i,j] != UNOBSERVED:
                n_neighbors_per_row[i] = n_neighbors_per_row[i] + 1

    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            if data_matrix[i,j] != UNOBSERVED and n_neighbors_per_row[i] > 0:
                data_matrix[i,j] = data_matrix[i,j] / n_neighbors_per_row[i]

    return data_matrix

import pandas as pd
'''Function to describe count of neighbors for every vertex and other values
   Also note, the values described might be slightly distorted because of symmetricity of neighbor matrices'''
def describe_neighbor_count(data_matrix):
    n_neighbor_vector = np.full((USERS+ITEMS), 0, dtype=float)
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            if data_matrix[i,j] != UNOBSERVED:
                n_neighbor_vector[i] = n_neighbor_vector[i] + 1

    df = pd.DataFrame(n_neighbor_vector)
    print('To effectively choose RADIUS value for next run of algorithm:')
    print('Showing distribution of count of neighbors for every vertex:')
    print(df.describe())


# Function which generates final neighbor boundary (vector) matrices at r-hop and r+1 hop distance:

# In[ ]:


import math
'''Function to return two product matrices
   one at r-hop distance and other at r+1 hop distance for dist1 computation'''
# if radius passed is less than 1 or not passed, this function evaluates the default radius as per paper
def generate_neighbor_boundary_matrix(data_csr):
    global RADIUS
    if RADIUS < 1:
        #TODO: Fix this
        print('ERROR: please do not use the radius formula as given in paper')
        print('     : the formula evaluates to a decimal values between 0 and 1')
        print('     : I am working on fixing this')
        RADIUS = (float(6) * math.log( 1.0 / SPARSITY)) / (8.0 * math.log( float(C1) * SPARSITY * (USERS + ITEMS)))
        return -1

    # First create the graph
    graph = create_dict_graph_from_csr(data_csr)              # to store the edges in adjacency list
    data_matrix = csr_to_matrix(data_csr, symmetry=True)      # to store the ratings as matrix

    radius = RADIUS
    print('Generating neighbor boundary matrix at {}-hop distance:'.format(radius))
    sys.stdout.flush()
    r_neighbor_matrix = generate_product_matrix(graph, data_matrix, radius=radius)
    r_neighbor_matrix = row_wise_normalize_matrix(r_neighbor_matrix)

    radius = radius+1
    print('Generating neighbor boundary matrix at {}-hop distance:'.format(radius))
    sys.stdout.flush()
    r1_neighbor_matrix = generate_product_matrix(graph, data_matrix, radius=radius)
    r1_neighbor_matrix = row_wise_normalize_matrix(r1_neighbor_matrix)

    return [r_neighbor_matrix, r1_neighbor_matrix]


# ### Step 3: Computing the distances
# Distance computation between two users (using matrix $M_2$) using the following formula (only $dist_1$ implemented for now):
#
# $$ dist(u,v) = \left(\frac{1 - c_1p}{c_2p}\right) (\tilde{N_{u,r}} - \tilde{N_{v,r}})^T M_2 (\tilde{N_{u,r+1}} - \tilde{N_{v,r+1}}) $$

# In[ ]:


def compute_distance_matrix(r_neighbor_matrix, r1_neighbor_matrix, m2_csr):
    m2_matrix = csr_to_matrix(m2_csr, symmetry=True)
    scale_factor = (1.0 - C1 * SPARSITY) / (C2 * SPARSITY)

    user_list = np.array(range(USERS+ITEMS))
    distance_matrix = np.full(((USERS+ITEMS), (USERS+ITEMS)), UNOBSERVED, dtype=float)

    print('Generating distance matrix')
    sys.stdout.flush()
    for user1 in tqdm(user_list):  # computing for all elements individually
        for user2 in user_list:    # not assuming any symmetricity for distance matrix
            user1_r_neighbor_vector = r_neighbor_matrix[user1]
            user2_r_neighbor_vector = r_neighbor_matrix[user2]
            r_neighbor_vector = user1_r_neighbor_vector - user2_r_neighbor_vector
            r_neighbor_vector = np.transpose(r_neighbor_vector)

            user1_r1_neighbor_vector = r1_neighbor_matrix[user1]
            user2_r1_neighbor_vector = r1_neighbor_matrix[user2]
            r1_neighbor_vector = user1_r1_neighbor_vector - user2_r1_neighbor_vector

            #print(r_neighbor_vector.shape)
            #print(m2_matrix.shape)
            dist_value = np.matmul(r_neighbor_vector, m2_matrix)
            dist_value = np.matmul(dist_value, r1_neighbor_vector)

            distance_matrix[user1,user2] = dist_value * scale_factor

    return distance_matrix

import pandas as pd
'''Function which gives an idea of how distance values are distributed to best choose THRESHOLD in Step 4'''
def describe_distance_matrix(distance_matrix):
    flat_distance_matrix = distance_matrix.flatten()
    observed = []
    for i in flat_distance_matrix:
        if i != UNOBSERVED:
            observed.append(i)

    df = pd.DataFrame(observed)
    print('To effectively choose THRESHOLD value in next step:')
    print('Showing distribution of non zero (or observed) entries of distance matrix:')
    print(df.describe())


# ### Step 4: Averaging datapoints to produce final estimate
# Average over nearby data points based on the distance(similarity) threshold $n_n$ (using matrix $M_3$). $n_n$ to be tuned using cross validation. Mathematically (from paper):
#
# $$ \hat{F_{u,v}} = \frac{1}{\mid E_{uv1} \mid} \sum_{(a,b) \in E_{uv1}} M_3(a,b) $$
# *where $E_{uv1}$ denotes the set of undirected edges $(a, b)$ such that $(a, b) \in E_3$ and both $dist(u, a)$ and $dist(v, b)$ are less than $n_n$*

# Function which generates a boolean similarity matrix which tells if $dist(u,a) < n_n$ or not:

# In[ ]:


'''Function to get similarity matrix using THRESHOLD'''
def generate_sim_matrix(distance_matrix, threshold):
    user_list = np.array(range(USERS+ITEMS))
    sim_matrix = np.full(((USERS+ITEMS), (USERS+ITEMS)), False, dtype=bool)

    print('Generating distance similarity matrix:')
    sys.stdout.flush()
    for user1 in tqdm(user_list):  # computing for all elements individually
        for user2 in user_list:    # not assuming any symmetricity for distance matrix
            if distance_matrix[user1,user2] != UNOBSERVED and distance_matrix[user1,user2] < threshold:
                sim_matrix[user1, user2] = True
    return sim_matrix


# Functions which are used to generate final predictions:

# In[ ]:


'''Function to get final prediction estimates for user-item ratings'''
def generate_averaged_prediction(u, v, sim_matrix, m3_matrix, bounded=True):
    prediction = 0
    n_prediction = 0

    # Making sure the vertex indices are ints
    u = int(u)
    v = int(v)

    for a in range(len(sim_matrix[u])):
        for b in range(len(sim_matrix[v])):
            if sim_matrix[u,a] and sim_matrix[v,b] and m3_matrix[a,b] != UNOBSERVED:
                prediction = prediction + m3_matrix[a,b]
                n_prediction = n_prediction + 1

    # the below two conditions drive inspiration from standard recommendation systems
    if n_prediction > 0:
        prediction = prediction / n_prediction
    else:
        prediction = UNPRED_RATING / 5                       # TODO: make it generic; here we assume ratings as 1 - 5

    if bounded:                                              #     : make it generic; here we assume ratings as 1 - 5
        if prediction > 1:
            prediction = 1
        elif prediction < 0.2:
            prediction = 0.2

    return prediction

'''Function to get final prediction estimates for user-item rating matrix
   Use this only if you want estimates for all the ratings'''
def generate_averaged_prediction_matrix(sim_matrix, m3_csr):
    m3_matrix = csr_to_matrix(m3_csr, symmetry=True)

    vertex_list = np.array(range(USERS+ITEMS))
    prediction_matrix = np.full(((USERS+ITEMS), (USERS+ITEMS)), UNOBSERVED, dtype='float16')

    print('Generating prediction matrix:')
    sys.stdout.flush()
    for u in tqdm(vertex_list):
        for v in vertex_list:
            prediction = generate_averaged_prediction(u, v, sim_matrix, m3_matrix, bounded=True)
            prediction_matrix[u, v] = prediction

    return prediction_matrix

'''Function to get final prediction estimates for given user-item list
   Use this only if you have a set of user-item pairs for which you want the final estimates
   Consider using this function for evaluation purposes(only)'''
def generate_averaged_prediction_array(sim_matrix, m3_csr, test_data_csr):
    m3_matrix = csr_to_matrix(m3_csr, symmetry=True)

    prediction_array = np.full((len(test_data_csr)), UNOBSERVED, dtype=float)

    print('Generating prediction array:')
    sys.stdout.flush()
    for i in tqdm(range(len(test_data_csr))):
        datapt = test_data_csr[i]
        # Considering only first two columns for rating estimation
        vertex1 = int(datapt[0])
        vertex2 = int(datapt[1])
        prediction = generate_averaged_prediction(vertex1, vertex2, sim_matrix, m3_matrix, bounded=True)
        prediction_array[i] = prediction
    return prediction_array


# # III. Other important functions

# ### Data Handling

# In[ ]:


''' Function to read data file, given in CSR format
    Assuming 1st 3 values of a row as: user_id, item_id, rating '''
def read_data_csr(fname, delimiter, dtype=float):
    data_csr = np.loadtxt(fname=fname, delimiter=delimiter, dtype=dtype) # Reading data to array
    data_csr = data_csr[:, :3]                                           # Extracting 1st 3 columns: 0,1,2
    if FIRST_INDEX == -1:                                                # Making sure user_id/item_id starts from 0
        first_index_user = min(data_csr[:,0])                            #    as it becomes easier to track in graphs
        first_index_item = min(data_csr[:,1])
        data_csr[:,0] = data_csr[:,0] - first_index_user
        data_csr[:,1] = data_csr[:,1] - first_index_item
    else:
        data_csr[:,0:2] = data_csr[:,0:2] - FIRST_INDEX
    return data_csr

''' Function to get data in CSR format for given data in matrix format '''
def matrix_to_csr(data_matrix):
    data_csr = np.array([ [i,j,data_matrix[i,j]]                         for i in range(len(data_matrix))                             for j in range(len(data_matrix[i]))                                if data_matrix[i,j] != UNOBSERVED])
    return data_csr


'''Function to find and replace some values
   for only 1d and 2d numpy arrays'''
def find_and_replace(data, find_value, replace_value):
    if len(data.shape) == 1:                # for 1D numpy array
        for i in range(len(data)):
            if data[i] == find_value:
                data[i] = replace_value
    elif len(data.shape) == 2:              # for 2D numpy array
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i,j] == find_value:
                    data[i,j] = replace_value
    return data

''' Function to check dataset'''
def check_and_set_data_csr(data_csr):
    global USERS, ITEMS, SPARSITY
    n_users = int(max(data_csr[:,0])) + 1
    n_items = int(max(data_csr[:,1])) + 1

    unique_users = len(np.array(list(set(data_csr[:,0]))))
    unique_items = len(np.array(list(set(data_csr[:,1]))))

    if USERS == -1:
        USERS = n_users
    if ITEMS == -1:
        ITEMS = n_items

    print('USERS = ' + str(USERS))
    print('ITEMS = ' + str(ITEMS))

    #checking if global USERS/ITEMS had wrong values entered:
    if n_users != USERS:
        print('ERROR: USERS entered by you is wrong. {} users found in dataset'.format(n_users))
    if n_items != ITEMS:
        print('ERROR: ITEMS entered by you is wrong. {} items found in dataset'.format(n_items))

    # checking unrated users/items : this is possible if some user/item index gets skipped in dataset
    if n_users != unique_users:
        print('ERROR: No. of users with no ratings: ' + str(n_users - unique_users))
        print('     : This notebook may not be robust to such dataset')
    if n_items != unique_items:
        print('ERROR: No. of items with no ratings: ' + str(n_items - unique_items))
        print('     : This notebook may not be robust to such dataset')
    if n_users == unique_users and n_items == unique_items:
        print('All users and items have at least one rating! Good!')

    #checking sparsity for large symmetricized matrix
    sparsity_symm =  float(2 * N_RATINGS) / ((USERS + ITEMS)**2)

    if SPARSITY == -1:
        SPARSITY = sparsity_symm
    print('SPARSITY (p) = ' + str(SPARSITY))
    if SPARSITY != sparsity_symm:
        print('ERROR: SPARSITY entered by you is wrong. {} sparsity found in dataset'.format(sparsity_symm))

    if sparsity_symm <= (float(1) / ((n_users + n_items)**2)):
        print('WARNING: For generated large symmetric matrix:')
        print('       : p is not polynomially larger than 1/n.')
        print('       : Using dist1 as distance computation may not gurantee that')
        print('       : expected square error converges to zero using this paper\'s algorithm.')
    else:
        print('Sym matrix : p is polynomially larger than 1/n, all guarantees applicable')
    print('Check and set dataset : done')

'''Function to generate training and testing data split from given CSR data'''
def generate_train_test_split_csr(data_csr, split, shuffle=True):
    # we use data_csr as it is easy to only shuffle it and accordingly create train and test set
    if shuffle:
        np.random.shuffle(data_csr) # inplace shuffle

    train_sz = int((1 - split) * data_csr.shape[0])

    train_data_csr = data_csr[: train_sz ,:]
    test_data_csr = data_csr[train_sz : ,:]

    if train_data_csr.shape[0]+test_data_csr.shape[0] == data_csr.shape[0]:
        print('Generating train test split: done')
    else:
        print('Generating train test split: FAILED')
    return [train_data_csr, test_data_csr]

'''Function to force reduce the size of dataset
   To be used only for testing purposes
   Note: this doesnt ensure if every item has a rating or not: TODO'''
def reduce_size_of_data_csr(data_csr):
    global N_RATINGS
    print('WARNING: FOR TESTING PURPOSES ONLY')
    if USER_LIMIT < 1 or ITEM_LIMIT < 1:
        print('ERROR: please set limits > 0')
        print('     : using same sataset without any reductions')
        return data_csr

    data_csr = data_csr[((data_csr[:,0] < USER_LIMIT)*(data_csr[:,1] < ITEM_LIMIT))]

    # Accounting for unvisited users
    visited = np.full((USER_LIMIT), True, dtype=bool)
    for i in data_csr:
        visited[int(i[0])] = False
    unvisited_users = [i for i in range(len(visited)) if visited[i]]
    # adding 1 rating for every unvisited user
    for i in unvisited_users:
        data_csr = np.append(data_csr, [[i, i, 3]], axis=0)

    N_RATINGS = data_csr.shape[0]
    return data_csr


# ### Evaluation
# We evaluate our recommendation algorithm using RMSE (root mean square error). <br>
# According to paper, if sparsity $p$ is polynomially larger than $n^{-1}$, i.e. if $p = n^{-1 + \epsilon}$ for $\epsilon > 0$, then we can safely use $dist_1$ distance computation formula and MSE is bounded by $O((pn)^{-1/5})$.

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt


'''Function to generate true and test labels from test_data_csr and predicted_matrix
   This function may not be required for evaluation purposes(only)'''
def generate_true_and_test_labels(test_data_csr, predicted_matrix):
    # for all the available ratings in testset
    # and for all the predicted rating for those available rating
    # put them in two separate vectors
    y_actual  = np.full((len(test_data_csr)), UNOBSERVED, dtype=float)
    y_predict = np.full((len(test_data_csr)), UNOBSERVED, dtype=float)

    print('Generating true and test label:')
    sys.stdout.flush()
    for i in tqdm(range(len(test_data_csr))):
        testpt = test_data_csr[i]
        y_actual[i]  = testpt[2]
        y_predict[i] = predicted_matrix[testpt[0], testpt[1]]
        if y_predict[i] == UNOBSERVED:       # i.e. we could not generate a rating for this test user item pair
            y_predict[i] = AVG_RATING
    return [y_actual, y_predict]

'''Function to get Mean Squared Error for given actual and predicted array'''
def get_mse(y_actual, y_predict):
    mse = mean_squared_error(y_actual, y_predict)
    return mse

'''Function to get ROOT Mean Squared Error for given actual and predicted array'''
def get_rmse(y_actual, y_predict):
    rmse = sqrt(mean_squared_error(y_actual, y_predict))
    return rmse

'''Function to get Average Error for given actual and predicted array'''
def get_avg_err(y_actual, y_predict):
    avg_err = sum(abs(y_actual - y_predict)) / len(y_actual)
    return avg_err

'''Function to check if obtained MSE is within the bound as calculated in the paper'''
def check_mse(data_csr, y_actual, y_predict):
    mse_upper_bound = (SPARSITY * (USERS+ITEMS)) ** (-1 / float(5))
    print('MSE Upper bound: {}'.format(mse_upper_bound))
    mse = get_mse(y_actual, y_predict)
    print('MSE of predictions: {}'.format(mse))
    if mse < mse_upper_bound:
        print('As per the discussion in the paper, MSE is bounded by O((pn)**(-1/5))')
    else:
        print('ERROR: Contrary to the discusssion in the paper, MSE is NOT bounded by O((pn)**(-1/5))')



# # IV: Test Script / Experiment
# The following jupyter notebook cells make calls to above cells to run experiments on a recommendation dataset.
#
# Please prefer using the testscripts like testscript_std.ipynb to run experiments. Use the below testscript layout only if you want to debug/modify some functions in this notebook.

# In[ ]:


#datetime.now().time()     # (hour, min, sec, microsec)


# ### Setting constants

# ### Read and prepare the dataset

# ### Make predictions using THE algorithm

# ##### Step 1: Sample splitting

# ##### Step 2: Expanding the Neighborhood

# ##### Step 3: Computing the distances

# ##### Step 4: Averaging datapoints to produce final estimate

# ### Evaluate the predictions

# In[ ]:


#datetime.now().time()     # (hour, min, sec, microsec)
