
# coding: utf-8

# # Test script
# This is a sample test script to run on very_small_graph.txt dataset. This script is created to help you better understand how exactly the functions operate.

# Running the main Jupyter notebook which has all the functions defined. Make sure the path is correct in next cell

# In[1]:


from datetime import datetime
datetime.now().time()     # (hour, min, sec, microsec)


# In[2]:


# if this way of importing another jupyter notebook fails for you
# then you can use any one of the many methods described here:
# https://stackoverflow.com/questions/20186344/ipynb-import-another-ipynb-file
# get_ipython().run_line_magic('run', "'../src/finalcode.ipynb'")


# In[3]:


datetime.now().time()     # (hour, min, sec, microsec)


# ### Setting constants

# In[4]:


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

from finalcode import *
# ### Read and prepare the dataset

# In[5]:


m1_csr = read_data_csr(fname='training.txt', delimiter=",", dtype=int)
m2_csr = read_data_csr(fname='video_small_testing_num.csv', delimiter=",", dtype=int)
check_and_set_data_csr(data_csr=m1_csr)


# In[6]:


m1_csr = normalize_ratings_csr(m1_csr)          ##### REMOVE THIS CELL
m1_csr = csr_to_symmetric_csr(m1_csr)


# ### Make predictions using THE algorithm

# ##### Step 1: Sample splitting

# In[7]:


# This step is  being skipped (not needed) for very_small_graph.txt dataset


# ##### Step 2: Expanding the Neighborhood

# In[8]:
# import gc
# gc.collect()
[r_neighbor_matrix, r1_neighbor_matrix] = generate_neighbor_boundary_matrix(m1_csr)
# all neighbor boundary vector for each user u is stored as u'th row in neighbor_matrix
# though here the vector is stored a row vector, we will treat it as column vector in Step 4
# Note: we might expect neighbor matrix to be symmetric with dimensions (USERS+ITEMS)*(USERS+ITEMS)
#     : since distance user-item and item-user should be same
#     : but this is not the case since there might be multiple paths between user-item
#     : and the random path picked for user-item and item-user may not be same
#     : normalizing the matrix also will result to rise of difference


# In[9]:


describe_neighbor_count(r_neighbor_matrix)


# In[10]:


describe_neighbor_count(r1_neighbor_matrix)


# ##### Step 3: Computing the distances

# In[11]:


distance_matrix = compute_distance_matrix(r_neighbor_matrix, r1_neighbor_matrix, m1_csr)
distance_matrix


# In[12]:


describe_distance_matrix(distance_matrix)


# ##### Step 4: Averaging datapoints to produce final estimate

# In[13]:


sim_matrix = generate_sim_matrix(distance_matrix, threshold=.26)
sim_matrix


# In[14]:


prediction_array = generate_averaged_prediction_array(sim_matrix, m1_csr, m2_csr)
prediction_array


# In[15]:


# prediction_matrix = generate_averaged_prediction_matrix(sim_matrix, m1_csr)
# prediction_matrix


# ### Evaluate the predictions

# In[16]:


# We have already prepared the test data (required for our algorithm)
test_data_csr = m2_csr
y_actual  = test_data_csr[:,2]
y_predict = prediction_array
# If we want, we could scale our ratings back to 1 - 5 range for evaluation purposes
#But then paper makes no guarantees about scaled ratings
#y_actual  = y_actual * 5
#y_predict = y_actual * 5


# In[17]:


get_rmse(y_actual, y_predict)


# In[18]:


get_avg_err(y_actual, y_predict)


# In[19]:


check_mse(m1_csr, y_actual, y_predict)


# In[20]:


datetime.now().time()     # (hour, min, sec, microsec)
