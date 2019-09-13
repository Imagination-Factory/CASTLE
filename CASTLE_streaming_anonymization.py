#!/usr/bin/env python
# coding: utf-8

# <h1>k-Anonymity on Streaming data</h1>
# 
# In this implementation we incorporate the CASTLE framework (concepts). The idea is to provide an anonymization of a data stream by also ensuring some quality of the anonymized output data stream.
# 
# _Implementation written by Christian Becker._

# <h2>Setting up the environment:</h2>

# In[1]:


# we use Pandas to work with the data as it makes working with categorical data very easy
import pandas as pd
# we use random for choosing random items of a list
import random


# In[2]:


# the quasi attributes used in this approach
# this is a list of the column names in our dataset (as the file doesn't contain any headers)
column_names = (
    'one',
    'two', 
    'three', 
    'four',
    'five',
    'six',
)

# if there are categorical attributes:
# these fields will require some special treatment
categorical = set((
  #  'education',
))
# we load the data example from the txt using panda's library function
df = pd.read_csv("ExampleListOfMAcadresses.txt", sep=":", header=None, names=column_names, index_col=False, engine='python');


# Modification of the sample data:

# In[3]:


# set the categorical attributes as types as such
for name in categorical:
    df[name] = df[name].astype('category')
   
def input_conversion():
    #convert hex string values into int values
    for name in column_names:
        df[name] = df[name].apply(lambda x: int(x, 16))

    df.head()
    
input_conversion()


# In[ ]:





# In[ ]:





# <h3>Definition of a cluster</h3>
# We work with ks-anonymized clusters where we have an n-dimensional space defined by intervals of values by the respective tuples of the tuples of the respective cluster.

# In[4]:


# create a dataframe with columns same as the input is
non_ks_clusters = [] # initially empty , array of DataFrame(columns=column_names_cluster) # stored in memory
ks_clusters = [] # already outputted, see above

# each attribute is stored as a dictionary which defines an interval
attr_range = {
  "min":  "NaN",
  "max": "NaN"
}

tau_global = 10 # initialize a global var (will be overwrittten in main function)


# In[5]:


# column names for the non-ks-clusters in memory because they also need to contain a list
# of the indezes of the tuples in order to know in which cluster a tuple falls or which and
# how many tuples have (already) been associated with a cluster
column_names_cluster = (
    'one',
    'two', 
    'three', 
    'four',
    'five',
    'six',
    'tuples_IDs',
    'clusterID'
)

#Creation of a non_ks_cluster
def create_new_cluster(new_tuple, new_cluster_index):
    global non_ks_clusters
    global number_of_cluster_indezes
    
    # create new cluster
    new_non_ks_cluster = pd.DataFrame(columns=column_names_cluster)
    for i_c, column in enumerate(new_non_ks_cluster):
        if column != 'tuples_IDs' and column != 'clusterID':
            # when normal attributes of the cluster/ dimensions of clusters constructed
            # min value equal to tuple's value
            attr_range = {
                "min": new_tuple[column],
                "max": new_tuple[column]
            }
            # set the only entry in the new non-ks-cluster to the attribute range
            new_non_ks_cluster.at[0, column] = [attr_range]
        elif column == 'tuples_IDs':
            # if the column to write is the list of associated tuples
            # set the tuples ID into the cluster
            new_non_ks_cluster.at[0, column] = [new_tuple["index"]]
    
    new_non_ks_cluster.at[0, 'clusterID'] = new_cluster_index
    # add newly created non-ks-cluster to the array
    non_ks_clusters.append(new_non_ks_cluster)
    #print("Successfully created new cluster for tuple ID: ", str(new_tuple["index"]), " - cluster ID: ", new_cluster_index)
    number_of_cluster_indezes += 1
    return new_non_ks_cluster


# In the following code parts we calculate information loss metrics and enlargement values when tuples are being added to existing clusters:

# In[6]:


def info_loss_cluster(current_cluster):
    # calculate the information loss of a specific cluster
    
    # calculate current info loss
    added_loss_values = 0
    # calculate current info loss of generalization
    for i_c, column in enumerate(current_cluster):
        if column != 'tuples_IDs' and column != 'clusterID':
            # get the range of the cluster
            attr_range = current_cluster.at[0, column]
            l_min_value = attr_range[0].get('min', "0")
            u_max_value = attr_range[0].get('max', "0")
        
            span_current_attr = u_max_value - l_min_value
            
            U = 255
            L = 0
            span_domain = U - L # depends on the real data
    
            info_loss_attr = span_current_attr / span_domain
            added_loss_values = added_loss_values + info_loss_attr
    # divide by number of attributes (HERE = 6) to obtain info loss of current generalization
    added_loss_values = added_loss_values / 6
    
    return added_loss_values


# In[ ]:





# In[7]:


# calculate the enlargement caused when tuple may be added to cluster
def enlargement_calculation(new_tuple, non_ks_cluster_item):
    # number of attributes
    n = 6 # to be adapted to real data, HERE = 6
    
    ## calculate current info loss
    #current_info_loss = info_loss_cluster(current_cluster=non_ks_cluster_item)
    
    
    added_info_loss_values = 0
    # calculate current info loss of generalization
    for i_c, column in enumerate(non_ks_cluster_item):
        if column != 'tuples_IDs' and column != 'clusterID':
            # get the range of the cluster
            attr_range = non_ks_cluster_item.at[0, column]
            l_min_value = attr_range[0].get('min', "0")
            u_max_value = attr_range[0].get('max', "0")
        
            span_current_attr = u_max_value - l_min_value
            
            U = 255
            L = 0
            span_domain = U - L # depends on the real data
    
            # the current info loss of this attribute
            info_loss_current = span_current_attr / span_domain
        
            # when the tuple would be added:
            
            # get value of current tuple
            tuple_value = new_tuple[column]
            # calculate new max ranges
            l_new_min_value = min(l_min_value, tuple_value)
            u_new_max_value = max(u_max_value, tuple_value)
            span_new_current_attr = u_new_max_value - l_new_min_value
            # the new info loss of this attribute after adding of the tuple
            info_loss_new = span_new_current_attr / span_domain
        
            # with each dimension/ each attribute add the calculated info loss difference
            added_info_loss_values = added_info_loss_values + (info_loss_new - info_loss_current)
    
    # calculate final enlargement value for adding this tuple to this cluster
    # divide by number of attributes (HERE = 6) to obtain info loss of current generalization
    added_info_loss_values = added_info_loss_values / n
    
    #print("enlargement value for cluster ID=", non_ks_cluster_item.at[0,'clusterID'], "e=", added_info_loss_values)
    
    return added_info_loss_values


# In[8]:


# calculate the enlargement caused when two clusters would be merged
def enlargement_clusters_calculation(cluster1, cluster2):
    # number of attributes
    n = 6 # to be adapted to real data, HERE = 6
    
    added_info_loss_values = 0
    # calculate current info loss of generalization
    for i_c, column in enumerate(cluster1):
        if column != 'tuples_IDs' and column != 'clusterID':
            # get the range of the cluster1
            attr_range = cluster1.at[0, column]
            l_min_value = attr_range[0].get('min', "0")
            u_max_value = attr_range[0].get('max', "0")
        
            span_current_attr = u_max_value - l_min_value
            
            U = 255
            L = 0
            span_domain = U - L # depends on the real data
    
            # the current info loss of this attribute
            info_loss_current = span_current_attr / span_domain
        
            # when the second cluster would be added:
            
            # get range of second cluster
            attr_range2 = cluster2.at[0, column]
            l_min_value2 = attr_range2[0].get('min', "0")
            u_max_value2 = attr_range2[0].get('max', "0")
    
            # calculate new min and max ranges
            l_new_min_value = min(l_min_value, l_min_value2)
            u_new_max_value = max(u_max_value, u_max_value2)
            span_new_current_attr = u_new_max_value - l_new_min_value
            # the new info loss of this attribute after adding of the second cluster
            info_loss_new = span_new_current_attr / span_domain
        
            # with each dimension/ each attribute add the calculated info loss difference
            added_info_loss_values = added_info_loss_values + (info_loss_new - info_loss_current)
    
    # calculate final enlargement value for adding this cluster to the other cluster
    # divide by number of attributes (HERE = 6) to obtain info loss of current generalization
    added_info_loss_values = added_info_loss_values / n
    
    return added_info_loss_values
    


# The following Best-Selection method figures out where to put a new tuple into. It checks whether there is already a cluster whose generalization entails the tuple already or whether a new cluster needs to be generated:

# In[9]:


# Best selection of a cluster where a tuple can be pushed into
def best_selection(new_tuple, k, betha):
    # parameter tau can be set initially in main and is influenced by last outputted ks-clusters
    # e.g. tau = 10
    # parameter betha can be set and influenced
    # betha = 5000
    global non_ks_clusters
    global tau_global
    global number_of_cluster_indezes
    
    #list of minimum enlargement clusters
    min_e_clusters = []
    min_e = 999999 # set initially the minimum enlargement very high
    index_of_current_non_ks_cluster = 0
    
    for cluster_item in non_ks_clusters:
        # calculate enlargement caused by adding tuple
        calculated_e = enlargement_calculation(new_tuple, cluster_item)
        #print("Calculated possible enlargement: " + str(calculated_e))
        if calculated_e < min_e:
            # replace the existing clusters in the min e clusters list
            min_e_clusters.clear()
            min_e_clusters.append(index_of_current_non_ks_cluster)# = calculated_e #[cluster_item] = calculated_e
            min_e = calculated_e
        else:
            if calculated_e == min_e:
                # add it to the min e clusters
                min_e_clusters.append(index_of_current_non_ks_cluster)# = calculated_e #[cluster_item] = calculated_e
                
        # increase index of current cluster under observation
        index_of_current_non_ks_cluster = index_of_current_non_ks_cluster + 1
    
    print("Calculated min enlargement for tuple:", str(min_e))
    # check whether the current found clusters have enlargement smaller than tau
    if min_e <= tau_global:
        # return any of the clusters in this list
        return non_ks_clusters[random.choice(min_e_clusters)]
    else:
        # create new cluster if possible
        if len(non_ks_clusters) >= betha:
            # return any cluster which is minimal because no new cluster can be created
            return non_ks_clusters[random.choice(min_e_clusters)]
        else:
            # create a new cluster
            return create_new_cluster(new_tuple, new_cluster_index = number_of_cluster_indezes)#len(non_ks_clusters))
    
    return "NULL"


# Some helper functions:

# In[10]:


def add_tuple_to_cluster(new_tuple, non_ks_cluster_item):
    # add a tuple to the calculated non-ks cluster or ks-cluster (LATER #TO DO)
    # calculate span updates of attributes
    
    # check whether this is not an item for which a new non-ks-cluster
    # has been created in this step
    if new_tuple["index"] in non_ks_cluster_item.at[0, "tuples_IDs"]:
        # then skip the "adding this tuple again" part and simply return the cluster
        return non_ks_cluster_item
    
    for i_c, column in enumerate(non_ks_cluster_item):
        if column != 'tuples_IDs' and column != 'clusterID':
            # get the range of the cluster
            attr_range = non_ks_cluster_item.at[0, column][0]
            min_value = attr_range.get('min', "0")
            max_value = attr_range.get('max', "0")
        
            # get value of this column of the new tuple to be added
            current_value = new_tuple[column]
            if min_value > current_value:
                # update min if tuple has smaller value
                attr_range['min'] = current_value
                non_ks_cluster_item.at[0, column] = [attr_range]
            if max_value < current_value:
                # update max if tuple has smaller value
                attr_range['max'] = current_value
                non_ks_cluster_item.at[0, column] = [attr_range]
        elif column == 'tuples_IDs':
            # add tuple ID to the list of tuple ID of this cluster
            # set the tuples ID into the cluster
            non_ks_cluster_item.at[0, column].append(new_tuple["index"])
            
            
    print("Tuple with ID: " + str(new_tuple["index"]) + " has been added to an existing non-ks-cluster")
    # return the updated non-ks-cluster
    return non_ks_cluster_item
            


# In[11]:


def check_tuple_already_outputted(target_count, expiring_tuple_ID):
    global output_tuple_max_index
    global output_list_of_tuples
    #print("Output?", output_tuple_max_index, "target?", target_count)
    # check whether expiring tuple already outputted or to be outputted now
    if expiring_tuple_ID in output_list_of_tuples:
        # already output
        return True
    else:
        return False
    
    #if output_tuple_max_index < target_count:
    #    # has to be outputted
    #    return False
    #else:
    #    # tuple has already been outputted
    #    return True


# In[12]:


def calculate_size_of_cluster(current_cluster):
    global stream_of_tuples
    
    temp_list = []
    unique_tuples_list = [temp_list.append(x) for x in current_cluster.at[0, 'tuples_IDs'] if x not in temp_list]
    # count the unique items in the list
    size_of_cluster = len(unique_tuples_list)
    
    if size_of_cluster > 1:
        # check whether not a duplet in tuples, meaning that individual MAC address captured several times
        distinct_mac_addresses = 0
        for one_entry in current_cluster.at[0, 'tuples_IDs']:
            #print(one_entry, type(one_entry))
            one_tuple = stream_of_tuples[int(one_entry)]
            one_tuple_unique = True
            for two_entry in current_cluster.at[0, 'tuples_IDs']:
                #check that not the same ID
                two_tuple = stream_of_tuples[int(two_entry)]
                if one_entry != two_entry:
                    # check whether values distinct
                    if one_tuple["one"] == two_tuple["one"] and one_tuple["two"] == two_tuple["two"] and one_tuple["three"] == two_tuple["three"] and one_tuple["four"] == two_tuple["four"] and one_tuple["five"] == two_tuple["five"] and one_tuple["six"] == two_tuple["six"]:
                        # if all are equal then same pid value = not distinct
                        one_tuple_unique = False
                                         
            # only if this one tuple was marked as unique then count it as a distinct MAC address, another distinct entry
            if (one_tuple_unique):
                distinct_mac_addresses = distinct_mac_addresses + 1
            
        # the size of a cluster is then the number of distinct mac addresses in the cluster
        size_of_cluster = distinct_mac_addresses
    #end of if whether size_non_ks_cluster > 1 because the size stays the same if just one tuple in there
    
    return size_of_cluster


# The following concentrates on splitting and merging clusters. Clusters need to be merged if they do not fulfill the criteria to be considered a ks-anonymized cluster (e.g., size less than k). Splitting is a task performed in order to increase the quality of the outputted data stream. If clusters are splitted than eventually the information loss of each cluster can be reduced.
# 
# _Splitting_ 
# 
# _Merging of clusters_

# In[13]:


from heapq import heappush, heappop, heapreplace
from functools import total_ordering
#following class used to store the tuples with their distance values in the heap
@total_ordering
class KeyDict(object):
    def __init__(self, key, dct):
        self.key = key
        self.dct = dct

    def __lt__(self, other):
        return self.key < other#.key

    def __eq__(self, other):
        return self.key == other#.key

    def __repr__(self):
        return '{0.__class__.__name__}(key={0.key}, dct={0.dct})'.format(self)


# In[14]:


def split(current_non_ks_cluster, k):
    global stream_of_tuples # we need to be able to access every distinct individual tuple with its attributes
    global number_of_cluster_indezes # needed in order to create a new cluster which seemleasly integrates into
    # set of non-ks-clusters in memory
    
    print("CASTLE - SPLIT requested..")
    
    
    SC = []
    
    # as next we need to check that we have k distinct individuals
    # we can do this by using one of the pid-Attributes of the tuple
    # TODO: adapt to real data use case
    
    # get the tuple list we need to work on
    tuple_list_split_cluster = []
    tuple_ID_list = current_non_ks_cluster.at[0, 'tuples_IDs']
    for individual_ID in tuple_ID_list:
        # search for the corresponding tuple
        for fitting_tuple in stream_of_tuples:
            if fitting_tuple["index"] == individual_ID:
                # tuple found
                # save the tuple in the internal tuple list for this splitting function
                tuple_list_split_cluster.append(fitting_tuple)#stream_of_tuples)
                
                break # we can break and escape inner for loop for saving computation
                # because once we found the corresponding tuple we can search for the
                # next one by directly continuing with the outter for loop
    
    # now we distribute the tuples over the buckets array BS
    # in order to have in each bucket tuples grouped by
    # their pid values
    BS = []
    for individual_tuple in tuple_list_split_cluster:
            tuple_has_new_pid = True
            # check whether there is already a bucket with this pid value
            for single_bucket in BS:
                tuple_2 = random.choice(single_bucket)
                #check that not the same ID
                #print("DEBUG: SPLIT: ", individual_tuple, "tuple_2: ", tuple_2)
                if individual_tuple["index"] != tuple_2["index"]:
                    # check whether values distinct
                    if individual_tuple["one"] == tuple_2["one"] and individual_tuple["two"] == tuple_2["two"] and individual_tuple["three"] == tuple_2["three"] and individual_tuple["four"] == tuple_2["four"] and individual_tuple["five"] == tuple_2["five"] and individual_tuple["six"] == tuple_2["six"]:
                        # if all are equal mac address not distinct but equal
                        tuple_has_new_pid = False
                        # add tuple to this bucket
                        ## search in which bucket second tuple is
                        #for buckets_item in BS:
                        #    if tuple_2 in buckets_item:
                        #        # add this tuple into this grouped bucket by pid values
                        #        buckets_item.append(individual_tuple)
                        #        # automatically BS gets updated by this
                        single_bucket.append(individual_tuple)

                        break # we can escape the for loop for searching for another match because already added to one bucket with same pid value
            # if there was not any other tuple in BS
            if (tuple_has_new_pid):
                # create a new bucket for this one tuple
                BS.append([individual_tuple])

    # now BS contains the buckets of tuples grouped by pid values

    ## here we can see the list of tuples as the buckets when we
    ## assume that they are all with different pid values
    #BS = tuple_list_split_cluster
    
    while len(BS) >= k:
    
        # we randomly select a bucket with corresponding contained tuples
        t_chosen_new_cluster = random.choice(BS)
        # we randomly select a tuple of this bucket, called "t dash"
        t_chosen_tuple = random.choice(t_chosen_new_cluster)
        # remove the picked tuple out of list
        t_chosen_new_cluster.remove(t_chosen_tuple)
        # remove the picked bucket out of list if it got empty
        if len(t_chosen_new_cluster) < 1:
            BS.remove(t_chosen_new_cluster)
        #print("t-chosen-new-cluster (t dash) diagnose", t_chosen_tuple)
        # create a new cluster around tuple
        C_new = create_new_cluster(new_tuple=t_chosen_tuple, new_cluster_index=(number_of_cluster_indezes+1))
        # this new C_new cluster is therefore now also in non-ks-clusters list in memory
        # side node: if we have a fast performing server and many new incoming tuples during
        # the split function that it might happen that other tuples are also gonna be added to this
        # new cluster C_new while computing the split function (but here not of further relevance)
        
        # next we generate a heap with k-1 nodes (because first tuple is already added to C_new)
        # heap creation with instantiation with infinite distances
        
        # we create a heap with k-1 nodes, initialized with infinite distance to t_dash
        heap = []
        for x in range(k-1): # iterates from 0 to k-2 meaning we get k-1 nodes into the heap
            heappush(heap, KeyDict(9999999, {'1':x})) #int(float('inf')), {'1': x})) # initialized with an infinite value

        #print("DEBUG: TEST: Key Dict form initialization:", KeyDict(9999999, {'1':x}))

        # for each remaining bucket in BS calulate distances to cluster C_new
        # therefore we pick one tuple out of each bucket
        for bucket_item in BS:
            t_picked_for_comparison = random.choice(bucket_item) # picked t
            
            # check if distance of this tuple closer to cluster than the one of the heap's root node

            # we calculate distance of this tuple to the tuple t_dash:
            distance_t = enlargement_calculation(new_tuple=t_picked_for_comparison, non_ks_cluster_item=t_chosen_new_cluster) # here we calculate distance of this tuple to the B_dash cluster

            #print("DEBUG: heap root:", heap[0])
            if distance_t < heap[0]:
                distance_t = heapreplace(heap, KeyDict(distance_t, t_picked_for_comparison))

        
        # after arranging the heap according to min distances/ min enlargements
        # iterate over heap and add tuples to C_new
        for i in range(len(heap)):
            current_heap_node = heappop(heap)
            current_t_tilde = current_heap_node.dct

            #print("DEBUG: Current t_tilde:", current_t_tilde)
            
            # insert t_tilde into C_new
            C_new = add_tuple_to_cluster(new_tuple=current_t_tilde, non_ks_cluster_item=C_new)
            # get bucket where t_tilde was contained and remove t_tilde out of bucket
            for bucket_item in BS:
                t_tilde_index = current_t_tilde["index"]
                for i_tuple in bucket_item:
                    if t_tilde_index == i_tuple["index"]:
                        bucket_item.remove(current_t_tilde)
                        # check if the bucket is empty after removing t_tilde
                        if len(bucket_item) < 1:
                            # remove bucket
                            BS.remove(bucket_item)
                        break # we can end the search for the bucket


        
        # After filling up C_new with enough tuples we can add C_new to SC
        SC.append(C_new)
    
    #print("Split: First half of split operation completed, now the remaining tuples will be distributed.")

    # Once less than k tuples/ buckets left we distribute remaining tuples over newly created clusters
    for bucket_item in BS:
        # for each bucket B_i we take a t_i and see which cluster has min enlargment when adding t_i
        # calculate closest cluster to which tuple(s) should be added
        # pick a random tuple out of this bucket, called t_i
        t_i = random.choice(bucket_item)

        # find t_i's nearest cluster meaning minimal enlargement
        #list of minimum enlargement clusters
        min_e_clusters = []
        min_e = 999999 # set initially the minimum enlargement very high
        index_of_current_non_ks_cluster = 0
        
        for cluster_item in SC:
            # calculate enlargement caused by adding tuple
            calculated_e = enlargement_calculation(t_i, cluster_item)
            #print("Calculated possible enlargement: " + str(calculated_e))
            if calculated_e < min_e:
                # replace the existing clusters in the min e clusters list
                min_e_clusters.clear()
                min_e_clusters.append(index_of_current_non_ks_cluster)
                min_e = calculated_e
            else:
                if calculated_e == min_e:
                    # add it to the min e clusters
                    min_e_clusters.append(index_of_current_non_ks_cluster)
                    
            # increase index of current cluster under observation
            index_of_current_non_ks_cluster = index_of_current_non_ks_cluster + 1
        
        #print("Calculated min enlargement for tuple:", str(min_e))
        # get one of these nearest clusters in regard to t_i
        nearest_SC_cluster = SC[random.choice(min_e_clusters)]
        # add each tuple of current B_i to this chosen cluster
        for tuple_within_BS in bucket_item:
            # we add tuple to cluster and update it each step with added tuples
            nearest_SC_cluster = add_tuple_to_cluster(new_tuple = tuple_within_BS, non_ks_cluster_item = nearest_SC_cluster)
            
        # after adding all the remaining tuples we can delete this bucket B_i
        BS.remove(bucket_item)
    # end of for each B_i in BS loop
    #print("..")
    #print("We would like to SPLIT - but to be implemented later")


    # at the end of this split function we need to delete the previous large cluster from memory
    # delete cluster C from set of non-ks-clusters
    intermediate_index = 0
    for cluster_item in non_ks_clusters: # maybe the for loop and if are not needed if removing of cluster works
        if cluster_item.at[0, 'clusterID'] == current_non_ks_cluster.at[0, 'clusterID']:
            #print("Remove cluster ID = ", cluster_item.at[0, 'clusterID'], " from non-ks-cluster set.")
            del non_ks_clusters[intermediate_index]
        intermediate_index += 1

    print("Splitting completed. We have ", len(SC), "new clusters after the split operation.")
    
    return SC


# In[15]:


def merge_clusters(expiring_cluster, set_of_other_clusters, k):
    global non_ks_clusters
    
    merged_clusters = expiring_cluster
    while calculate_size_of_cluster(merged_clusters) < k:
        min_enlargement = 100000000 # high number
        min_enlargement_cluster = set_of_other_clusters[0]
        # for each cluster calculate the enlargement when merged together
        for cluster_item in set_of_other_clusters:
            # calculate possible enlargement when merging clusters
            
            current_e = enlargement_clusters_calculation(cluster1=merged_clusters, cluster2=cluster_item)
            
            if current_e < min_enlargement:
                min_enlargement_cluster = cluster_item
        
        # merge the cluster with this new min enlargement cluster
        for i_c, column in enumerate(merged_clusters):
            if column != 'tuples_IDs' and column != 'clusterID':
                # get the range of the cluster
                attr_range = merged_clusters.at[0, column][0]
                min_value = attr_range.get('min', "0")
                max_value = attr_range.get('max', "0")
                # get the range of the to be added cluster
                attr_range2 = min_enlargement_cluster.at[0, column][0]
                min_value2 = attr_range2.get('min', "0")
                max_value2 = attr_range2.get('max', "0")
        
                if min_value > min_value2:
                    # update min if new cluster has smaller value
                    attr_range['min'] = min_value2
                    merged_clusters.at[0, column] = [attr_range]
                if max_value < max_value2:
                    # update max if new cluster has higher value
                    attr_range['max'] = max_value2
                    merged_clusters.at[0, column] = [attr_range]
            elif column == 'tuples_IDs':
                # add tuple ID to the list of tuple ID of this cluster
                merged_clusters.at[0, column].extend(min_enlargement_cluster.at[0, column])
        
        # delete min enlargement cluster from set of other clusters
        intermediate_index = 0
        for cluster_item in set_of_other_clusters: # maybe the for loop and if are not needed if removing of cluster works
            if cluster_item.at[0, 'clusterID'] == min_enlargement_cluster.at[0, 'clusterID']:
                del set_of_other_clusters[intermediate_index]
            intermediate_index += 1
            
        # delete min enlargement cluster from set of non-ks-clusters
        intermediate_index = 0
        for cluster_item in non_ks_clusters: # maybe the for loop and if are not needed if removing of cluster works
            if cluster_item.at[0, 'clusterID'] == min_enlargement_cluster.at[0, 'clusterID']:
                #print("Remove cluster ID = ", cluster_item.at[0, 'clusterID'], " from non-ks-cluster set.")
                del non_ks_clusters[intermediate_index]
            intermediate_index += 1
        
        print("..successfully merged cluster ID = ", min_enlargement_cluster.at[0, 'clusterID'], " to the current cluster..")
    
    return merged_clusters


# The following concentrates on the outputting of clusters.
# _The special case of outputting is suppression when the most general generalization needs to be applied._
# The methods thereafter simply concentrate on outputting the tuples already generalized with their respective cluster generalizations:

# In[16]:


# output the tuple as generalization in csv file
def official_outputting(generalization): 
    #print("Saving to file..")
    with open("generalization_output.txt", "a") as f:
        f.write(generalization + "\n")


# In[17]:


# Suppression of tuple if no cluster generalization can be applied
def suppress_tuple(single_tuple):
    global non_ks_clusters
    
    # Output
    output_string = ""
    # build the current attribute string
    for i_c, column in enumerate(single_tuple):
        if column != 'tuples_IDs' and column != 'clusterID':
            output_string = output_string + str(column) + ","
                
    output_string = output_string + " with G=[]"
    
    # build the generalization string
    generalization_string = ""
    generalization_string = "[" + str(0) + "-" + str(255) + "]"
    generalization_string = generalization_string + "[" + str(0) + "-" + str(255) + "]"
    generalization_string = generalization_string + "[" + str(0) + "-" + str(255) + "]"
    generalization_string = generalization_string + "[" + str(0) + "-" + str(255) + "]"
    generalization_string = generalization_string + "[" + str(0) + "-" + str(255) + "]"
    generalization_string = generalization_string + "[" + str(0) + "-" + str(255) + "]"
    
    output_string = output_string + generalization_string
    
    print("Output of Tuple ID =",str(single_tuple["index"]), ":", output_string)
    
    #official_outputting(generalization = generalization_string) # saving to output file
    
    # save tuple as already outputted:
    output_list_of_tuples.append(single_tuple["index"])
    
    # delete non-ks-cluster entries the tuple was inside beforehand
    intermediate_index = 0
    for cluster_item in non_ks_clusters:
        for i_c, column in enumerate(cluster_item):
            # delete the tuple ID out of this cluster
            if column == 'tuples_IDs':
                if single_tuple["index"] in cluster_item.at[0, column]:
                    # delete tuple out of this list
                    cluster_item.at[0, column].remove(single_tuple["index"])
                    # if the non-ks-cluster had only this single tuple inside then delete cluster
                    if len(cluster_item.at[0, column]) == 0:
                        # delete the cluster from the non-ks-anonymized cluster set
                        print("Remove cluster ID = ", cluster_item.at[0, 'clusterID'], " from non-ks-cluster set.")
                        del non_ks_clusters[intermediate_index]
        intermediate_index += 1
        
    return True


# In[18]:


def output_with_generalization(single_tuple, generalization_cluster):
    # output a single tuple with a given generalization of a cluster
    global non_ks_clusters
    global output_list_of_tuples
    
    
    output_string = ""
    # build the current attribute string
    for i_c, column in enumerate(single_tuple):
        if column != 'tuples_IDs' and column != 'clusterID':
            output_string = output_string + str(column) + ","
                
    output_string = output_string + " with G=[]"
            
    # build the generalization string
    generalization_string = ""
    for i_c, column in enumerate(generalization_cluster):
        if column != 'tuples_IDs' and column != 'clusterID':
            # get the range of the cluster
            attr_range = generalization_cluster.at[0, column][0]
            min_value = attr_range.get('min', "0")
            max_value = attr_range.get('max', "0")
            generalization_string = generalization_string + "[" + str(min_value) + "-" + str(max_value) + "]"
    output_string = output_string + generalization_string
    print("Output of Tuple ID =",str(single_tuple["index"]), ":", output_string)
    
    #official_outputting(generalization = generalization_string) # output to file
    
    # save tuple as already outputted:
    output_list_of_tuples.append(single_tuple["index"])
    
    # delete non-ks-cluster entries the tuple was inside beforehand
    intermediate_index = 0
    for cluster_item in non_ks_clusters:
        for i_c, column in enumerate(cluster_item):
            # delete the tuple ID out of this cluster
            if column == 'tuples_IDs':
                if single_tuple["index"] in cluster_item.at[0, column]:
                    # delete tuple out of this list
                    cluster_item.at[0, column].remove(single_tuple["index"])
                    # if the non-ks-cluster had only this single tuple inside then delete cluster
                    if len(cluster_item.at[0, column]) == 0:
                        # delete the cluster from the non-ks-anonymized cluster set
                        #intermediate_index = 0
                        #for cluster_item_2 in non_ks_clusters:
                        #    if cluster_item_2.at[0, 'clusterID'] == single_C_i.at[0, 'clusterID']:
                        print("Remove cluster ID = ", cluster_item.at[0, 'clusterID'], " from non-ks-cluster set.")
                        del non_ks_clusters[intermediate_index]
        intermediate_index += 1
    
    return True


# In[19]:


def output_cluster(current_non_ks_cluster, current_size_cluster, stream_of_tuples,  k, my):
    global ks_clusters # the already outputted ks-anonymized clusters
    global tau_global
    global non_ks_clusters
    global output_list_of_tuples
    
    # set of clusters returned by splitting cluster
    SC = []
    
    # let's check whether cluster can be split
    if current_size_cluster >= 2*k:
        SC = split(current_non_ks_cluster=current_non_ks_cluster, k=k)
    else:
        # cluster can not be split
        SC = [current_non_ks_cluster]
        
    for single_C_i in SC:
        # output all tuples in this cluster with its generalization
        print("we are about to output the cluster with ID ", str(single_C_i.at[0, 'clusterID']))
        
        # TODO with real outputting (when PRODUCTION ready)
        
        # output each tuple
        for single_tuple_ID in single_C_i.at[0, 'tuples_IDs']:
            current_tuple = stream_of_tuples[single_tuple_ID]
                  
            output_string = ""
            # build the current attribute string
            for i_c, column in enumerate(current_tuple):
                if column != 'tuples_IDs' and column != 'clusterID':
                    output_string = output_string + str(column) + ","
                
            output_string = output_string + " with G=[]"
             
             # build the generalization string
            generalization_string = ""
            for i_c, column in enumerate(single_C_i):
                if column != 'tuples_IDs' and column != 'clusterID':
                    # get the range of the cluster
                    attr_range = single_C_i.at[0, column][0]
                    min_value = attr_range.get('min', "0")
                    max_value = attr_range.get('max', "0")
                    generalization_string = generalization_string + "[" + str(min_value) + "-" + str(max_value) + "]"
            output_string = output_string + generalization_string
            print("Output of Tuple ID =",str(current_tuple["index"]), ":", output_string)
            
            # save outputted tuple as outputted
            output_list_of_tuples.append(single_tuple_ID)
            #print("output list of tuples:", output_list_of_tuples)
            # TODO delete tuple out of maybe existing clusters??
            
            #output_with_generalization(single_tuple=current_tuple, generalization_cluster=single_C_i)
            
            #official_outputting(generalization = generalization_string) # output to file
            
            
        
        # update tau according to InfoLoss(of this cluster)
        #calculate how many last ks-clusters can be viewed (depending on my and the number of existing ones)
        # upper-bound minus one because the last to be calculated cluster is the current one which is
        # not already in the outputted ks-clusters
        upper_bound = min(my-1, len(ks_clusters)-1)
        intermediate_result = info_loss_cluster(single_C_i)
        if upper_bound >= 0: # catch division by zero in the beginning when there's no ks-cluster yet created
            for i in range (0, upper_bound):
                # take the last my-1 (because current one also observed) clusters
                intermediate_result += info_loss_cluster(ks_clusters[len(ks_clusters)-1-i])
            # calculate average info loss
            intermediate_result = intermediate_result / (upper_bound + 1)
            # originally divided by my but since in beginning not enough ks-clusters we have to look at min(..)
        tau_global = intermediate_result
        print("param:tau_global updated to: ", tau_global, " and last generalization had info loss:", info_loss_cluster(single_C_i))
        
        if info_loss_cluster(single_C_i) <= tau_global:
            # insert this cluster as a good cluster into the set of ks-anonymized clusters
            
            # create new cluster
            #new_ks_cluster = pd.DataFrame(columns=column_names_cluster)
            #print("single_c_i:", single_C_i)
            #new_ks_cluster = single_C_i
            #print("We can store this good non-ks-anonymized cluster ID = ", single_C_i.at[0, 'clusterID'], " as ks-cluster")
            ks_clusters.append(single_C_i)
            print("Successfully created ks-anonymized cluster ID: ",str(single_C_i.at[0, 'clusterID']))
            #print("new cluster to be: ",new_ks_cluster)
            #print("ks-clusters:",ks_clusters)
           
            
            # if the info loss to large then do not save this cluster in ks-anonymized cluster set (info loss too bad)
        
        
        # delete the cluster from the non-ks-anonymized cluster set
        intermediate_index = 0
        for cluster_item in non_ks_clusters: # maybe the for loop and if are not needed if removing of cluster works
            if cluster_item.at[0, 'clusterID'] == single_C_i.at[0, 'clusterID']:
                print("We can remove the cluster ID = ", single_C_i.at[0, 'clusterID'], " from the non-ks-cluster set.")
                del non_ks_clusters[intermediate_index]
            intermediate_index += 1
                #non_ks_clusters.remove#(single_C_i) if single_C_i in non_ks_clusters else None
            
    
    return True


# As next we have the delay_constraint function which handles expiring tuples that need to be outputted:

# In[20]:


# handling of an expiring tuple which has to be outputted
def delay_constraint(expiring_tuple, stream_of_tuples, k, my):
    global non_ks_clusters
    global ks_clusters
    
    # current non-ks-cluster the tuple belongs to
    current_cluster = non_ks_clusters[0] # initialize with anything, can also be empty
    # get current cluster the tuple is in
    #cluster_index_tuple = -1
    for cluster_item in non_ks_clusters:
        for one_ID in cluster_item.at[0, 'tuples_IDs']:
            if one_ID == expiring_tuple["index"]:
                #cluster_index_tuple = cluster_item.at[0, 'clusterID']
                #save this cluster as the cluster this tuple belongs to
                current_cluster = cluster_item
    temp_list = []
    unique_tuples_list = [temp_list.append(x) for x in current_cluster.at[0, 'tuples_IDs'] if x not in temp_list]
    # count the unique items in the list
    size_non_ks_cluster = len(unique_tuples_list)
    
    if size_non_ks_cluster > 1:
        # check whether not a duplet in tuples, meaning that individual pid values captured several times
        distinct_pid_values = 0
        for one_entry in current_cluster.at[0, 'tuples_IDs']:
            #print(one_entry, type(one_entry))
            one_tuple = stream_of_tuples[int(one_entry)]
            one_tuple_unique = True
            for two_entry in current_cluster.at[0, 'tuples_IDs']:
                # check that not the same ID
                if one_entry != two_entry:
                    two_tuple = stream_of_tuples[int(two_entry)]
                    # check whether values distinct
                    if one_tuple["one"] == two_tuple["one"] and one_tuple["two"] == two_tuple["two"] and one_tuple["three"] == two_tuple["three"] and one_tuple["four"] == two_tuple["four"] and one_tuple["five"] == two_tuple["five"] and one_tuple["six"] == two_tuple["six"]:
                        # if all are equal then not distinct
                        one_tuple_unique = False
                                         
            # only if this one tuple was marked as unique then count it as a distinct pid value, another distinct entry
            if (one_tuple_unique):
                distinct_pid_values = distinct_pid_values + 1
        
        if distinct_pid_values == 0:
            distinct_pid_values = 1 # at least one pid value is in cluster if there are tuples contained
   
        # the size of a cluster is then the number of distinct pid values in the cluster
        size_non_ks_cluster = distinct_pid_values
    #end of if whether size_non_ks_cluster > 1 because the size stays the same if just one tuple in there
    
    print("size_non_ks_cluster:", size_non_ks_cluster)
    
    if size_non_ks_cluster >= k:
        # output the cluster
        
        output_cluster(current_non_ks_cluster=current_cluster, current_size_cluster=size_non_ks_cluster, stream_of_tuples=stream_of_tuples, k= k, my=my)
        
    else: 
        
        #print("... tuple not ready to be outputted yet (non-ks-cluster size not large enough) ...")
        
        # check whether expiring tuple contained in a ks-anonymized cluster
        # if so then select one of them randomly and output and remove from current non-ks-cluster
        
        # set of possible ks-clusters, initialized to be empty
        possible_ks_clusters = []
        # check for each ks-cluster
        for current_ks_cluster in ks_clusters:
            #print("current looking at ks cluster", current_ks_cluster)
            possible_fit = True
            # check for each attribute
            for i_c, column in enumerate(current_ks_cluster):
                if column != 'tuples_IDs' and column != 'clusterID':
                    # get the range of the cluster
                    attr_range = current_ks_cluster.at[0, column]
                    min_value = attr_range[0].get('min', "0")
                    max_value = attr_range[0].get('max', "0")
                    
                    # get value of current expiring tuple
                    current_value = expiring_tuple[column]
                    
                    # if within range then okay
                    # if not then ks-cluster not fitting
                    if current_value > max_value or current_value < min_value:
                        possible_fit = False
            if possible_fit == True:
                possible_ks_clusters.append(current_ks_cluster)
        
        #print("# possible ks-clusters containing tuple:", len(possible_ks_clusters))
        # check whether the set of fitting ks-anonymized clusters is not empty
        if len(possible_ks_clusters) > 0:
            # there are fitting ks-anonymized clusters for this tuple
            random_ks_cluster = random.choice(possible_ks_clusters)
            # we can output this tuple with the cluster 
            output_with_generalization(single_tuple=expiring_tuple, generalization_cluster=random_ks_cluster)
            
            return "NULL"
        
        m = 0
        # check for each non-ks-anonymized cluster
        for cluster_item in non_ks_clusters:
            # calculate size of this cluster_item
            size_other_cluster = calculate_size_of_cluster(current_cluster = cluster_item)
            
            # check if cluster size larger than current cluster
            if size_non_ks_cluster < size_other_cluster:
                m = m+1
        if m > (len(non_ks_clusters)/2):
            # suppress tuple t
            
            print("We need to suppress this current tuple..")
            suppress_tuple(single_tuple=expiring_tuple)
            #print("Output of Tuple ID =",str(expiring_tuple["index"]), ":" + "  MOST GENERAL GENERALIZATION")
            
            return "NULL"
        
        # otherwise we need to merge clusters
        #print("We need to merge some clusters..")
        # get all non-ks-anonymized clusters without current_cluster
        other_clusters = []
        for current_non_ks_cluster in non_ks_clusters:
            if current_non_ks_cluster.at[0, 'clusterID'] != current_cluster.at[0, 'clusterID']:
                other_clusters.append(current_non_ks_cluster)
        MC = merge_clusters(expiring_cluster=current_cluster, set_of_other_clusters=other_clusters, k=k)
        MC_size_cluster = calculate_size_of_cluster(current_cluster=MC)
        output_cluster(current_non_ks_cluster=MC, current_size_cluster=MC_size_cluster, stream_of_tuples=stream_of_tuples,  k=k, my=my)
    
    return "NULL"


# In[ ]:





# <h3>Main function of CASTLE</h3>
# Firstly the main CASTLE function which takes a new tuple as input and if there is a tuple which needs to be outputted then it outputs the expiring tuple as well:

# In[21]:


current_position_of_stream = 0
stream_of_tuples = []

#save the highest number of outputted tuple
output_tuple_max_index = -1

# save list of outputted tuples
output_list_of_tuples = []

# cluster indexes supposed to be increasing
number_of_cluster_indezes = 0

def CASTLE_main(new_tuple, k, delay_counts, tau_param, my, betha):
    global non_ks_clusters
    # current position of tuple
    global current_position_of_stream
    current_position_of_stream = new_tuple["index"]
    
    # set the initial value for the tau for the start, later on updated on recent my outputted ks-clusters
    global tau_global
    tau_global = tau_param
    
    
    # add new tuple to our local memory repository of stream received = input stream tuples
    stream_of_tuples.append(new_tuple)
    
    # PART I
    
    # check whether we already have clusters to put tuple into
    if not non_ks_clusters:
        create_new_cluster(new_tuple, new_cluster_index = number_of_cluster_indezes)#len(non_ks_clusters))
    else:
        # for the new tuple check whether a good cluster exists for it
        cluster_good = best_selection(new_tuple, k=k, betha=betha)
        #if cluster_good == None:
        # create new cluster
        #create_new_cluster(new_tuple, new_cluster_index = len(non_ks_clusters))
        #else:
        # add to the best fitting cluster
        # get index of this fitting cluster within non-ks-clusters
        index_cluster_good = cluster_good.at[0, 'clusterID']
        updated_cluster_good = add_tuple_to_cluster(new_tuple, cluster_good)
        # update the returned cluster
        for current_non_ks_cluster in non_ks_clusters:
            if current_non_ks_cluster.at[0, 'clusterID'] == index_cluster_good:
                current_non_ks_cluster = updated_cluster_good
        #non_ks_clusters[index_cluster_good] = updated_cluster_good
        
    # PART II
    
    # check whether there exists an expiring tuple
    target_count = int(current_position_of_stream - delay_counts)
    # only output if there are at least delay counts tuples beforehand present
    # basically handle case that in the very beginning there cannot be outputted any tuple before
    if target_count >= 0:
        #print("We test expiring tuple ID: ", target_count)
        if not check_tuple_already_outputted(target_count, expiring_tuple_ID=target_count):
            # we have to make sure that expiring tuples are outputted
            delay_constraint(stream_of_tuples[target_count], stream_of_tuples=stream_of_tuples, k=k, my=my)
        else:
            print("..this tuple has already been outputted before")


# In[ ]:





# <h2>TESTING
# Functionality of CASTLE</h2>
# <i>Please be aware of the fact that in order to test the functionality corrctly it may be needed to re-run the whole
# notebook because it can be that otherwise the non-ks-anonymized clusters persist in memory of previous insertions
# which is in reality not the case. Thanks. :) </i>

# We have some functions to visualize the output of our tests better:

# In[22]:


# IMPROVED VISUALIZATION OF OUTPUT

def printCluster(clusters_to_output):
    output_string = ""
            
    # for each cluster to be outputted
    for cluster_item in clusters_to_output:
        # build the generalization string
        output_string = output_string + "G=[] "
        for i_c, column in enumerate(cluster_item):
            if column != 'tuples_IDs' and column != 'clusterID':
                # get the range of the cluster
                attr_range = cluster_item.at[0, column][0]
                min_value = attr_range.get('min', "0")
                max_value = attr_range.get('max', "0")
                output_string = output_string + "[" + str(min_value) + "-" + str(max_value) + "]"
            if column == 'tuples_IDs':
                output_string = output_string + "tupleIDs[" + str(cluster_item.at[0, column]) + "]"
            if column == 'clusterID':
                output_string = output_string + "clusterID =[" + str(cluster_item.at[0, column]) + "]"
           
        output_string = output_string + "\n"

    return output_string


# Actual TESTS follow:

# In[23]:


def reset_environment():
    # reset the non-ks-clusters in memory
    global non_ks_clusters
    non_ks_clusters = []
    # reset the generated ks-clusters
    global ks_clusters
    ks_clusters = []
    global current_position_of_stream
    current_position_of_stream = 0
    global stream_of_tuples
    stream_of_tuples = []
    global output_tuple_max_index
    output_tuple_max_index = -1
    global output_list_of_tuples
    output_list_of_tuples = []
    global number_of_cluster_indezes
    number_of_cluster_indezes = 0
    
def reset_df(new_df):
    global df
    df = new_df


# In[24]:


# TESTING
# FUNCTIONALITY OF CASTLE

def testing():
    reset_environment()

    # "streaming" the first tuples to the CASTLE anonymization function

    for x in range(100): # do for 0..99
        # get data from sample MAC list input
        get_tuple = df.iloc[x]
        get_tuple["index"] = x
        print("+++++++++++ We give the CASTLE algorithm the next tuple (", x, ") +++++++++++++++")
        # call CASTLE main function
        CASTLE_main(get_tuple, k=2, delay_counts=5, tau_param=0.2, betha= 5000, my=5)

    # output non-ks-clusters
    print("--> --> --> --> --> --> --> CASTLE rounds are over --> --> --> --> --> --> -->")
    print("--> Output the non-ks-clusters in memory: \n", printCluster(clusters_to_output=non_ks_clusters))
    print("--> Output the ks-clusters reusable: \n", printCluster(clusters_to_output=ks_clusters))
    print("Number of non-ks-clusters (not yet outputted): " + str(len(non_ks_clusters)))
    print("Number of re-usuable ks-clusters outputted: " + str(len(ks_clusters)))


# In[25]:


testing()


# In[ ]:





# In[ ]:





# In[ ]:




