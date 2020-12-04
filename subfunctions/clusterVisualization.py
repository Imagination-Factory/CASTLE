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