#coding:utf-8
import string
from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C



def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt

def modularity(vector_dict, edge_dict):
    Q = 0.0
    # m represents the total wight
    m = 0
    for i in edge_dict.keys():
        edge_list = edge_dict[i]
        for j in xrange(len(edge_list)):
            l = edge_list[j].strip().split(":")
            m += string.atof(l[1].strip())

    # cal community of every vector
    #find member in every community
    community_dict = {}
    for i in vector_dict.keys():
        if vector_dict[i] not in community_dict:
            community_list = []
        else:
            community_list = community_dict[vector_dict[i]]

        community_list.append(i)
        community_dict[vector_dict[i]] = community_list

    #cal inner link num and degree
    innerLink_dict = {}
    for i in community_dict.keys():
        sum_in = 0.0
        sum_tot = 0.0
        #vector num
        vector_list = community_dict[i]
        #print "vector_list : ", vector_list
        #two loop cal inner link
        if len(vector_list) == 1:
            tmp_list = edge_dict[vector_list[0]]
            tmp_dict = {}
            for link_mem in tmp_list:
                l = link_mem.strip().split(":")
                tmp_dict[l[0]] = l[1]
            if vector_list[0] in tmp_dict:
                sum_in = string.atof(tmp_dict[vector_list[0]])
            else:
                sum_in = 0.0
        else:
            for j in xrange(0,len(vector_list)):
                link_list = edge_dict[vector_list[j]]
                tmp_dict = {}
                for link_mem in link_list:
                    l = link_mem.strip().split(":")
                    #split the vector and weight
                    tmp_dict[l[0]] = l[1]
                for k in xrange(0, len(vector_list)):
                    if vector_list[k] in tmp_dict:
                        sum_in += string.atof(tmp_dict[vector_list[k]])

        #cal degree
        for vec in vector_list:
            link_list = edge_dict[vec]
            for i in link_list:
                l = i.strip().split(":")
                sum_tot += string.atof(l[1])
        Q += ((sum_in / m) - (sum_tot/m)*(sum_tot/m))
    return Q

def chage_community(vector_dict, edge_dict, Q):
    vector_tmp_dict = {}
    for key in vector_dict:
        vector_tmp_dict[key] = vector_dict[key]

    #for every vector chose it's neighbor
    for key in vector_tmp_dict.keys():
        neighbor_vector_list = edge_dict[key]
        for vec in neighbor_vector_list:
            ori_com = vector_tmp_dict[key]
            vec_v = vec.strip().split(":")

            #compare the list_member with ori_com
            if ori_com != vector_tmp_dict[vec_v[0]]:
                vector_tmp_dict[key] = vector_tmp_dict[vec_v[0]]
                Q_new = modularity(vector_tmp_dict, edge_dict)
                #print Q_new
                if (Q_new - Q) > 0:
                    Q = Q_new
                else:
                    vector_tmp_dict[key] = ori_com
    return vector_tmp_dict, Q

def modify_community(vector_dict):
    #modify the community
    community_dict = {}
    community_num = 0
    for community_values in vector_dict.values():
        if community_values not in community_dict:
            community_dict[community_values] = community_num
            community_num += 1
    for key in vector_dict.keys():
        vector_dict[key] = community_dict[vector_dict[key]]
    return community_num

def data_proprecess(graph):
    vector_dict = {}
    edge_dict = {}
    dta = graph.A
    P = np.nonzero(dta)
    adj_arr = []
    for s in zip(P[0],P[1]):
        adj_arr.append(s)
    #print(adj_arr)
    for line in adj_arr:
        for i in xrange(1):
            edge_list = []
            if str(line[i]) not in vector_dict:
                vector_dict[str(line[i])] = str(line[i])
                edge_list.append(str(line[1-i])+":"+"1")
                edge_dict[str(line[i])] = edge_list
            else:
                edge_list = edge_dict[str(line[i])]
                edge_list.append(str(line[1 - i]) + ":" + "1")
                edge_dict[str(line[i])] = edge_list
    return vector_dict,edge_dict




def louvain_preprocess(graph):
    vector_dict,edge_dict = data_proprecess(graph)

    # 1. initilization:put every vector into different communities
    #   the easiest way:use the vector num as the community num
    for i in vector_dict.keys():
        vector_dict[i] = i

    Q = modularity(vector_dict, edge_dict)
    # print "Q_orginal = ",Q   
    print "original community_num :", len(vector_dict.keys())  
    print "%"*60
    # 2. for every vector, chose the community
    Q_new = 0.0
    while (Q_new != Q):
        Q_new = Q
        vector_dict, Q = chage_community(vector_dict, edge_dict, Q)
        print "change Q ：" , Q
    community_num = modify_community(vector_dict)


    return vector_dict,edge_dict,community_num




def louvain(max_node_wgt, graph):

    node_num = graph.node_num 
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap  # 
    norm_adj_wgt = normalized_adj_wgt(graph)
    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num  #

    vector_dict,edge_dict,community_num = louvain_preprocess(graph) # the first of step

    vector_dict, edge_new_dict, community_dict = rebuild_graph(vector_dict, edge_dict, community_num) # the next of step
    g = []
    for key in community_dict.keys():
        if len(community_dict[key])>1:
            for i in community_dict[key]:
                matched[eval(i)] = True
                g.append(eval(i))
            groups.append(g)
            g = []


    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]  
    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue 
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j] 
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node.
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])

    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1

    return (groups, coarse_graph_size)


def rebuild_graph(vector_dict, edge_dict, community_num):
    vector_new_dict = {}
    edge_new_dict = {}
    # cal the inner connection in every community
    community_dict = {}  
    for key in vector_dict.keys():
        if vector_dict[key] not in community_dict:
            community_list = []
        else:
            community_list = community_dict[vector_dict[key]]

        community_list.append(key)
        community_dict[vector_dict[key]] = community_list

    # cal vector_new_dict
    for key in community_dict.keys():
        vector_new_dict[str(key)] = str(key) 


    # put the community_list into vector_new_dict

    #cal inner link num
    innerLink_dict = {}
    for i in community_dict.keys(): # 0
        sum_in = 0.0
        #vector num
        vector_list = community_dict[i] 
        #two loop cal inner link
        if len(vector_list) == 1:
            sum_in = 0.0
        else:
            for j in xrange(0,len(vector_list)):
                link_list = edge_dict[vector_list[j]] 
                tmp_dict = {}
                for link_mem in link_list:
                    l = link_mem.strip().split(":")
                    #split the vector and weight
                    tmp_dict[l[0]] = l[1]
                for k in xrange(0, len(vector_list)):
                    if vector_list[k] in tmp_dict:  
                        sum_in += string.atof(tmp_dict[vector_list[k]])

        inner_list = []
        inner_list.append(str(i) + ":" + str(sum_in))
        edge_new_dict[str(i)] = inner_list  

    #cal outer link num
    community_list = community_dict.keys()
    for i in xrange(len(community_list)): 
        for j in xrange(len(community_list)):
            if i != j:
                sum_outer = 0.0
                member_list_1 = community_dict[community_list[i]] 
                member_list_2 = community_dict[community_list[j]] 

                for i_1 in xrange(len(member_list_1)):
                    tmp_dict = {}
                    tmp_list = edge_dict[member_list_1[i_1]] 

                    for k in xrange(len(tmp_list)): 
                        tmp = tmp_list[k].strip().split(":");
                        tmp_dict[tmp[0]] = tmp[1]
                    for j_1 in xrange(len(member_list_2)):
                        if member_list_2[j_1] in tmp_dict:
                            sum_outer += string.atof(tmp_dict[member_list_2[j_1]])

                if sum_outer != 0:
                    inner_list = edge_new_dict[str(community_list[i])]
                    inner_list.append(str(j) + ":" + str(sum_outer))
                    edge_new_dict[str(community_list[i])] = inner_list
    return vector_new_dict, edge_new_dict, community_dict



def create_coarse_graph(graph, groups, coarse_graph_size):
    '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree

    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    for idx in range(len(groups)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop.
                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges

    coarse_graph.edge_num = nedges


    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)   
    return coarse_graph
