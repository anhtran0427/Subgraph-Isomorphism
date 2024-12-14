#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt


# In[2]:


adj_data=pd.read_csv(r'facebook_combined_adj.tsv', sep='\t',header=None)
inc_data=pd.read_csv(r'facebook_combined_inc.tsv',sep='\t',header=None)


# In[3]:


# Create inc matrix and adj matrix
nb_ver=adj_data.max().max()
nb_edge=inc_data[1].max()
adj_ma=np.zeros((nb_ver,nb_ver),dtype=np.int8)
inc_ma=np.zeros((nb_ver,nb_edge),dtype=np.int8)

for x in range (adj_data.shape[0]):  
    adj_ma[adj_data.iat[x,0]-1][adj_data.iat[x,1]-1]=1
for x in range (inc_data.shape[0]):  
    inc_ma[inc_data.iat[x,0]-1][inc_data.iat[x,1]-1]=1


# In[4]:


#Triangle counting
print('Number of triangle:')
csr_inc=csr_matrix(inc_ma)
tri_ma=adj_ma@csr_inc
print(np.count_nonzero(tri_ma==2)/3)


# In[5]:


#Ktruss decompose
k = int(input("K truss with value k: "))
S=np.count_nonzero(tri_ma==2,axis=0)
x_dismiss=np.asarray(np.nonzero(S<k-2))

inc_ma_work=inc_ma.copy()
adj_ma_work=adj_ma.copy()
while x_dismiss.size!=0:
    for x in range (x_dismiss.shape[1]):
        #print(np.transpose(inc_ma1)[x_dismiss[0][x]])
        position=np.asarray(np.transpose(inc_ma_work)[x_dismiss[0][x]].nonzero())[0]
        adj_ma_work[position[0]] [position[1]]=0
        adj_ma_work[position[1]] [position[0]]=0
    inc_ma_work=np.delete(inc_ma_work,x_dismiss,1)
    if inc_ma_work.size==0:
        print('The subgraph cannot be found')
        break
    csr_inc_work=csr_matrix(inc_ma_work)
    tri_ma=adj_ma_work@csr_inc_work
    S=np.count_nonzero(tri_ma==2,axis=0)
    x_dismiss=np.asarray(np.nonzero(S<k-2))

if inc_ma_work.size!=0:
    tit="Maximal k-truss subgraph has "+str(inc_ma_work.shape[1])+" edges"
    print (tit)


# In[6]:


save_ques= input("Do you want to save a pdf file of the graph (y/n): ")
adj_ma_visualize=adj_ma_work+adj_ma
G=nx.from_numpy_array(adj_ma_visualize)
np.random.seed(123)
color_map_edge=[]
alpha_map=[]
for node1,node2,data in G.edges.data():
    if data['weight']==2:
        color_map_edge.append('pink')
        alpha_map.append(1)
    else:
        color_map_edge.append('lightgrey')
        alpha_map.append(0.05)
plt.figure(figsize=(20,14))
nx.draw(G,with_labels=False, node_size=10,edge_color=color_map_edge,node_color='black',alpha=alpha_map)
file_name="input_graph"+str(nb_ver)+str(k)+".pdf"
plt.title("Input graph with the found maximal k-truss in pink") 
if save_ques=='y':
    plt.savefig(file_name, dpi=600, bbox_inches='tight')
plt.show()   


# In[7]:


if inc_ma_work.size!=0:
    number_of_vertex=np.count_nonzero(adj_ma_work,axis=0)
    vertex_dismiss=np.asarray(np.nonzero(number_of_vertex==0))
    adj_ma_work=np.delete(adj_ma_work,vertex_dismiss,1)
    adj_ma_work=np.delete(adj_ma_work,vertex_dismiss,0)
    G=nx.from_numpy_array(adj_ma_work)
    np.random.seed(123)
    plt.figure(figsize=(20,14))
    nx.draw(G,with_labels=False, node_size=10,edge_color='pink',node_color='black')
    file_name="ktruss_subgraph"+str(nb_ver)+str(k)+".pdf"
    ktitle="Maximal k-truss subgraph with "+str(adj_ma_work.shape[0])+" vertices and "+str(inc_ma_work.shape[1])+" edges"
    plt.title(ktitle) 
    if save_ques=='y':
        plt.savefig(file_name, dpi=600, bbox_inches='tight')
    plt.show()   


# In[ ]:




