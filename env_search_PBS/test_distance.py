import numpy as np
import scipy
import random
from utils import (eecbs_obj_types, MIN_SCORE,
                              eecbs_env_number2str,
                              eecbs_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)

import json
import networkx as nx
import heapq
import seaborn as sns
import matplotlib.pyplot as plt

# Open the JSON file
with open('../logs/PBS_python_start_goal/generated_maps/40_49.json', 'r') as file:
    data = json.load(file)

start = data['start']
goal = data['goal']
layout = data['layout']

G = eecbs_env_str2number(layout)
n_row, n_col = G.shape
print(G)

start_locations = start[0]
start_index = (start_locations // n_col, start_locations % n_col)
goal_locations = goal[0]
target_index = (goal_locations // n_col, goal_locations % n_col)

def dijkstra(matrix, start, target):
    rows, cols = matrix.shape
    distances = np.full((rows, cols), np.inf)
    distances[start] = 0
    queue = [(0, start)]
    visited = set()
    predecessors = {}

    while queue:
        curr_dist, (i, j) = heapq.heappop(queue)
        visited.add((i, j))
        if (i, j) == target:
            path = []
            curr_node = target
            while curr_node:
                path.append(curr_node)
                curr_node = predecessors.get(curr_node)
            return distances[i, j], path[::-1]
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and matrix[ni, nj] != 1 and (ni, nj) not in visited:
                new_dist = curr_dist + 1
                if new_dist < distances[ni, nj]:
                    distances[ni, nj] = new_dist
                    predecessors[(ni, nj)] = (i, j)
                    heapq.heappush(queue, (new_dist, (ni, nj)))
    return np.inf, None

tile_usage = np.zeros((n_row, n_col))

for i in range(len(start)):
    start_locations = start[i]
    start_index = (start_locations // n_col, start_locations % n_col)
    goal_locations = goal[i]
    target_index = (goal_locations // n_col, goal_locations % n_col)
    shortest_path_length,passing_indices = dijkstra(G, start_index, target_index)

    for j in range(len(passing_indices)):
        index_0 = passing_indices[j][0]
        index_1 = passing_indices[j][1]
        tile_usage[index_0][index_1] += 1

print(np.max(tile_usage))

# shortest_path_length,passing_indices = dijkstra(G, start_index, target_index)
# print(f"Shortest path length from {start_index} to {target_index}: {shortest_path_length}")
# print(f"Passing indices: {passing_indices}")
print(tile_usage)

sns.heatmap(
        tile_usage,
        square=True,
        cmap="Reds",
        # ax=ax_tile_use,
        # cbar_ax=ax_tile_use_cbar,
        cbar=True,
        rasterized=False,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=80,
        cbar_kws={
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )
plt.show()


# Show plot
plt.savefig('heatmap.png')

# breakpoint()

# def cal_connectivity(repaired_env):
#     n_row, n_col = repaired_env.shape
#     total_grid = n_row * n_col
#     num_of_obstacle = 0
#     total_grids = n_row * n_col
#     distance_matrix = np.zeros((total_grid,total_grid), dtype=int) 
#     connectivity_graph = np.zeros(repaired_env.shape, dtype=int)

#     #build the distance matrix for graphs
#     for i in range(n_row):
#         for j in range(n_col):
#             if repaired_env[i,j] == eecbs_obj_types.index("@"):
#                 num_of_obstacle += 1
#             else: 
#                 if i == 0 and j == 0:
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
#                 elif i == n_row-1 and j == 0:
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
#                 elif i == 0 and j == n_col - 1:
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                 elif i == n_row-1 and j == n_col - 1:
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                 #upper and down edge 
#                 elif i == 0:
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
#                 elif i == n_row -1:
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
#                 #left and right edge  
#                 elif j == 0:
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1
#                 elif j == n_col -1:
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                 #normal cases
#                 else:
#                     if repaired_env[i-1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i-1)*n_col+j] = 1
#                     if repaired_env[i+1,j] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,(i+1)*n_col+j] = 1
#                     if repaired_env[i,j-1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j-1] = 1
#                     if repaired_env[i,j+1] != eecbs_obj_types.index("@"): distance_matrix[i*n_col+j,i*n_col+j+1] = 1

#     distance_graph = scipy.sparse.csr_matrix(distance_matrix)
#     # print("distance_graph: " ,distance_graph)

#     #The matrix of distances between graph nodes. dist_matrix[i,j] gives the shortest distance from point i to point j along the graph.
#     dist_matrix, predecessors= scipy.sparse.csgraph.floyd_warshall(csgraph=distance_graph, directed=False, return_predecessors=True)
#     # for i in range(0, dist_matrix.shape[0]):
#     #     print(dist_matrix[i])
#     # print("dist_matrix: ",dist_matrix.shape)

#     for i in range(predecessors.shape[0]):
#         for j in range(predecessors.shape[1]):
#             if predecessors[i,j] != -9999:
#                 # print("Processor: ", predecessors[i,j])
#                 result_row = predecessors[i,j] // n_col
#                 # print("Row: ", result_row)
#                 result_col = predecessors[i,j] % n_col
#                 # print("Col: ", result_col)
#                 connectivity_graph[result_row,result_col] += 1

#     normalized_graph = connectivity_graph/(np.sum(connectivity_graph))

#     # print("Conectivity graph: ", connectivity_graph)
#     # print("Normalized graph: ", normalized_graph)
    
#     # print("Sum: ", np.sum(connectivity_graph))
#     # print("Mean: ", np.mean(connectivity_graph))
#     connectivity_graph_nozero = normalized_graph[normalized_graph != 0]
#     # sqr_diff= np.sum(np.square(normalized_graph-1))

#     # total_obstacle_diff = 1 * num_of_obstacle

#     # sq_sum_diff = sqr_diff - total_obstacle_diff

#     # connectivity_std = np.sqrt(sq_sum_diff / (total_grids - num_of_obstacle))

#     connectivity_std = np.std(connectivity_graph_nozero)
#     before_norm = connectivity_std * (np.sum(connectivity_graph))

#     # print("Connectivity_Std: ",connectivity_std)
#     # print("Connectivity_Std before norm: ",before_norm)

#     return dist_matrix

# repaired_env = [
#         "@@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@...@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@...@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@...@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@...@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@...@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@...@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@...@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.@.@.@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@.@.....@.@.@",
#         ".@.@.@.@.@.@.@.@.@.@...@.@.@.@.@",
#         "...@.@.@...@.@.......@.@.@.@.@.@",
#         "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        
#     ]

# repaired_env = eecbs_env_str2number(repaired_env)
# dist_matrix = cal_connectivity(repaired_env)
# print(dist_matrix)

# # select points with length greater than half the longest path
# n_row, n_col = repaired_env.shape
# total_grid = n_row * n_col

# dist_matrix[np.isinf(dist_matrix)] = 0
# less_restrict_indices = np.where(dist_matrix > (np.max(dist_matrix)/8)) 
# indices = np.where(dist_matrix > (np.max(dist_matrix)/2))

# print(less_restrict_indices[0])
# print(less_restrict_indices[1])


# print(indices[0][0])
# print(indices[1][0])

# num_of_agents = 150

# def split_list(lst, parts=5):
#     length = len(lst)
#     part_size = length // parts
#     remainder = length % parts
#     result = []
#     start = 0
#     for i in range(parts):
#         end = start + part_size + (1 if i < remainder else 0)
#         result.append(lst[start:end])
#         start = end
#     return result

# parts = 5
# result_start = split_list(indices[0])
# print(result_start)
# print("len:", [(result_start[i].shape) for i in range(0,5)])
# print([result_start[i][0] for i in range(0,5)])

# result_goal = split_list(indices[1])
# print(result_goal)
# print("len:", [(result_goal[i].shape) for i in range(0,5)])
# print([result_goal[i][0] for i in range(0,5)]) 


# start_locations = []
# goal_locations = []
# test_indices = []

# print("start_len",len(result_start))
# print("goal_len",len(result_goal))
# i = 0

# interval = np.arange(num_of_agents//parts,num_of_agents+num_of_agents//parts,num_of_agents//parts)
# print(interval)

# k = 0
# while i < num_of_agents:
#     if i <= interval[k]:
#         temp_start = result_start[k]
#         temp_goal = result_goal[k]
#     else:
#         k += 1
#         temp_start = result_start[k]
#         temp_goal = result_goal[k]
#     print(k)
#     random_index = np.random.randint(0,len(temp_start)-1,size = 1)
#     index_0 = temp_start[random_index]
#     index_1 = temp_goal[random_index]
#     if (index_0 not in start_locations) and (index_1 not in goal_locations):
#         start_locations.append(int(index_0))
#         goal_locations.append(int(index_1))
#         i += 1
        



# print(len(start_locations))
# print(start_locations)

# print(len(goal_locations))
# print(goal_locations)

# # for i in range(0,150):
# #     print(dist_matrix[start_locations[i]][goal_locations[i]])


# def has_duplicates(lst):
#     seen = set()
#     for num in lst:
#         if num in seen:
#             return True
#         seen.add(num)
#     return False

# if has_duplicates(start_locations):
#     print("List has duplicates")
# else:
#     print("List does not have duplicates")

# if has_duplicates(goal_locations):
#     print("List has duplicates")
# else:
#     print("List does not have duplicates")
