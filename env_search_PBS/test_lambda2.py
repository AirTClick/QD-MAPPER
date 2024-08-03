# This is a simple example of how to calculate the lambda2 of map using networkx library
import networkx as nx
import numpy as np
import yaml

from utils import (eecbs_obj_types, MIN_SCORE,
                              eecbs_env_number2str,
                              eecbs_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)

map_folder = "maze_maps/"
map_name = "maze512-1-0.map"
map_path = map_folder + map_name



# # load yaml map
# if map_path.split('.')[-1] == "yaml":
#     with open(map_path) as f:
#         map_data = yaml.load(f, Loader=yaml.FullLoader)
#     dim_x = map_data["dimensions"][0]
#     dim_y = map_data["dimensions"][1]
#     obs = map_data["obstacles"]
# # load MAPF benchmark map
# elif map_path.split('.')[-1] == "map":
#     with open(map_path) as f:
#         if f.readline() != "type octile\n":
#             print("Bad mapfile!")
#             exit()
#         else:
#             dim_y = int(f.readline().split(" ")[-1])
#             dim_x = int(f.readline().split(" ")[-1])
#             f.readline()
#             x = y = 0
#             obs = []
#             for line in f.readlines():
#                 for l in line:
#                     if l != ".":
#                         obs.append([x, y])
#                     x += 1
#                 x = 0
#                 y += 1
# else:
#     print("Only support .yaml and .map format!")
#     exit()

# print("Down here")

# # ceate graph
# G = nx.Graph()
# movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# # add nodes
# for i in range(dim_x):
#     for j in range(dim_y):
#         if [i, j] not in obs:
#             G.add_node((i, j))
#         print(f"[{i},{j}] node established!")

# # add edges
# for i in range(dim_x):
#     for j in range(dim_y):
#         if [i, j] not in obs:
#             # check neighbors
#             for mi, mj in movements:
#                 if -1 < mi + i < dim_x and -1 < mj + j < dim_y and [mi + i, mj + j] not in obs:
#                     G.add_edge((i, j), (mi + i, mj + j))
#             print(f"[{i},{j}] edge established!")

# # calculate eigen values
# eigen = nx.normalized_laplacian_spectrum(G)
# print('map: %s' % map_name)
# print('lambda2 is: %f' % eigen[1])


repaired_env = [
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
            ".............@........@........@",
            ".............@........@........@",
            ".@@@@@@@@@@..@@@@..@..@..@..@@@@",
            ".......@........@..@..@..@.....@",
            ".......@........@..@..@..@.....@", 
            "@@@@@@@@..@@@@..@..@..@..@@@@@@@", 
            ".@..@.....@..@..@..@..@..@.....@", 
            ".@..@.....@..@..@..@..@..@.....@", 
            ".@..@..@..@..@..@..@..@..@..@..@", 
            "....@..@..@........@..@.....@..@", 
            "....@..@..@........@..@.....@..@", 
            "@@..@..@@@@@@@@@@..@..@@@@..@..@", 
            "....@...........@..@.....@..@..@", 
            "....@...........@..@.....@..@..@", 
            ".@@@@..@@@@@@@..@@@@@@@..@..@..@",
            ".......@.....@.....@..@..@..@..@",
            ".......@.....@.....@..@..@..@..@", 
            ".@@@@..@@@@..@@@@..@..@..@..@..@", 
            ".@..@.....@........@.....@..@..@", 
            ".@..@.....@........@.....@..@..@", 
            ".@..@..@@@@@@@..@@@@@@@..@@@@..@", 
            "....@........@..@.....@........@", 
            "....@........@..@.....@........@", 
            "@@..@@@@@@@..@..@..@@@@@@@@@@..@", 
            ".@.....@..@..@..............@..@", 
            ".@.....@..@..@..............@..@", 
            ".@@@@..@..@..@@@@@@@@@@..@@@@..@", 
            "....@..@..@........@..@..@.....@",
            "....@..@..@........@..@..@.....@",
            ".@@@@..@..@@@@@@@..@..@@@@@@@..@", 
            "................@.....@........@"
        ]

repaired_env = eecbs_env_str2number(repaired_env)
print(repaired_env)

def cal_lambda2(map_np_repaired):
    dim_x, dim_y = map_np_repaired.shape

    G = nx.Graph()
    movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for i in range(dim_x):
        for j in range(dim_y):
            if map_np_repaired[i][j] != 1:
                G.add_node((i, j))
    
    for i in range(dim_x):
        for j in range(dim_y):
            if map_np_repaired[i][j] != 1:
                # check neighbors
                for mi, mj in movements:
                    if -1 < mi + i < dim_x and -1 < mj + j < dim_y and map_np_repaired[mi + i, mj + j] != 1:
                        G.add_edge((i, j), (mi + i, mj + j))


    eigen = nx.normalized_laplacian_spectrum(G)
    # print('lambda2 is: %f' % eigen[1])

    return (eigen[1])


# draw_graph(repaired_env)
print (networkx.__version__)