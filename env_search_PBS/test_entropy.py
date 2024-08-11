# from warehouseEECBS.module import calc_layout_entropy
from itertools import repeat, product
from utils import (eecbs_obj_types, MIN_SCORE,
                              eecbs_env_number2str,eecbs_env_str2number,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)
from scipy.stats import entropy
import json
import os
import numpy as np
import os
import matplotlib.pyplot as plt


#Convert All available single agent 512 * 512 maps (.map file) to .json file
convert_on = False



def turnMaptoJson(file):
    with open(file, 'r') as map_file:
        # Read the contents of the .map file
        map_data = map_file.read()

    data = dict()
    data["layout"] = []
    start = 37

    for i in range(37,len(map_data)):
        # print(i)
        if map_data[i] == '\n':
            row_data = map_data[start:i].replace("T","@")
            data["layout"].append(row_data)
            start = i+1
    
    index_new = file.index("/")
    file_new = file[index_new+1:]
    print(file_new)

    file_path = "MAPF_JSON/"+file_new+'.json'

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if convert_on:
    dir_maze = 'maze_maps'
    dir_random = "random_maps"
    dir_room = "room_maps"

    dir_MAPF = "MAPF_MAPS"

    map_filenames = []

    for filename in os.listdir(dir_MAPF):
        if os.path.isfile(os.path.join(dir_MAPF,filename)):
            map_filenames.append(os.path.join(dir_MAPF,filename))

    # for filename in os.listdir(dir_maze):
    #     if os.path.isfile(os.path.join(dir_maze,filename)):
    #         map_filenames.append(os.path.join(dir_maze,filename))

    # for filename in os.listdir(dir_random):
    #     if os.path.isfile(os.path.join(dir_random,filename)):
    #         map_filenames.append(os.path.join(dir_random,filename))

    # for filename in os.listdir(dir_room):
    #     if os.path.isfile(os.path.join(dir_room,filename)):
    #         map_filenames.append(os.path.join(dir_room,filename))
    
    for filename in map_filenames:
        turnMaptoJson(filename)




# compute the distribution of tile among all maps

def kl_divergence(p, q,epsilon=1e-12):
    p_smooth = p + epsilon
    q_smooth = q + epsilon
    return np.sum(p_smooth * np.log(p_smooth / q_smooth))


def calc_layout_entropy_test(map_layout):
    """
    Calculate entropy of the storage area of the layout.

    We first formulate the layout as a tile pattern distribution by following
    Lucas, Simon M. M. and Vanessa Volz. “Tile pattern KL-divergence for
    analysing and evolving game levels.” Proceedings of the Genetic and
    Evolutionary Computation Conference (2019).

    Then we calculate the entropy.
    """
    # Separate the storage area
    # h, w = map_np_repaired.shape
    map_np_repaired = eecbs_env_str2number(map_layout)
    # print("map_np_repaired: ",map_np_repaired)


    storage_area = map_np_repaired
    # if w_mode:
    #     storage_area = map_np_repaired[:,
    #         KIVA_WORKSTATION_BLOCK_WIDTH:w-KIVA_WORKSTATION_BLOCK_WIDTH]
    # else:
    #     storage_area = map_np_repaired[
    #         KIVA_ROBOT_BLOCK_WIDTH:h-KIVA_ROBOT_BLOCK_WIDTH,
    #         KIVA_ROBOT_BLOCK_WIDTH:w-KIVA_ROBOT_BLOCK_WIDTH]

    # Generate list of patterns (we use 2 x 2)
    storage_obj_types = eecbs_obj_types[:2]
    tile_patterns = {
        # for 2 * 2 window
        # "".join(x): 0 for x in product(storage_obj_types, repeat=4) 
        "".join(x): 0 for x in product(storage_obj_types, repeat=9)
    }
    # print("tile_patterns",tile_patterns)

    s_h, s_w = storage_area.shape
    # Iterate over 2x2 blocks
    for i in range(s_h - 2):
        for j in range(s_w - 2):
            # take the constant as a parameter to test different window sizes
            curr_block = storage_area[i:i+3,j:j+3]
            curr_pattern = "".join(eecbs_env_number2str(curr_block))
            tile_patterns[curr_pattern] += 1
    pattern_dist = list(tile_patterns.values())
    # print("tile_patterns",tile_patterns)
    # print(pattern_dist)

    np_pattern_dist = np.array(pattern_dist,dtype = int)

    sum_pattern = np.sum(np_pattern_dist)
    norm_pattern_dist = np_pattern_dist/sum_pattern

    epsilon = 1e-12
    norm_pattern_dist += epsilon 
    # sum_pattern = np.sum(pattern_dist)
    # norm_pattern_dist = pattern_dist/sum_pattern
    
    # epsilon=1e-12
    # norm_pattern_dist += epsilon

    # Use number of patterns as the base to bound the entropy to [0, 1]
    # return entropy(pattern_dist, base=len(pattern_dist))
    # print(entropy(pattern_dist, base=len(pattern_dist)))

    return norm_pattern_dist

compute_signal = False

if compute_signal:
    map_filenames = []
    dir = "room_maps_json"

    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,filename)):
            map_filenames.append(os.path.join(dir,filename))


    def sum_dist_all_maps(file):
        with open(file, 'r') as json_file:
            # Read the contents of the .map file
            data = json.load(json_file)
        
        map_layout = data['layout']

        distributon = calc_layout_entropy_test(map_layout)

        return distributon

    distribution = np.zeros(512)
    print(distribution)

    for file in map_filenames:
        print(file)
        distribution += sum_dist_all_maps(file) 

    epsilon=1e-12

    sum_pattern = np.sum(distribution)
    norm_pattern_dist = distribution/sum_pattern
    norm_pattern_dist += epsilon

    print(norm_pattern_dist)

    norm_pattern_dist_list = norm_pattern_dist.tolist() 
    print("what i need:", norm_pattern_dist_list)
    print(len(norm_pattern_dist_list))

with open("maze512-1-0.map.json", "r") as f:
    data = json.load(f)

map_layout = data['layout']

epsilon=1e-12

# distribution = calc_layout_entropy_test(map_layout)
# sum_pattern = np.sum(distribution)
# norm_pattern_dist = distribution/sum_pattern
# norm_pattern_dist += epsilon
# print(norm_pattern_dist.tolist() )


# map_layout = data['layout']

# # map_layout_2 = data_2['layout']
# # map_layout_random = data_random['layout']
# # map_layout_random_2 = data_random_2['layout']
# # # print(map_layout)

# dist_room = calc_layout_entropy_test(map_layout)
# # print(dist_1)

# with open("maze_maps_json/maze512-2-0.map.json", "r") as f:
#     data2 = json.load(f)
# map_layout2 = data2['layout']
# dist_maze = calc_layout_entropy_test(map_layout2)
# # print(dis_maze)
# # print("Diff in tile_distribution: ", dist_maze - dist_room)
# # dist_maze = calc_layout_entropy_test(map_layout_2)
# # # print(dist_2)
# # dist_random = calc_layout_entropy_test(map_layout_random) 
# # # dist_ref = dist_1
# # dist_random_2 = calc_layout_entropy_test(map_layout_random_2) 

# epsilon=1e-12

# #room
# sum_pattern_room = np.sum(dist_room)
# norm_pattern_dist_room = dist_room/sum_pattern_room
# norm_pattern_dist_room += epsilon

# #maze
# sum_pattern_maze = np.sum(dist_maze)
# norm_pattern_dist_maze = dist_maze/sum_pattern_maze
# norm_pattern_dist_maze += epsilon
# # print("Diff after norm: ", norm_pattern_dist_maze - norm_pattern_dist_room)
# # #random_1
# # sum_pattern_random = np.sum(dist_random)
# # norm_pattern_dist_random = dist_random/sum_pattern_random
# # norm_pattern_dist_random += epsilon

# # #random_2
# # sum_pattern_random_2= np.sum(dist_random_2)
# # norm_pattern_dist_random_2 = dist_random_2/sum_pattern_random_2
# # norm_pattern_dist_random_2 += epsilon

# # # print(norm_pattern_dist_maze)

# # #kl_div_room
# kl_room_maze = entropy(norm_pattern_dist_room,norm_pattern_dist_maze , base = 512)
# print(kl_room_maze)
# kl_room_random = entropy(norm_pattern_dist_room,norm_pattern_dist_random , base = 512)
# kl_room_random_2= entropy(norm_pattern_dist_room,norm_pattern_dist_random_2, base = 512)

# #kl_div_maze
# kl_maze_room = entropy(norm_pattern_dist_maze,norm_pattern_dist_room , base = 512) 
# kl_maze_random = entropy(norm_pattern_dist_maze,norm_pattern_dist_random , base = 512)
# kl_maze_random_2 = entropy(norm_pattern_dist_maze,norm_pattern_dist_random_2 , base = 512)

# #kl_div_random
# kl_random_room = entropy(norm_pattern_dist_random,norm_pattern_dist_room , base = 512)
# kl_random_maze= entropy(norm_pattern_dist_random,norm_pattern_dist_maze , base = 512)
# kl_random_random_2 = entropy(norm_pattern_dist_random,norm_pattern_dist_random_2 , base = 512)

# #kl_div_random_2
# kl_random_2_room = entropy(norm_pattern_dist_random_2,norm_pattern_dist_room , base = 512)
# kl_random_2_maze= entropy(norm_pattern_dist_random_2,norm_pattern_dist_maze , base = 512)
# kl_random_2_random = entropy(norm_pattern_dist_random_2,norm_pattern_dist_random , base = 512)


# print("kl_room_maze:",kl_room_maze)
# print("kl_room_random:",kl_room_random)
# print("kl_room_random_2:",kl_room_random_2)
# print("--------------------------------------------")
# print("kl_maze_room:", kl_maze_room)
# print("kl_maze_random:",kl_maze_random)
# print("kl_maze_random_2:",kl_maze_random_2)
# print("--------------------------------------------")
# print("kl_random_room:",kl_random_room)
# print("kl_random_maze:",kl_random_maze)
# print("kl_random_random_2:",kl_random_random_2)
# print("--------------------------------------------")
# print("kl_random_2_room:",kl_random_2_room)
# print("kl_random_2_maze:",kl_random_2_maze)
# print("kl_random_2_random:",kl_random_2_random)


def calc_layout_entropy(map_np_repaired):
    """
    Calculate entropy of the storage area of the layout.

    We first formulate the layout as a tile pattern distribution by following
    Lucas, Simon M. M. and Vanessa Volz. “Tile pattern KL-divergence for
    analysing and evolving game levels.” Proceedings of the Genetic and
    Evolutionary Computation Conference (2019).

    Then we calculate the entropy.
    """
    # Separate the storage area
    h, w = map_np_repaired.shape
    storage_area = map_np_repaired
    # if w_mode:
    #     storage_area = map_np_repaired[:,
    #         KIVA_WORKSTATION_BLOCK_WIDTH:w-KIVA_WORKSTATION_BLOCK_WIDTH]
    # else:
    #     storage_area = map_np_repaired[
    #         KIVA_ROBOT_BLOCK_WIDTH:h-KIVA_ROBOT_BLOCK_WIDTH,
    #         KIVA_ROBOT_BLOCK_WIDTH:w-KIVA_ROBOT_BLOCK_WIDTH]

    # Generate list of patterns (we use 2 x 2)
    storage_obj_types = eecbs_obj_types[:2]
    tile_patterns = {
        # for 2 * 2 window
        # "".join(x): 0 for x in product(storage_obj_types, repeat=4) 
        "".join(x): 0 for x in product(storage_obj_types, repeat=9)
    }

    s_h, s_w = storage_area.shape
    # Iterate over 2x2 blocks
    for i in range(s_h - 2):
        for j in range(s_w - 2):
            # take the constant as a parameter to test different window sizes
            curr_block = storage_area[i:i+3,j:j+3]
            curr_pattern = "".join(eecbs_env_number2str(curr_block))
            tile_patterns[curr_pattern] += 1
    pattern_dist = list(tile_patterns.values())

    np_pattern_dist = np.array(pattern_dist,dtype = int)

    sum_pattern = np.sum(np_pattern_dist)
    norm_pattern_dist = np_pattern_dist/sum_pattern

    epsilon = 1e-12
    norm_pattern_dist += epsilon

    # Use number of patterns as the base to bound the entropy to [0, 1]
    return norm_pattern_dist,entropy(pattern_dist, base=len(pattern_dist))

map_filenames = []
obstacles = []
KL_div = []
dir = "MAPF_JSON"

for filename in os.listdir(dir):
    if os.path.isfile(os.path.join(dir,filename)):
        map_filenames.append(os.path.join(dir,filename))

# print(map_filenames)
maze_ref_dist = [0.20222222222322223, 0.018888888889888888, 0.004444444445444445, 0.014444444445444444, 0.011111111112111112, 1e-12, 0.011111111112111112, 0.09722222222322223, 0.004444444445444445, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.004444444445444445, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.08111111111211111, 0.004444444445444445, 0.001111111112111111, 1e-12, 0.010555555556555556, 1e-12, 1e-12, 1e-12, 0.009444444445444445, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 0.05388888888988889, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.014444444445444444, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011111111112111112, 0.0033333333343333335, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0016666666676666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 1e-12, 0.03222222222322222, 0.007222222223222222, 1e-12, 1e-12, 0.010555555556555556, 0.0038888888898888888, 0.004444444445444445, 1e-12, 0.002777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 0.002777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0033333333343333335, 1e-12, 0.0005555555565555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.007777777778777778, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011666666667666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.011666666667666668, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 0.07055555555655556, 1e-12, 1e-12, 0.012222222223222223, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0066666666676666675, 1e-12, 1e-12, 1e-12, 0.005555555556555556, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.012777777778777779, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.006111111112111112, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.11611111111211112, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.010000000001, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.0038888888898888888, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 0.017777777778777776, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]

for file in map_filenames:
    obstacle = 0
    with open(file, 'r') as json_file:
        # Read the contents of the .map file
        data = json.load(json_file)
            
    map_layout = data['layout']

    width = len(map_layout)
    height = len(map_layout[0])
    # print(file,width,height)
    all_grid = width * height
    # print(all_grid)
    print(file)
    norm_pattern_dist = calc_layout_entropy_test(map_layout)
    maze_kl_div = entropy(norm_pattern_dist, np.array(maze_ref_dist), base = len(maze_ref_dist))
    KL_div.append(maze_kl_div)

    for i in range(width):
        for j in range(height):
            if map_layout[i][j] == "@":
                obstacle += 1
    
    obstacles.append(obstacle/all_grid)

print(obstacles)
print(KL_div)

plt.scatter(KL_div, obstacles)

plt.xlim(0, 5)  # Optional padding of 1 unit
plt.ylim(0, 1)  # Optional padding of 1 unit
# Label the axes
plt.grid(True)
plt.xlabel('KL_div')
plt.ylabel('Proportion of obstalces')

# Add a title
plt.title('Archive')

# Show the plot
plt.savefig('plot_grid.png')
