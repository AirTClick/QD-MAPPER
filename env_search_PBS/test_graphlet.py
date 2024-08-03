import numpy as np
import json
# use while running the experiment
# from env_search_PBS.utils import (eecbs_obj_types, MIN_SCORE,
#                               eecbs_env_number2str,eecbs_env_str2number,
#                               kiva_env_str2number, format_env_str,
#                               read_in_kiva_map, flip_tiles)
from utils import (eecbs_obj_types, MIN_SCORE,
                              eecbs_env_number2str,eecbs_env_str2number,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)
import os
import copy
import itertools
import fire
from scipy.stats import entropy

obj_types = "@."

def env_str2number(env_str):
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)

# doing the node hashing 
def node_connection(map_int,map_hash):
    n_row, n_col = map_int.shape
    for i in range(n_row):
        for j in range(n_col):
            if map_int[i][j] == 0:
                continue
            else:
                if i == 0 and j == 0:
                    number = str(map_int[i,j+1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == n_row-1 and j == 0:
                    number = str(map_int[i,j+1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == 0 and j == n_col - 1:
                    number = str(map_int[i,j-1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                elif i == n_row-1 and j == n_col - 1:
                    number = str(map_int[i,j-1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)

                #upper and down edge 
                elif i == 0:
                    number = str(map_int[i,j-1] * 100 + map_int[i,j+1] * 10 + map_int[i+1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
            
                elif i == n_row -1:
                    number = str(map_int[i,j-1] * 100 + map_int[i,j+1] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                #left and right edge  
                elif j == 0:
                    number = str(map_int[i,j+1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    
                elif j == n_col -1:
                    number = str(map_int[i,j-1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    
                #normal cases
                else:
                    number = str(map_int[i,j-1] * 1000 + map_int[i,j+1] * 100 + map_int[i+1,j] * 10 + map_int[i-1,j])
                    result_number = int(number.replace("0", ""))
                    if (i,j) not in map_hash:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
                    else:
                        map_hash[(i,j)] = (map_int[i][j],result_number)
    
    return map_hash

# add to the graphlet vector 
def add_to_vector(map,glv):
    n_row, n_col = map.shape
    for i in range(n_row):
        for j in range(n_col):
            if map[i][j] not in glv:
                glv[map[i][j]] = 1
            else:
                glv[map[i][j]] += 1 
    
    return glv

# normalize the graphlet vector
def normalize(glv,sum_value):
    for key in glv.keys():
        glv[key] /= sum_value     
    return glv

# sort an integer
def sort_int(num):
    # Convert the integer to a string
    num_str = str(num)
    
    # Convert the string to a list of characters
    num_list = list(num_str)
    
    # Sort the list of characters
    num_list.sort()
    
    # Join the sorted list back into a string
    sorted_num_str = ''.join(num_list)
    
    # Convert the string back to an integer
    sorted_num = int(sorted_num_str)
    
    return sorted_num


def node_hashing(json_file):
    # open the json file and convert from string to int
    print("Node Hashing begin: ")
    # with open(map_path, "r") as map_json:
    #     map_data = json.load(map_json)

    map_string = json_file["layout"]
    map_int = env_str2number(map_string)

    # directionary for collecting the index and value after connection check
    map_hash = dict()

    n_row, n_col = map_int.shape

    map_hash = node_connection(map_int,map_hash)
    
    # create graphlet vector and add initial map information 
    glv = dict()

    glv = add_to_vector(map_int,glv) 

    # first hash table
    hash_table_a = dict()

    hash_table_a[(1,1)] = 2
    hash_table_a[(1,11)] = 3
    hash_table_a[(1,111)] = 4
    hash_table_a[(1,1111)] = 5

    # create the map after one iteration of hashing

    map_one_iter = copy.deepcopy(map_int)

    for key in map_hash:
        x_axis = key[0]
        y_axis = key[1]

        map_one_iter[x_axis][y_axis] = hash_table_a[map_hash[key]]

    # add first iteration hash information to the graphlet vector

    glv = add_to_vector(map_one_iter,glv) 

    # do the second node connection 

    map_hash = node_connection(map_one_iter,map_hash)

    # create the second hash table

    last_hash = [2,3,4,5]
    new_hash = []

    # Collect all combinations from length 1 to 4
    all_combinations = []
    for length in range(1, 5):  # Lengths from 1 to 4
        combinations = itertools.combinations_with_replacement(last_hash, length)
        all_combinations.extend(combinations)
    

    all_combinations = list(all_combinations)

    # Normalize combinations
    normalized_combinations = {tuple(sorted(combination)) for combination in all_combinations}

    # Convert back to a list of tuples
    unique_combinations = list(normalized_combinations)

    # Print all combinations
    for combination in unique_combinations:
        # Join the tuple into a string for a better readable format
        new_hash_value = ''.join(map(str, combination)) 
        new_hash.append(int(new_hash_value))


    num = 6 
    new_hash_table = dict()
    for i in last_hash:
        for j in new_hash:
            new_hash_table[(i,j)] = num
            num += 1
    

    # create the map after the second iteration of hashing

    map_second_iter = copy.deepcopy(map_one_iter)

    for key in map_hash:
        x_axis = key[0]
        y_axis = key[1]

        value_0 = map_hash[key][0]
        value_1 = map_hash[key][1]
        value_1 = sort_int(value_1)
        value = (value_0,value_1)

        map_second_iter[x_axis][y_axis] = new_hash_table[value]

    # add second iteration hash information to the graphlet vector

    glv = add_to_vector(map_second_iter,glv) 
    glv[0] /= 3

    # delete the first two positions (since it only represent number of obstacles and empty space)
    glv[0] = 0 
    glv[1] = 0

    # do normalization to the graphlet vector
    sum_value = 0
    epsilon=1e-12

    for value in glv.values():
        sum_value += value

    glv_normalized = copy.deepcopy(glv)

    glv_normalized = normalize(glv_normalized,sum_value)
    
    final_node_array = np.zeros(281)

    for key in glv_normalized.keys():
        final_node_array[key] = glv_normalized[key]
    
    # for key in glv.keys():
    #     final_node_array[key] = glv[key]

    for i in range(0,len(final_node_array)):
        final_node_array[i] += epsilon

    return final_node_array.tolist()

# map_path = "room_maps_json/16room_007.map.json" 
# result_compare_list2 = []

# with open("maze_32_32_2.json", "r") as file:
#     data3 = json.load(file)

# node2 = node_hashing(data3)

# for i in range(len(data)-1):
#     dot_result = np.dot(node2,data[i]) 
#     result_compare_list2.append(dot_result)
# print("---------------------------------------")
# print(node2)

# print("---------------------------------------")
# print("Compare_list: ",result_compare_list2)
# print("---------------------------------------")
# diff_list2= [result_list[i]-result_compare_list2[i] for i in range(0,len(result_compare_list2))] 
# print(sum(diff_list2))

def avg_pooling_2d(input_array, pool_size, stride):
    # Get the dimensions of the input array
    input_height, input_width = input_array.shape
    
    # Calculate the dimensions of the output array
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    
    # Initialize the output array
    output_array = np.zeros((output_height, output_width))
    
    # Perform the pooling operation
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Calculate the start and end indices of the current pooling window
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + pool_size
            end_j = start_j + pool_size
            
            # Extract the pooling window
            window = input_array[start_i:end_i, start_j:end_j]
            
            # Calculate the average value of the window and assign it to the output array
            output_array[i, j] = window[0][0]
    
    return output_array


# Get the map fully connected
def find_starting_points(grid):
    points = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '.':
                points.append((i, j))
    return points

def flood_fill(grid, x, y, visited, current_component):
    # Define the directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [(x, y)]
    
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited and grid[cx][cy] == '.':
            visited.add((cx, cy))
            current_component.append((cx, cy))
            # Add neighboring cells to the stack
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                    stack.append((nx, ny))
    return current_component

def find_components(grid):
    visited = set()
    components = []
    starting_points = find_starting_points(grid)
    
    for point in starting_points:
        if point not in visited:
            component = flood_fill(grid, point[0], point[1], visited, [])
            components.append(component)
    
    return components

def connect_components(grid, components):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(len(components) - 1):
        component1 = components[i]
        component2 = components[i + 1]
        min_distance = float('inf')
        connect_points = None
        
        for x1, y1 in component1:
            for x2, y2 in component2:
                distance = abs(x1 - x2) + abs(y1 - y2)
                if distance < min_distance:
                    min_distance = distance
                    connect_points = (x1, y1, x2, y2)
        
        if connect_points:
            x1, y1, x2, y2 = connect_points
            if x1 == x2:  # Same row
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid[x1][y] = '.'
            elif y1 == y2:  # Same column
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid[x][y1] = '.'
            else:  # Different row and column
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid[x1][y] = '.'
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid[x][y2] = '.'

def fully_connect_dots(grid):
    grid = [list(row) for row in grid]
    
    components = find_components(grid)
    connect_components(grid, components)
    
    return [''.join(row) for row in grid]


directory = "room8-1"
output_dir = "room32-1"

def filter_32 (directory, output_dir):
    json_files = [file for file in os.listdir(directory) if file.endswith('.json')]

    json_list = []

    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        with open(file_path, 'r') as file:
            data = json.load(file)
            map_string = data["layout"]
            map_int = eecbs_env_str2number(map_string)

            # filter the 512 * 512 map to 32 * 32 map
            output_array = avg_pooling_2d(map_int, 418, 3).astype(int)
            output_map = eecbs_env_number2str(output_array) 

            output_map = fully_connect_dots(output_map)

            data_json = dict()
            data_json["layout"] = output_map

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            file_path = f"{output_dir}/{json_file}"

            with open(file_path, "w") as f:
                json.dump(data_json, f, indent = 4)

# filter_32(directory, output_dir)


def main(
    directory_path: str,
):  

    json_files = [file for file in os.listdir(directory_path) if file.endswith('.json')]

    json_list = []
    node_hashing_list = []

    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                json_list.append(data)
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")

    # do node hashing for all json files in map_json folder and store them in a json file

    for i in json_list:
        hashed = node_hashing(i)
        node_hashing_list.append(hashed)

    with open('node_hashing_maze.json', 'w') as f: 
        json.dump(node_hashing_list, f)
   

def rotate(matrix):
    return [list(reversed(col)) for col in zip(*matrix)]

# test for rotation
# with open("maze_32_32_2.json", 'r') as f:
#         data = json.load(f) 

# layout = data["layout"]
# layout = eecbs_env_str2number(layout)

# rotate_layout = rotate(layout)
# rotate_layout = eecbs_env_number2str(np.array(rotate_layout))

# print(rotate_layout)

# if __name__ == "__main__":
#     fire.Fire(main)

def test_on_other_maps(map_path, hashing_path, compare_path):
    with open(hashing_path, 'r') as f:
        data = json.load(f) 

    node_hashing_array = []
    result_list = []    

    for i in range(len(data)-1):
        dot_result = np.dot(data[i],data[i])
        result_list.append(dot_result)

    print("Identity List: ", result_list)

    print("---------------------------------------")

    result_compare_list = []
    result_compare_list3 = []

    # test on the value of node hashing
    with open(map_path, "r") as file:
        data2 = json.load(file)

    node = node_hashing(data2)
    print("Node hashing: ", node)

    for i in range(len(data)-1):
        dot_result = np.dot(node,data[i]) 
        result_compare_list.append(dot_result)


    print("---------------------------------------")
    # print("Compare_list: ",result_compare_list)
    print("---------------------------------------")
    diff_list = [result_list[i]-result_compare_list[i] for i in range(0,len(result_compare_list))] 
    proportion_list = [result_compare_list[i] / result_list[i] for i in range(0,len(diff_list))]
    # print("Proportion_list: ", proportion_list)
    print(sum(diff_list))
    epsilon=1e-12
    with open(compare_path, "r") as file:
        data3 = json.load(file)
    
    node3 = node_hashing(data3)

    node = [node[i] + epsilon for i in range(len(node))]
    node3 = [node3[i] + epsilon for i in range(len(node3))]
    print("Node hashing Compare: ", node3)
    print(np.dot(node3,node3))
    print(np.dot(node,node3))
    print(entropy(node,node3, base = 281))
    diff_list_node = [node3[i]-node[i] for i in range(0,len(node))]  
    print("diff in node hashing: ", diff_list_node)
    # for i in range(len(data)-1):
    #     dot_result = np.dot(node3,data[i]) 
    #     result_compare_list3.append(dot_result)


    # print("---------------------------------------")
    # print("Compare_list: ",result_compare_list3)
    # print("---------------------------------------")
    # diff_list3 = [result_list[i]-result_compare_list3[i] for i in range(0,len(result_compare_list3))] 
    # print(sum(diff_list3))

# test_on_other_maps("maze_maps_json/maze512-32-9.map.json","node_hashing_maze.json","maze_32_32_test_1.json")

with open("maze512-1-0.map.json", "r") as f:
    data = json.load(f)
node = node_hashing(data)
epsilon=1e-12
node = [node[i] + epsilon for i in range(len(node))]
print(node)

# write hashing identity to a json file
# with open("node_hashing_maze.json", 'r') as f:
#         data = json.load(f) 

# result_list = []    

# for i in range(len(data)):
#     dot_result = np.dot(data[i],data[i])
#     result_list.append(dot_result)
    
# new_dict = dict()
# new_dict["hashing"] = result_list 

# with open("node_hashing_maze_identity.json", "w") as f:
#     json.dump(new_dict, f, indent = 4)

def maze_hashing():
    with open ("env_search_PBS/node_hashing_maze.json","r") as f:
        data = json.load(f)
    return data

def maze_hashing_identity():
    with open ("env_search_PBS/node_hashing_maze_identity.json","r") as f:
        data = json.load(f)
    return data["hashing"]

