import json

def maze_hashing():
    with open ("env_search_PBS/node_hashing_maze.json","r") as f:
        data = json.load(f)
    return data