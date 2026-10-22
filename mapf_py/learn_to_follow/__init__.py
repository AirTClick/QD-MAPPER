"""Inference-only subset of Learn-to-Follow (Follower), vendored for QD-MAPPER.

Source: https://github.com/Cognitive-AI-Systems/learn-to-follow (commit d02680a, MIT licence).
QD-MAPPER changes: the top-level imports `follower`, `env`, `utils` and `follower_cpp` are
rewritten to `mapf_py.learn_to_follow.<name>` so they cannot clash with other modules in
the Dask workers. The trained weights are read from env_search_PBS/LTF/model/follower.
"""
