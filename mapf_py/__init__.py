"""Python MAPF solvers plugged into QD-MAPPER.

- PIBT: vendored pypibt (https://github.com/Kei18/pypibt) -> warehouse_sim_PIBT
- LaCAM: vendored pylacam, branch `pibt` (https://github.com/Kei18/pylacam) -> warehouse_sim_LaCAM
- Learn-to-Follow: vendored learn-to-follow
  (https://github.com/Cognitive-AI-Systems/learn-to-follow) -> warehouse_sim_LTF

Each ``warehouse_sim_<ALG>`` module plays the role of the C++ modules
``warehouse_sim_PBS`` / ``warehouse_sim_EECBS``: it takes the repaired map JSON
produced by QD-MAPPER's MILP repair and returns a JSON string with the
statistics that ``WarehouseModule.process_eval_result`` reads.

Learn-to-Follow needs extra packages (POGEMA, sample-factory, ...) that are not
in the container. scripts/install_mapf_py_deps.sh installs them into
mapf_py/_deps. See MAPF_PY_CHANGELOG.md at the repository root for every change.
"""
import os
import site
import sys

DEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deps")


def add_deps_to_path():
    """Makes the packages installed by scripts/install_mapf_py_deps.sh importable.

    site.addsitedir() appends the folder to sys.path, so the container's own
    numpy and torch keep priority, and it processes the folder's .pth files
    (protobuf needs them).
    """
    if not os.path.isdir(DEPS_DIR):
        raise ImportError(f"{DEPS_DIR} not found. Run "
                          "`bash scripts/install_mapf_py_deps.sh` first.")
    if DEPS_DIR not in sys.path:
        site.addsitedir(DEPS_DIR)
