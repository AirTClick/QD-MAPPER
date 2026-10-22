#!/bin/bash
# Installs the extra Python packages that mapf_py needs for Learn-to-Follow
# (POGEMA, sample-factory, ...) into mapf_py/_deps. It uses the container's pip,
# so the packages match its Python 3.9; no sudo or container rebuild is needed.
# PIBT and LaCAM need only numpy and work without this step.
#
# Usage (from the repository root):
#   bash scripts/install_mapf_py_deps.sh

set -e

SIF=singularity/ubuntu_warehouse.sif
DEPS=mapf_py/_deps

# --no-deps so the container's numpy and torch are never replaced. Versions
# follow learn-to-follow's docker/requirements.txt, except pydantic 1.10: 1.9.1
# fails with the container's typing_extensions 4.16.
singularity exec --cleanenv "$SIF" pip install --no-deps --upgrade --target "$DEPS" \
  pydantic==1.10.26 gymnasium==0.28.1 jax-jumpy==1.0.0 farama-notifications==0.0.6 \
  pogema==1.3.0 pogema-toolbox==0.1.0 sample-factory==2.1.1 signal-slot-mp==1.0.5 \
  faster-fifo==1.5.2 colorlog==6.8.2 tensorboardX==2.6.2.2 protobuf==3.20.3 \
  filelock==3.13.1 cppimport==22.8.2 mako==1.3.5 pybind11==2.13.1

# Compile LTF's C++ planner (cppimport) once, so the Dask workers don't all
# compile it at the same time.
singularity exec --cleanenv "$SIF" python -c \
  "from mapf_py.warehouse_sim_LTF import build_planner; build_planner()"

echo "mapf_py dependencies installed in $DEPS"
