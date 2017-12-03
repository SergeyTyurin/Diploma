#!/usr/bin/env sh
set -e

/home/caffe/build/tools/caffe train --solver=lenet_solver.prototxt -gpu $1
