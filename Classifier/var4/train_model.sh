#!/usr/bin/env sh
set -e

/home/caffe/build/tools/caffe train --solver=solver.prototxt -gpu $1
