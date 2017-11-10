#!/usr/bin/env bash

# Download mnist files
base_url="http://yann.lecun.com/exdb/mnist"
file_list=(
  train-images-idx3-ubyte train-labels-idx1-ubyte
  t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
)

for file in "${file_list[@]}"; do
  if [ ! -f ${file} ] ; then
    url="${base_url}/${file}.gz"
    curl ${url} | gzip -dv > ${file}
  fi
done
