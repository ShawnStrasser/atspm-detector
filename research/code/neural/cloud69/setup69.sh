#!/usr/bin/env bash
set -e
[ -d /root/pkg ] || cp -r /workspace/cloud_pkg /root/pkg
cp /workspace/c69/run69.sh /root/pkg/run69.sh; chmod +x /root/pkg/run69.sh
cp /workspace/c69/tcn69_func.py /root/pkg/code/research/code/neural/tcn69_func.py
cp /workspace/c69/data2.py /root/pkg/code/research/code/neural/data2.py
mkdir -p /workspace/out69
echo setup ok
