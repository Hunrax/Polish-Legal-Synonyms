#!/bin/bash
export PATH="$HOME/java-jre/bin:$PATH"
nohup python workflows/automated_bert_dbscan.py "${1:-1}" > output_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &