#!/bin/bash

protocols=("MI")

for protocol in "${protocols[@]}"; do
  ../../build/bin/pcc-translate "../$protocol.pcc" --pcc-to-mlir > "$protocol.mlir"
done




