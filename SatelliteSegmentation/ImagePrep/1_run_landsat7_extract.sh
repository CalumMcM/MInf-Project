#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for batch_num in {0..3800..950}
  do
      echo "STARTING NEW BATCH AT: $batch_num"
      python EarthEngine/ExractLandsat7.py --num_imgs 950 --seed $batch_num
  done


for batch_num in {4750..7680..950}
do
    echo "STARTING NEW BATCH AT: $batch_num"
    python EarthEngine/ExtractLandsat7.py --num_imgs 950 --seed $batch_num
done
