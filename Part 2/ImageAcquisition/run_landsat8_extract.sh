#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

#for batch_num in {3599..7040..110} # Started 7009 that is 31 off: = 3520+(105-(7040-70??))
for batch_num in {0..6016..94} #{ 0..640..10}
  do
    echo "STARTING BATCH NUM: $batch_num"
    python EarthEngine/ExtractLandsat8.py --num_imgs 94 --seed $batch_num --outputType 'RGB' --QuadNum 'vis' &
done

wait
# python ExtractLandsat8.py --num_imgs 1 --seed 0 --outputType 'RGB' --QuadNum 1