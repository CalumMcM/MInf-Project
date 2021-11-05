#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

CUR_START_DATE='2015-05-01' # THIS SHOULD BE THE SAME AS IT IS IN ExtractLandsat8.py
NUM_IMGS=16 # The number of images that are downloaded per thread divided by 3 (given there are three biomes)
NUM_THREADS=50 

total_quad_images=$((NUM_IMGS*NUM_THREADS))

# Go through each quad number
for QUAD_NUM in {2..4} 
  do
    echo "QUAD_NUM: $QUAD_NUM"
    #for batch_num in {0..25..5} #{0..$total_quad_images..$NUM_IMGS}    #
    for (( batch_num = 0; batch_num < $total_quad_images; batch_num += $NUM_IMGS))
      do
        echo "STARTING BATCH NUM: $batch_num"
        python src/ExtractLandsat8.py --num_imgs $NUM_IMGS --seed $batch_num --QuadNum $QUAD_NUM --start_date $CUR_START_DATE &
    done
    wait
done
wait

# python ExtractLandsat8.py --num_imgs 1 --seed 0 --z`