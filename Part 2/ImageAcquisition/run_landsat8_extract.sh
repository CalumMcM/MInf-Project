#!/bin/bash
echo "Bash version${BASH_VERSION}..."

CUR_START_DATE='2020-05-01' # THIS SHOULD BE THE SAME AS IT IS IN ExtractLandsat8.py
NUM_IMGS=20 # The number of images that are downloaded per thread divided by 3 (given there are three biomes)
NUM_THREADS=50 

total_quad_images=$((NUM_IMGS*NUM_THREADS))

# Go through each quad number
# for QUAD_NUM in {1..4} 
#   do
#     echo "QUAD_NUM: $QUAD_NUM"
#     #for batch_num in {0..25..5} #{0..$total_quad_images..$NUM_IMGS}    #
#     for (( batch_num =0; batch_num < $total_quad_images; batch_num += $NUM_IMGS))
#       do
        
#          echo "STARTING BATCH NUM: $batch_num"
#          python src/ExtractLandsat8.py --num_imgs $NUM_IMGS --seed $batch_num --QuadNum $QUAD_NUM --start_date $CUR_START_DATE --OutputType "Date" &
         
#     done
    
#     wait
# done
# wait

# For interuptted execution of previous threads
# for QUAD_NUM in {1..4} 
#   do
#     echo "QUAD_NUM: $QUAD_NUM"
#     #for batch_num in {0..25..5} #{0..$total_quad_images..$NUM_IMGS}    #
#     for (( batch_num =0; batch_num < $total_quad_images; batch_num += $NUM_IMGS))
#       do
#         # if [ $QUAD_NUM -eq 4 ];
#         # then
#         # # num_imgs = NUM_IMGS-seed
#         # # seed = broken point
#         #   temp_batch_num=$((batch_num+29))
#         #   temp_num_images=$((NUM_IMGS-29))
#         #   echo "STARTING BATCH NUM: $temp_batch_num"
#         #   python src/ExtractLandsat8.py --num_imgs $temp_num_images --seed $temp_batch_num --QuadNum $QUAD_NUM --start_date $CUR_START_DATE --OutputType "Date" &
#         # fi 
#         if [ $QUAD_NUM -gt 3 ];
#         then
#           echo "STARTING BATCH NUM: $batch_num"
#           python src/ExtractLandsat8.py --num_imgs $NUM_IMGS --seed $batch_num --QuadNum $QUAD_NUM --start_date $CUR_START_DATE --OutputType "Date" &
#         fi
#     done
#     wait
# done
# wait

# For Visual Extractions:
for (( batch_num =0; batch_num < $total_quad_images; batch_num += $NUM_IMGS))
  do
    echo "STARTING BATCH NUM: $batch_num"
    python src/ExtractLandsat8.py --num_imgs $NUM_IMGS --seed $batch_num --QuadNum "vis" --start_date $CUR_START_DATE --OutputType "RGB" &
done
wait


# python ExtractLandsat8.py --num_imgs 1 --seed 0 --z`

## WHEN DOWNLOADING NEW IMAGES:
# MAKE SURE THE YEAR AT THE TOP OF THIS SCRIPT is the
# same as the year in the ExtractLandsat8.py script
# MAKE SURE TO CHANGE THE FOLDER NAME in the 
# ExtractLandsat8.py main() function 
# MAKE SURE TO ChANGE THE DATE in the filter 
# in the ExtractLandsat8.py file