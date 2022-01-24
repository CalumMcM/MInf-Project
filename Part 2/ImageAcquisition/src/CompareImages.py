import numpy as np

def process_file(f):

    lines = f.readlines()

    biome_points = {"ama" : [], "cer": [], "cat": [], "inc": []}
    biome = "ama"

    for line in lines:
        # Determine Current Biome
        if "ama_fc" in line:
            biome = "ama"
            pass
        elif "cer_fc" in line:
            biome = "cer"
            pass
        elif "cat_fc" in line:
            biome = "cat"
            pass
        elif "inconclusive_fc" in line:
            biome = "inc"
            pass
        # Classify future points
        if "ee.Feature(" in line:
            all_points = line.split("(")[2]
            bl_points = [all_points.split(",")[0], all_points.split(",")[1]]

            biome_points[biome].append(bl_points)
            
    return biome_points
        
def compare_points(old_points, new_points):

    conversions = {
        "ama->cer": 0,
        "ama->cat": 0,
        "ama->inc": 0,
        "cer->ama": 0,
        "cer->cat": 0,
        "cer->inc": 0,
        "cat->ama": 0,
        "cat->cer": 0,
        "cat->inc": 0,
        "inc->cat": 0,
        "inc->cer": 0,
        "inc->ama": 0
        }
    converted = 0
    for new_biome, new_biome_points in new_points.items():

        for point in new_biome_points:
            found = False
            for old_biome, old_biome_points in old_points.items():
                if point in old_biome_points and old_biome != new_biome:
                    key = old_biome + "->" + new_biome
                    conversions[key] += 1
                    converted += 1
                elif point in old_biome_points:
                    found = True
    
    new_num_points = 0
    for new_biome, new_biome_points in new_points.items():
        new_num_points += len(new_biome_points)
    old_num_points = 0
    for old_biome, old_biome_points in old_points.items():
        old_num_points += len(old_biome_points)

    print ("Converted {}/{}|{}".format(converted, old_num_points, new_num_points))
    
    return conversions

def main():

    f_2016 = open("EarthEngine_Classifications2016.txt", "r")
    f_2021 = open("EarthEngine_Classifications2021.txt", "r")

    biome_points_2016 = process_file(f_2016)
    biome_points_2021 = process_file(f_2021)

    conversions = compare_points(biome_points_2016, biome_points_2021)

    for key, value in conversions.items():
        print ("{} : {}".format(key, value))

if __name__ == "__main__":
    main()