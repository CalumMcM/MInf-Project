import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

quad = ['1','2','3', '4']
X = ['1      2      3      4\nAmazon', '1      2      3      4\nCerrado', '1      2      3      4\nCaatinga']
old_ama_quads = [6499,6901, 6701, 6772]
old_cer_quads = [6553, 6204, 6693, 6796]
old_cat_quads = [6573, 6202, 6687, 6710]

# No clouds but has corrupt
cloud_ama_quads = [4595,1776, 3571, 1967] # Training:     9929    Test:   1715    # Total: 11644
cloud_cer_quads = [5695, 5879, 5882, 5951] # Training:    17168   Test:   5851    # Total: 23019
cloud_cat_quads = [4747, 2910, 3704, 4677] # Training:    11952   Test:   3607    # Total: 15559
                                    # Total:        39049   Total:  11173   # Total: 50222


# No clouds or corrupt images
clean_ama_quads = [4453, 1701, 3458, 1899] # Training:     9810   Test:   1701    # Total: 11511
clean_cer_quads = [5529, 5712, 5730, 5758] # Training:    16971   Test:   5758    # Total: 22729
clean_cat_quads = [4569, 2798, 3571, 4498] # Training:    11865   Test:   3571    # Total: 15436
                                          # Total:        38646   Total:  11030   # Total: 49676
total_quad_1 = [6499, 6553, 6573]
total_quad_2 = [6901, 6204, 6202]
total_quad_3 = [6701, 6693, 6687]
total_quad_4 = [6772, 6796, 6710]

cloudy_quad_1 = [4595, 5695, 4747]
cloudy_quad_2 = [1776, 5879, 2910]
cloudy_quad_3 = [3571, 5882, 3704]
cloudy_quad_4 = [1967, 5951, 4677]

clean_quad_1 = [4453, 5529, 4569]
clean_quad_2 = [1701, 5712, 2798]
clean_quad_3 = [3458, 5730, 3571]
clean_quad_4 = [1899, 5758, 4498]

cloudy_quad_1 = np.array(cloudy_quad_1) - np.array(clean_quad_1) + np.array(total_quad_1)
cloudy_quad_2 = np.array(cloudy_quad_2) - np.array(clean_quad_2) + np.array(total_quad_2)
cloudy_quad_3 = np.array(cloudy_quad_3) - np.array(clean_quad_3) + np.array(total_quad_3)
cloudy_quad_4 = np.array(cloudy_quad_4) - np.array(clean_quad_4) + np.array(total_quad_4)



colours = ["red", "blue", "green"]
colours2 = ["blue", "green", "red"]
colours3 = ["grey", "grey", "grey"]
labels = ["Good", "", "Caatinga"]
labels = ["Amazon", "Cerrado", "Caatinga"]

def subcategorybar(X, vals, vals2, vals3, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        if i == 1:

            plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                    width=width/float(n), align="edge", color="red",edgecolor='black', label="Corrupt")

            plt.bar(_X - width/2. + i/float(n)*width, vals3[i],
                    width=width/float(n), align="edge", color="grey", edgecolor='black',label="Cloudy")

            plt.bar(_X - width/2. + i/float(n)*width, vals2[i],
                    width=width/float(n), align="edge", color="#22c916", edgecolor='black',label="Satisfactory")
        else:

            plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                    width=width/float(n), align="edge", color="red",edgecolor='black')

            plt.bar(_X - width/2. + i/float(n)*width, vals3[i],
                    width=width/float(n), align="edge", color="grey", edgecolor='black')

            plt.bar(_X - width/2. + i/float(n)*width, vals2[i],
                    width=width/float(n), align="edge", color="#22c916", edgecolor='black')

    plt.xticks(_X, X)

subcategorybar(X, [cloudy_quad_1,cloudy_quad_2,cloudy_quad_3,cloudy_quad_4], [clean_quad_1,clean_quad_2,clean_quad_3, clean_quad_4], [total_quad_1,total_quad_2,total_quad_3, total_quad_4])
plt.legend(loc="lower right")
plt.xlabel('Biome and Quadrant')
plt.ylabel('Number of Images')
#plt.title("Number of Images Per Quadrant for ecBiome")
plt.tight_layout()
plt.savefig("../../tile2vec/figures/quadrantImages.png")
plt.show()
