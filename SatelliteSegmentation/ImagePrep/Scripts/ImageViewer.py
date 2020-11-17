from PIL import Image
import numpy as np
import rasterio as rs
import math
from rasterio.plot import show
import xarray as xr
import georaster
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from osgeo import gdal
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#####################################################
########## SCRIPT TO DISPLAY A .GEOTIF IMG ##########
##########   OR SAVE IMAGE MATRIX AS .TXT  ##########
#####################################################
"""
save_dir = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Amazonia/png'
scale = '-scale min_val max_val'

options_list = [
    '-ot Byte',
    '-of PNG',
    scale
]
options_string = " ".join(options_list)

gdal.Translate(save_dir,
               image_dir,
               options=options_string)
"""

os_dir = '/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/'

def plot_group(biome):
    path = os_dir + biome + '/tif'

    all_files = [f for f in listdir(path) if isfile(join(path, f)) ]
    png_files = [f.split('.')[0] for f in all_files if not 'png.aux.xml' in f and not '.DS_Store' in f]
    png_files.sort(key=int)
    print (png_files)
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    plt.title('NDVI Scores For Cerrado', pad=20)
    plt.axis('off')
    plt.tight_layout()
    columns = 5
    rows = 6
    for i in range(1, columns*rows +1):
        img = mpimg.imread(path + '/' + png_files[i-1] + '.tif')
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(str(png_files[i-1])) # Adds tile name
        plt.imshow(img)
        fig.tight_layout()
    plt.savefig(os_dir + biome + '/' + biome + '_classes.png')

    plt.show()

# Takes in the band of an image and saves the matrix
# for that band to a .txt file
def make_matrix(i):
    output_dir = "Extracted/LC080900862019072401T1-SC20200930162234/"

    img = Image.open(output_dir + 'LC08_L1TP_090086_20190724_20190801_01_T1_sr_band' + str(i) + '.tif')

    mat = np.matrix(img)
    with open('band'+str(i)+'.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

# Using Rasterio will plot an image
def view_image(image_dir):

    # Show the image:
    fp = r'/Users/calummcmeekin/Downloads/1.tif'
    img = rs.open(fp)

    # Print number of bands in the image:
    show(img.read([1,2,3]))

def view_batch_image(image_dir):
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and '.DS_Store' not in f]

    iter = 1

    for file in files:
        fp = r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/Amazonia/tif/' + file
        img = rs.open(fp)

        # Print number of bands in the image:
        show(img.read([1,2,3]))

def convertPNG(biome):

    image_dir = os_dir + biome + '/tif'
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and '.DS_Store' not in f]

    iter = 1

    for file in files:
        iter += 1

        print (file)

        save_dir = os_dir + biome + '/png/' + file.split('.')[0] + '.png'
        file_dir = os_dir + biome + '/tif/' + file

        scale = '-scale min_val max_val'

        print (file_dir)
        options_list = [
            '-ot Byte',
            '-of PNG',
            scale
        ]
        options_string = " ".join(options_list)

        gdal.Translate(save_dir, file_dir, options=options_string)

def plotNDVI():
    caatingaNDVI = [0.282,0.270,0.334,0.159,0.019,0.048,-0.018,0.114,-0.103,0.232,0.364,0.165,0.082,0.003,-0.021,-0.035,0.172,-0.119,0.040,0.040,0.040,-0.060,-0.128,0.025,0.048,-0.003,-0.260,0.354,0.061,-0.183]
    caatingaNDVIm = [0.2823461086798797, 0.5265747439792994, 0.4707043347055328, 0.43159584122660954, 0.4047154751971473, 0.26972919498135184, 0.19888039780020556, 0.11335088785802429, 0.20659872384354266, 0.3340599089327463, 0.17101818088032616, 0.1699797965278704, 0.18304529167401037, 0.13739078186322443, 0.15948769074548166, 0.030837077659197028, 0.04428192010547661, 0.05043061152412471, -0.010871004438681432, 0.3639478668531958, 0.34585390703694485, 0.34062359383199825, 0.38988508362746943, 0.29389730957919324, 0.16530529683844478, 0.11008867412384421, 0.10765708324008023, 0.14860821659390588, 0.1551332437857897, 0.08242598342408076, 0.0579028191667507, -0.06956130438897268, 0.008270720354268747, -9.649121324150615E-4, 0.002626351064776269, 0.022764019793449844, 0.015932970320386614, 0.04034780711838353, 0.10727603467295681, 0.1692223045797797, 0.06481817090468059, 0.07999671698952324, 0.039741612791815366, 0.10596467627820534, 0.02831427324941751, 0.016070623464201808, -0.008186175204013076, 0.03985825443034404, -0.0032115745520253603, 0.03123136714848712, -0.2259382914499103, -0.05979889158888384, 0.04808714971349215, -0.025698121157434174, 0.050642995360575864, 0.031217181108993398, -0.04526415988538376, -0.002743960635799839, 0.016376019637114818, 0.10614127268614709, -0.24425152707010214, -0.28590513740861073, -0.25949625185676334, -0.2727703027400947, -0.14017088866537594, 0.06598313875976408, 0.06135473497936668, 0.03730125268485757, 0.010127871749387871, 0.02247269122507389, -0.21449531369810204, -0.1833187052254317, -0.17085767329254511, -0.031550021321589294, -0.21858459246735185, -0.19274652787344285, -0.20950675829118823, 0.09834714262464357, -0.06338679175684064, 0.031715133858258124, 0.1565077845090734, -0.13843362170213722, -0.051975101534844366, 0.12297216157318987, -0.13419500853632824, -0.08161887454847291, 0.14908555865451073]

    cerradoNDVI = [0.181,0.270,0.086,0.117,0.111,0.062,0.028,0.191,0.141,0.227,0.092,0.291,0.124,0.267,0.149,0.334,0.123,0.251,0.250,0.261,0.297,0.192,0.264,0.225,0.163,0.157,0.157,0.044]
    cerradoNDVIm = [0.18095096462683377, 0.20929803786973025, 0.40673242731434284, 0.2483129473354336, 0.27006419685722316, 0.22635820241071475, 0.40385749625838563, 0.1834508945450152, 0.08869240887362263, 0.08585387506019265, 0.12463688411022039, 0.15771924483185762, 0.10055513948989954, 0.13103527817182278, 0.11663351606590504, 0.23954995503846382, 0.12628817369122491, 0.14771452335359503, 0.3635212845317317, 0.2272375672604866, 0.2538752836590807, 0.16238701493615632, 0.12684311535997952, 0.26361145026827143, 0.1313401416868143, 0.13175764854825842, 0.22305042882856632, 0.20573107853802836, 0.09197448295357037, 0.12450379158454601, 0.09990553687908889, 0.05843931644411852, 0.2021206418349322, 0.29099699685624286, 0.30710638232653253, 0.27314058994128565, 0.12251591547525859, 0.14673130051939814, 0.18300875980089495, 0.11470005385197442, 0.4360284767960416, 0.25007661835055894, 0.3562198230755845, 0.3127811094590075, 0.28678758857413267, 0.2502554756939352, 0.27760076059129446, 0.20884807456809312, 0.3169962422377134, 0.2129504298273226, 0.26104946549271707, 0.2640770721115836, 0.288219640126656, -0.0650539491548578, -0.05045236439031398, 0.25814411820333305, 0.22469898246302342, 0.22573107780500914, 0.26209783138039816, 0.29877748478807237, 0.21690032819794086, 0.1634338163071831, 0.1754378064931505, 0.19234294315739556, 0.04857942791869176, -0.06866976113855233, -0.03295588129256469, 0.1713333836279286, 0.029629622980045015, 0.044455027233345046, 0.3088172535106875, 0.06145152036807826, 0.12579451594121366, 0.0326900143856668, 0.04612860056969427, 0.0911988746125139, 0.04166396245249405, 0.031033171302127023, 0.07946023238559467, -0.0017287477903849863, -0.0070590439998321555, 0.06353450558712027, 0.09112994472907858, -0.022674274967487758, 0.12788429137051088, 0.14691854795418988]
    amazoniaNDVI = [-0.393,0.356,0.526,0.554,0.077,0.362,0.484,0.544,0.544,0.367,0.374,0.360,0.358,0.511,0.477,0.386,0.353,0.098,0.319,0.513,0.466,0.287,0.388,0.308,0.377,0.362,0.386,0.351,0.124]
    amazoniaNDVIm = [-0.3927615481359617, -0.3336745036252422, 0.3977855413801961, 0.4157985661035598, 0.397940212380932, 0.3564883420636795, 0.35170005898625184, 0.5361641338507876, 0.5533883125557116, 0.5260380996357731, 0.5360761657540948, 0.5510540737376183, 0.5350191912830943, 0.5280224391107743, 0.5535910771380033, 0.5581355420646417, 0.5314731880048079, 0.5568358693977234, 0.5214639342256583, 0.0770948602973265, -0.4068055026010037, 0.16417758459517706, 0.3752810431185135, 0.394490086883629, 0.3617121428837197, -0.31277830160772574, 0.5425606433122855, 0.5104630782460855, 0.45714687556525346, 0.48420661166502454, 0.5031246155320591, 0.48380383337261346, 0.5115678922457724, 0.5199874891202345, 0.543909539383239, 0.5489865824966016, 0.5096505051347746, 0.48442749770175586, 0.5680264298810261, 0.544377855847882, 0.15648024123676973, -0.5619192202491032, 0.36888108698161715, 0.31467038245561896, 0.3668670955265941, 0.3812439087412236, 0.3694315233503628, 0.3645718487807297, 0.4037058082741772, 0.37357819218072075, -0.4318685349107691, 0.34680650339702174, 0.36287376751580774, 0.34390739127634956, 0.36023722918036444, 0.38956396403414356, 0.36522250266089784, 0.3280031466908592, 0.28646602360839335, 0.3567583002360551, 0.3700630813049535, 0.4616641136183806, 0.40314316944603446, 0.48207366513103744, 0.5109246458456581, 0.5021079665503531, 0.4866630159014179, 0.5076367018851247, 0.4237637295355898, 0.476989260842547, 0.4632878047947618, 0.37971336046563675, 0.3709557122377787, 0.3629886482884487, 0.38569602507906153, 0.39182217295236543, 0.3900894148052068, 0.4144396138591339, 0.497609584084868, 0.3528865553759835, 0.32066825454342857, 0.2975411184056882, 0.31967880139085203, 0.3141477345512742, 0.09845863441133153, -0.009892089089171328, 0.26021188795747785, 0.2086740911834564, 0.1246907522286458, 0.3185384210479014, 0.3362551462044193, 0.2752036788589408, 0.5128671328998028, 0.5143376688796144, 0.5125622204708568, 0.509943038236188, 0.5291298810751938, 0.5044066319164227, 0.5119501419083232]
    print (len(amazoniaNDVI))
    xCAT = np.arange(1,31)
    xCER = np.arange(1,29)
    xAMA = np.arange(1,30)

    plt.scatter(xCAT, caatingaNDVI, color= 'green', marker='.', label='Caatinga')
    m, b = np.polyfit(xCAT, caatingaNDVI, 1)
    plt.plot(xCAT, m*xCAT + b, color= 'green', label='Caatinga LBF')

    plt.scatter(xCER, cerradoNDVI, color= 'blue', marker='*', label='Cerrado')
    m, b = np.polyfit(xCER, cerradoNDVI, 1)
    plt.plot(xCER, m*xCER + b, color= 'blue', label='Cerrado LBF')

    plt.scatter(xAMA, amazoniaNDVI, color= 'red', marker='v', label='Amazonia')
    m, b = np.polyfit(xAMA, amazoniaNDVI, 1)
    plt.plot(xAMA, m*xAMA + b, color= 'red', label='Amazonia LBF')

    plt.legend()
    plt.grid(color='grey', linestyle='-', linewidth=0.1)
    plt.ylim(-1,1)
    plt.xlim(0,35)
    plt.title('NDVI\'s For Each Biome')
    plt.xlabel('Tile Number')
    plt.ylabel('NDVI Score')

    plt.savefig(os_dir + 'ndviPlot.png')
    plt.show()


def plot_histogram(file_ama, file_cat, file_cer):

    df_ama = pd.read_csv(file_ama, sep=',',header=0, index_col =0)
    df_cat= pd.read_csv(file_cat, sep=',',header=0, index_col =0)
    df_cer = pd.read_csv(file_cer, sep=',',header=0, index_col =0)

    df_ama['Caatinga'] = df_cat['NDVI']
    df_ama['Cerrado'] = df_cer['NDVI']

    del df_ama['.geo']
    df_ama.columns = ['Amazonia', 'Caatinga', 'Cerrado']

    knn(df_ama)

    ax = df_ama.plot.hist(bins=100, alpha=0.9)
    plt.title("Median NDVI Frequency per Biome")
    plt.xlim(-1,1)
    plt.xlabel('Median NDVI')
    plt.savefig('medianNDVIHistogram.png')
    plt.show()

def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)

def kmeans(df):

    #data = {'sample': df.Amazonia,
    #        'class' : ['Amazonia']*len(df.Amazonia)}
    X = df.Amazonia + df.Caatinga + df.Cerrado
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0)
    X = np.array(X)
    SecDim = np.zeros(len(X))
    X = np.array(list(zip(X,SecDim)))

    kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
    fitted = kmeans.fit(X)
    prediction = kmeans.predict(X)

    plt.figure(figsize = (10,8))

    def plot_kmeans(kmeans, X, n_clusters=3, rseed=0, ax=None):
        labels = kmeans.fit_predict(X)

        # plot the input data
        ax = ax or plt.gca()
        ax.axis('equal')
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

        # plot the representation of the KMeans model
        centers = kmeans.cluster_centers_
        radii = [cdist(X[labels == i], [center]).max()
                 for i, center in enumerate(centers)]
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

    plot_kmeans(kmeans, X)


if __name__ == "__main__":

    MapBiome_data = '/Users/calummcmeekin/Downloads/mosaic-2019-0000000000-0000000000.tif'
    Landsat_image = '/Users/calummcmeekin/Downloads/LE07_220076_20000608.tif'

    #convertPNG('AmazoniaNN')

    #view_batch_image(r'/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/ImagePrep/Labelled/AmazoniaNN/tif')

    #plot_group("CerradoNN")
    #plotNDVI()
    #view_image(Landsat_image)

    dir_amazon = os_dir + 'AmazoniaNDVI/AmazoniaMedianNDVI1000.csv'
    dir_caat = os_dir + 'CerradoNDVI/CerradoMedianNDVI1000.csv'
    dir_cerr = os_dir + 'CaatingaNDVI/CaatingaMedianNDVI1000.csv'
    plot_histogram(dir_amazon, dir_caat, dir_cerr)
