#!/usr/bin/env python
# -*- coding: utf-8 -*-
from geetools import batch
import argparse
import ee
import time

#start_dates = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01']
#end_dates = ['2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31']

_start_date = ""
# Returns false if the requested number of images (value)
# is greater than 950 or less than 1
def check_valid_req(value):
    ivalue = int(value)
    if ivalue > 950 or ivalue < 1:
        raise argparse.ArgumentTypeError("%s is an invalid number of images" % value)
    return ivalue

# Returns a named tuple with arguments
def get_args():

    parser = argparse.ArgumentParser(
        description='Welcome to the bulk LANDSAT 8 image extraction helper script')

    parser.add_argument('--num_imgs', nargs="?", type=check_valid_req, default=1, help='Total number of images you wish to extract (Max 950)')

    parser.add_argument('--seed', nargs="?", type=int, default=0,
                        help='Initial seed that will be used for random feature point creation')

    parser.add_argument('--start_date', nargs="?", type=str, default=0,
                        help='\'YYY-MM-DD\' for the starting date that is used in this script')

    parser.add_argument('--QuadNum', nargs="?", type=str, default=0,
                        help='The quadrant you want to extract from.\n Choices: 1=Quad 1, 2=Quad 2, 3=Quad 3, 4=Quad 4, vis=Visual')

    parser.add_argument('--OutputType', nargs="?", type=str, default="RGB",
                        help='Type of date to be extracted for each image, Date = date, RGB = RGB')


    args = parser.parse_args()

    #print (args)

    return args

# Given a geometry point this returns a 1km (polygon) tile
# with the given point being the bottom left corner
def getCoord(feature):

  x1 = feature.geometry().coordinates().get(0)
  y1 = feature.geometry().coordinates().get(1)


  x2 = ee.Number(0.01364).add(x1)
  y2 = ee.Number(0.01364).add(y1)

  sliding_geometry = ee.Geometry.Polygon([[
  [x1, y1],
  [x2, y1],
  [x2, y2],
  [x1, y2],
  ]]);

  return ee.Feature(sliding_geometry)


# Given a polygon geometry this returns a clipped
# Landsat 7 image of that geometry
def getImages(feature):
  geo = ee.Geometry.Polygon(feature.geometry().coordinates().get(0))
  centroid = feature.geometry().centroid();

  image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2020-05-01', '2020-09-30').filterBounds(geo).sort('CLOUD_COVER').first(); # September -> April

  image = image.clip(geo)

  return ee.Image(image)


# Returns an image with only the Blue, Green and Red bands
def getBGR (image):
  return image.select(['B1','B2', 'B3', 'B4','B5','B6','B7'])

"""
def maskClouds(image):

  # Add a cloud score band.  It is automatically called 'cloud'.
  scored = ee.Algorithms.Landsat.simpleCloudScore(image);

  # Create a mask from the cloud score and combine it with the image mask.
  mask = scored.select(['cloud']).lte(0);

  # Apply the mask to the image and display the result.
  masked = image.updateMask(mask);

  return masked
"""

def getDate(image):
    return ee.Date(image)

def createFeature(date):

    return ee.Feature(None, {"Date" : date})

# Extracts the red and near-infrared bands from an image
# and computes the NDVI band, appending it to the returned image
def addNDVI(image):
  nir = image.select(['B4']);
  red = image.select(['B3']);

  # NDVI = (NIR-RED)/(NIR+RED)
  result = image.normalizedDifference(['B4', 'B3']).rename('NDVI');

  return image.addBands(result);


# Returns the mean NDVI score for a given image
def meanNDVI(image):

    ndvi = image.select('NDVI')

    ndviMean = ndvi.reduceRegion(ee.Reducer.median()).get('NDVI')

    return ee.Feature(None, {'NDVI': ndviMean})

# Normalised the passed image
def scale(image):
  # calculate the min and max value of an image
  minMax = image.reduceRegion( reducer= ee.Reducer.minMax(),
    geometry= image.geometry(),
    scale= 30,
    maxPixels= 10e9,
    tileScale= 16 # Reduces image size making it exportable
  );
  def scaler(name):
      name = ee.String(name)
      band = image.select(name)
      return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))

  # use unit scale to normalize the pixel values
  unitScale = ee.ImageCollection.fromImages(
    image.bandNames().map(scaler)).toBands().rename(image.bandNames());

  return unitScale


# Given a biomes geometry this function will extract either the NDVI
# data or the BGR images - based on the flag BGR_images - for each
# tile in the image
def getBiomeImage(biomeGeometry, geometry_colour, biome, BGR_images, seed, num_points, cur_point):

    # Add geomtry area to Google Earth Engine
    #Map.addLayer(biomeGeometry, {color: geometry_colour}, biome);

    # Get Feature Collection of random points within biome geometry
    random_points_FC = ee.FeatureCollection.randomPoints(biomeGeometry, num_points, seed, 10)

    # For selected list of points:
    chosen_FC = ProjectPresentation_GridPoints(cur_point)

    geometries = chosen_FC.map(getCoord)

    images = ee.ImageCollection(geometries.map(getImages, True))

    # Remove images that have any cloud cover
    filteredClouds = images.filter(ee.Filter.lte("CLOUD_COVER", 10))

    # If BGR_images flag is false then extract NDVI data from each image tile
    if (not BGR_images):

        """ FOR NDVI """

        # withNDVI = filteredClouds.map(addNDVI);

        # """ TO ADD EACH NDVI IMAGE TO THE MAP
        # ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']}

        # Map.addLayer(withNDVI.select('NDVI'),
        #          {min: -1, max: 1, palette: ['blue', 'white', 'green']},
        #          'NDVI classification');
        # """

        # meanNDVIs = ee.FeatureCollection(withNDVI.map(meanNDVI))

        # return meanNDVIs

        """ FOR DATE """

        dates = ee.List(images.aggregate_array("system:time_start")).map(getDate);
      
        datesFeatureCollection = ee.FeatureCollection(dates.map(createFeature))

        return datesFeatureCollection

    else:
        # For now extract all bands
        imagesBGR = filteredClouds.map(getBGR);
        #imagesBGR = filteredClouds;
        return imagesBGR.first();


# Export the given image to the Google drive in the
# folder named folder with the given file name
def export(image, folder, fileName):

    task = ee.batch.Export.image.toDrive(image=image,
                                     folder=folder,
                                     fileNamePrefix=str(fileName))
    task.start()

    status = task.status()['state']

    print ("BIOME: " + folder + "\tDATE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))

    while (status != 'COMPLETED' and status != 'FAILED'):

        status = task.status()['state']

        time.sleep(10)

    print ("BIOME: " + folder + "\tDATE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))


# Export the given image to the Google drive in the
# folder named folder with the given file name
def exportTable(table, folder, fileName):

    task = ee.batch.Export.table.toDrive(collection=table,
                                        folder=folder,
                                        fileNamePrefix=str(fileName),
                                        fileFormat='CSV')
    task.start()

    status = task.status()['state']

    print ("BIOME: " + folder + "\tTABLE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))

    while (status != 'COMPLETED' and status != 'FAILED'):

        status = task.status()['state']
        time.sleep(10)

    print ("BIOME: " + folder + "\tTABLE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))

def geometryPicker(quadNum, biome):
    """
    Depending on the batchnumber and biome returns the corresponding geometry
    """
    if quadNum == 1:
        if biome == "Amazonia":
            return geometryAMAQuad1()
        if biome == "Cerrado":
            return geometryCERQuad1()
        if biome == "Caatinga":
            return geometryCATQuad1()
    if quadNum == 2:
        if biome == "Amazonia":
            return geometryAMAQuad2()
        if biome == "Cerrado":
            return geometryCERQuad2()
        if biome == "Caatinga":
            return geometryCATQuad2()
    if quadNum == 3:
        if biome == "Amazonia":
            return geometryAMAQuad3()
        if biome == "Cerrado":
            return geometryCERQuad3()
        if biome == "Caatinga":
            return geometryCATQuad3()
    if quadNum == 4:
        if biome == "Amazonia":
            return geometryAMAQuad4()
        if biome == "Cerrado":
            return geometryCERQuad4()
        if biome == "Caatinga":
            return geometryCATQuad4()


def visualExtract():
    """
    A function used for specifically only exporting from one
    geometry, usually without regard for the biome the geometry
    is in and so it's main purpose is for predictions on new unseen data
    """
    
    args = get_args()

    print ("Thread {} Start_Date: {}  num_imgs: {} seed: {} quad_num: {}                            STARTING\tVisual Strain".format(args.seed, args.start_date, args.num_imgs, args.seed, args.QuadNum))

    geometryVisual = AreaofDeforestation()

    # Extract RGB
    if args.OutputType == "RGB":

        # Number of total tiles to extract
        N = args.seed+args.num_imgs

        # Only extract 1 tile for each biome at a time
        num_points = 1
        
        # Variable purely for Ama_To_Inc Exploration:
        cur_point = 0

        for i in range(args.seed, N):

            biomeVisual = getBiomeImage(geometryVisual, 'ff0aad', 'Visual', True, i, num_points, cur_point)

            visCollection = ee.FeatureCollection([biomeVisual]);

            if (ee.Algorithms.IsEqual(visCollection.size(), 1)):
                export(biomeVisual, 'PP_GridPoints_2020', i)
                cur_point += 1


    # Extract NDVI
    elif args.OutputType == "NDVI":

        # Total number of tiles that will be extracted
        N = args.seed+args.num_imgs

        # Extract 1 tile for each biome at a time
        num_points = 1

        mainVisualFC = getBiomeImage(geometryVisual, 'ff0aad', 'Visual', False, 0, num_points)

        for i in range(args.seed,N):

            biomeVisualFC = getBiomeImage(geometryVisual, 'ff0aad', 'Visual', False, i, num_points)

            mainVisualFC = mainVisualFC.merge(biomeVisualFC);

        exportTable(mainVisualFC,  'Visual NDVI', str(args.seed))
    
# Creates N many images for each biome and exports them
# to the google drive
def main():

    ee.Initialize()

    args = get_args()

    season = "Summer"

    if (args.QuadNum == 'vis'):
        visualExtract()
        exit()

    print ("Thread {} Start_Date: {}  num_imgs: {} seed: {} quad_num: {}                            STARTING".format(args.seed, args.start_date, args.num_imgs, args.seed, args.QuadNum))

    # Create Caatinga geometries
    geometryCAT = geometryPicker(int(args.QuadNum), "Caatinga")
    catFileName = str(args.QuadNum)
    # Create Cerrado geometries
    geometryCER = geometryPicker(int(args.QuadNum), "Cerrado")
    cerFileName = str(args.QuadNum)
    # Create Amazonia geometries
    geometryAMA = geometryPicker(int(args.QuadNum), "Amazonia")


    # Number of total tiles to extract per biome
    N = args.seed+args.num_imgs

    # Only extract 1 tile for each biome at a time
    num_points = 1

    # Extract RGB
    if args.OutputType == "RGB":
        for i in range(args.seed, N):

            biomeCaatinga = getBiomeImage(geometryCAT, '32a83a', 'Caatinga', True, i, num_points)
            biomeCerrado = getBiomeImage(geometryCER, '324ca8', 'Cerrado', True, i, num_points)
            biomeAmazonia = getBiomeImage(geometryAMA, 'FF0000', 'Amazonia', True, i, num_points)

            catFileName = str(args.QuadNum)
            cerFileName = str(args.QuadNum)
            amaFileName = str(args.QuadNum)

            normCaatinga = scale(biomeCaatinga)
            normCerrado  = scale(biomeCerrado)
            normAmazonia = scale(biomeAmazonia)

            catCollection = ee.FeatureCollection([biomeCaatinga]);
            cerCollection = ee.FeatureCollection([biomeCerrado]);
            amaCollection = ee.FeatureCollection([biomeAmazonia]);

            # Append image folder name to directory
            catFileName += "_"+str(args.seed)+"-"+str(i)+"_"
            cerFileName += "_"+str(args.seed)+"-"+str(i)+"_"
            amaFileName += "_"+str(args.seed)+"-"+str(i)+"_"
            # Extract the images
            if (ee.Algorithms.IsEqual(catCollection.size(), 1)):
                export(biomeCaatinga, "Caatinga 2015 Quad "+args.QuadNum, i)
                pass
            if (ee.Algorithms.IsEqual(cerCollection.size(), 1)):
                export(biomeCerrado, "Cerrado 2015 Quad "+args.QuadNum, i)
                pass
            if (ee.Algorithms.IsEqual(amaCollection.size(), 1)):
                export(biomeAmazonia, "Amazonia 2015 Quad "+args.QuadNum, i)
                pass

    # Extract Date
    elif args.OutputType == "Date":

        mainCaatingaFC = getBiomeImage(geometryCAT, '32a83a', 'Caatinga', False, 0, num_points)
        mainCerradoFC = getBiomeImage(geometryCER, '324ca8', 'Cerrado', False, 0, num_points)
        mainAmazoniaFC = getBiomeImage(geometryAMA, 'FF0000', 'Amazonia', False, 0, num_points)

        for i in range(args.seed, N):

            biomeCaatinga = getBiomeImage(geometryCAT, '32a83a', 'Caatinga', False, i, num_points)
            biomeCerrado = getBiomeImage(geometryCER, '324ca8', 'Cerrado', False, i, num_points)
            biomeAmazonia = getBiomeImage(geometryAMA, 'FF0000', 'Amazonia', False, i, num_points)

            catFileName = str(args.QuadNum)
            cerFileName = str(args.QuadNum)
            amaFileName = str(args.QuadNum)

            # Append image folder name to directory
            catFileName += "_"+str(args.seed)+"-"+str(i)+"_"
            cerFileName += "_"+str(args.seed)+"-"+str(i)+"_"
            amaFileName += "_"+str(args.seed)+"-"+str(i)+"_"

            mainCaatingaFC = mainCaatingaFC.merge(biomeCaatinga);
            mainCerradoFC = mainCerradoFC.merge(biomeCerrado);
            mainAmazoniaFC = mainAmazoniaFC.merge(biomeAmazonia);

        # Extract the images
        exportTable(mainCaatingaFC, "Caatinga Dates 2015 Quad"+args.QuadNum, str(args.seed))
        exportTable(mainCerradoFC, "Cerrado Dates 2015 Quad "+args.QuadNum, str(args.seed))
        exportTable(mainAmazoniaFC, "Amazonia Dates 2015 Quad "+args.QuadNum, str(args.seed))

# Returns the quadrant area of the CERRADO biome
def geometryCERQuad1():
    return ee.Geometry.Polygon(
        [[[-46.487368685428514, -8.49532829272219],
          [-46.663149935428514, -16.426734334697056],
          [-42.752017122928514, -16.258055183343203],
          [-42.883853060428514, -3.1636909745123796],
          [-43.191470247928514, -4.960847720145211],
          [-44.377993685428514, -5.179714366301134],
          [-44.729556185428514, -6.534874422166738],
          [-45.740298372928514, -6.447547254236228],
          [-45.740298372928514, -8.538788966980599]]])
def geometryCERQuad2():
    return ee.Geometry.Polygon(
        [[[-46.729067904178514, -16.994938752479133],
          [-46.487368685428514, -8.49532829272219],
          [-47.168521029178514, -8.473596107318164],
          [-46.992739779178514, -10.445814548028476],
          [-48.926333529178514, -10.55383820204988],
          [-48.684634310428514, -11.54582875792169],
          [-50.420474154178514, -11.696484372154377],
          [-50.750063997928514, -14.329458515215412],
          [-55.056704622928514, -14.393316579093161],
          [-55.056704622928514, -17.14197276094385]]])

def geometryCERQuad3():
    return ee.Geometry.Polygon(
        [[[-46.7185276488705, -17.00630357668756],
          [-49.57409687032773, -17.088537260736626],
          [-49.673403841678514, -24.909953578641808],
          [-48.069399935428514, -24.919917455666877],
          [-48.179263216678514, -23.597817861922383],
          [-46.816958529178514, -23.557541165794817],
          [-46.773013216678514, -21.487420271920822],
          [-45.256899935428514, -21.691731433712686],
          [-45.234927279178514, -20.585069070422772],
          [-43.762759310428514, -20.60563741160706],
          [-43.740786654178514, -19.573867813815017],
          [-42.795962435428514, -19.553163620088668],
          [-42.773989779178514, -16.313390498349662],
          [-46.707095247928514, -16.418802198317614]]])

def geometryCERQuad4():
    return ee.Geometry.Polygon(
        [[[-49.597967837028236, -17.06897159948214],
          [-55.069159243278236, -17.152971885605673],
          [-55.135077212028236, -19.260957325609837],
          [-57.617987368278236, -19.21946656521266],
          [-57.508124087028236, -21.995622125184745],
          [-56.365545962028236, -21.648854593195942],
          [-55.310858462028236, -21.995622125184745],
          [-55.266913149528236, -22.889185290685344],
          [-50.674627993278236, -22.889185290685344],
          [-50.586737368278236, -24.91792920500477],
          [-49.707831118278236, -24.91792920500477]]])

def geometryAMAQuad1():
    return ee.Geometry.Polygon(
        [[[-60.3027185661364, 5.014100325118193],
          [-60.86027471848015, 4.543331722153364],
          [-62.1484216911364, 3.9599222312287097],
          [-62.62358038254265, 3.6118680555594675],
          [-64.07926885910514, 3.8804577219302154],
          [-63.90898077316765, 2.7727256667460436],
          [-63.24430792160515, 2.641036859550794],
          [-63.32670538254265, 1.7627881871458588],
          [-65.33821475532949, 0.4007583995167459],
          [-66.34895694282949, 0.53258993489017],
          [-67.66731631782949, 1.9384776827884538],
          [-69.44710147407949, 1.7408258004520911],
          [-69.51301944282949, 1.1697233720670268],
          [-68.80989444282949, 1.103818380543689],
          [-68.83186709907949, 0.3348417925616316],
          [-69.84260928657949, 0.3787862518272467],
          [-68.98567569282949, -0.8077076724877975],
          [-69.62288272407949, -3.4423803058624287],
          [-69.8316229584545, -4.341172512022767],
          [-58.73969979517706, -4.384990469009014],
          [-58.47602792017706, 1.0818497211377653],
          [-59.53071542017706, 1.3454622122270785],
          [-60.14594979517706, 2.158063720693137],
          [-60.14594979517706, 3.1018820422183238],
          [-59.94819588892706, 4.17635537618005],
          [-60.36567635767706, 4.395467347147514]]]);
def geometryAMAQuad2():
    return ee.Geometry.Polygon(
        [[[-58.71262569520882, -4.630063550268986],
          [-46.775849441371555, -4.770693426163744],
          [-46.79760961352516, -3.9663232359733644],
          [-45.15549678895883, -3.7206281470585307],
          [-45.30930538270883, -2.2065868420492074],
          [-50.75852413270882, -0.1856478091329733],
          [-51.35178585145882, 3.5693650949125373],
          [-52.62619991395882, 1.791598895463007],
          [-55.28489132020882, 1.8355221945989084],
          [-58.44895382020882, 1.0887004094351023]]]);

def geometryAMAQuad3():
    return ee.Geometry.Polygon(
        [[[-60.43792876162971, -4.378103583585094],
          [-69.8202529803797, -4.356194723416142],
          [-69.9301162616297, -4.706657738440283],
          [-72.6327529803797, -5.341413354968707],
          [-73.21116542841867, -7.074088858989475],
          [-72.48606777216867, -9.07542565960017],
          [-71.65110683466867, -9.704091644147308],
          [-70.35472011591867, -9.118818215885124],
          [-70.37669277216867, -10.418033932333586],
          [-68.46507167841867, -10.4612517877352],
          [-65.21311855341867, -9.335701399272628],
          [-64.53196620966867, -11.065659073017887],
          [-60.77464199091867, -11.173459861040493],
          [-60.37913417841867, -11.216568983361958],
          [-60.41209316279367, -11.151902895537326],
          [-60.41209316279367, -11.130344331049084]]])

def geometryAMAQuad4():
    return ee.Geometry.Polygon(
        [[[-60.4230117702848, -11.098391076229241],
          [-59.6759414577848, -10.753201743179911],
          [-56.8634414577848, -10.753201743179911],
          [-56.7096328640348, -11.206179744659483],
          [-53.8531875515348, -11.59388380940364],
          [-53.8092422390348, -12.646528234726553],
          [-52.7545547390348, -12.77513323501617],
          [-52.7106094265348, -9.801928793535012],
          [-51.2823867702848, -9.866878128188636],
          [-51.2384414577848, -7.93495786745994],
          [-49.8102188015348, -7.82613214368],
          [-49.7003555202848, -4.7469974105322645],
          [-58.7311172390348, -4.681302437964171],
          [-58.6871719265348, -4.418461719001867],
          [-60.4230117702848, -4.440368747930787]]])

def geometryCATQuad1():
    return ee.Geometry.Polygon(
        [[[-39.17876581754097, -8.111804629213445],
          [-39.09087519254097, -3.6979936973857868],
          [-39.83794550504097, -3.0618983616936823],
          [-42.23296503629097, -3.0618983616936823],
          [-42.16704706754097, -8.111804629213445]]])

def geometryCATQuad2():
    return ee.Geometry.Polygon(
        [[[-39.17876581754097, -8.54662134024153],
          [-35.11382441129097, -8.524892065857923],
          [-34.89409784879097, -7.6111848641617925],
          [-35.13579706754097, -6.673703170101882],
          [-35.55327753629097, -5.275126739097582],
          [-37.09136347379097, -5.100065558991891],
          [-39.00298456754097, -3.6979936973857868]]])
def geometryCATQuad3():
    return ee.Geometry.Polygon(
        [[[-40.07730079868196, -8.116223808224865],
          [-40.07730079868196, -14.76680727124421],
          [-39.136255598588846, -14.757534114226114],
          [-39.59669656252119, -12.440230475556035],
          [-38.36892677524446, -12.492786396717618],
          [-37.19888282993196, -10.652400767363076],
          [-36.29800392368196, -10.090448183023401],
          [-36.21011329868196, -8.833399050831082],
          [-35.15542579868196, -8.876820568186263],
          [-35.11148048618196, -8.507576328631423],
          [-39.16524119090839, -8.546802144104868],
          [-39.20394474078499, -8.125019974381553]]])
def geometryCATQuad4():
    return ee.Geometry.Polygon(
        [[[-42.09699972300834, -15.854916110197014],
          [-40.64680441050834, -15.918316356674904],
          [-40.71272237925834, -14.816609446700122],
          [-40.09748800425834, -14.816609446700122],
          [-40.07551534800834, -8.167213808697552],
          [-42.16291769175834, -8.167213808697552]]])

def getgeometryVisual():
    return ee.Geometry.Polygon(
        [[[-56.52021772397042, -10.996590189089476],
          [-53.901190598627, -11.528700890869507],
          [-51.61391357832542, -13.15770063469457],
          [-39.95146192971439, -13.37502178242121],
          [-38.3610788770815, -4.204793428269779],
          [-57.49372559557569, -4.758397070489507],
          [-57.4123362029629, -6.3903877549743795]]])

def geometryVisual2():
    return ee.Geometry.Polygon(
        [[[-47.1423939100282, -8.523180983808203],
          [-46.9226673475282, -12.968973869407934],
          [-42.7918079725282, -13.097413627759398],
          [-42.8357532850282, -8.957529695943153]]])

def geometryVisual3():
        return ee.Geometry.Polygon(
        [[[-57.49492187500001, -3.1353623493934886],
          [-56.08867187500001, -3.1353623493934886],
          [-56.08867187500001, -1.4668462140274576],
          [-57.84648437500001, -1.4668462140274576]]])

def cross_section():
    return ee.Geometry.Polygon(
        [[[-61.89365800589345, -4.5399394671546665],
          [-51.47861894339345, -16.7800184307609],
          [-54.73057206839345, -19.45279929665714],
          [-50.86338456839345, -22.69175991037202],
          [-36.09775956839345, -7.816239031431601],
          [-39.43760331839345, -4.627548973443063],
          [-47.91904863089345, -12.786822890083796],
          [-57.98252519339345, -2.215044065234536]]])

def cross_section_quad1():
    return ee.Geometry.Polygon(
        [[[-57.924086322308696, -2.272890936940089],
          [-61.791273822308696, -4.466223060913949],
          [-56.473891009808696, -11.210064167568504],
          [-51.859633197308696, -8.699824361262168]]])

def cross_section_quad2():
    return ee.Geometry.Polygon(
        [[[-56.429945697308696, -11.210064167568504],
          [-51.420180072308696, -16.83543660987813],
          [-47.904555072308696, -12.714701746210313],
          [-51.903578509808696, -8.743261530544876]]])

def cross_section_quad3():
    return ee.Geometry.Polygon(
        [[[-54.672133197308696, -19.5073767171173],
          [-50.804945697308696, -22.664079901212293],
          [-43.729750384808696, -15.654118363327436],
          [-43.993422259808696, -9.047178561379742]]])

def cross_section_quad4():
    return ee.Geometry.Polygon(
        [[[-43.597914447308696, -15.823309272123375],
          [-41.664320697308696, -13.954729799491469],
          [-41.884047259808696, -6.9584665602756],
          [-43.949476947308696, -9.133965255455138]]])

def cross_section_quad5():
    return ee.Geometry.Polygon(
        [[[-41.927992572308696, -7.0893130728872045],
          [-41.620375384808696, -13.826749394867543],
          [-36.171156634808696, -7.786517330825019],
          [-39.379164447308696, -4.6852491653173125]]])

def geometryBorderMargin():
    return ee.Geometry.Polygon(
        [[[-55.19662236356561, -14.404946958404846],
          [-50.88998173856561, -14.277219272827114],
          [-50.36263798856561, -11.66520804727291],
          [-48.73666142606561, -11.579119417971635],
          [-49.00033330106561, -10.500835237371732],
          [-46.93490361356561, -10.500835237371732],
          [-47.19857548856561, -8.463739679013091],
          [-45.74838017606561, -8.681011399966275],
          [-45.79232548856561, -6.37213857245612],
          [-44.78158330106561, -6.2411004995890025],
          [-44.56185673856561, -5.060364150352382],
          [-43.24349736356561, -4.97281020566104],
          [-42.80404423856561, -3.044038418759337],
          [-42.67220830106561, -19.448004022279626],
          [-42.05697392606561, -19.448004022279626],
          [-42.18880986356561, -2.8684911686936556],
          [-44.47396611356561, -1.990383519102023],
          [-45.17709111356561, -2.3416942738971054],
          [-45.30892705106561, -3.570502015611671],
          [-46.62728642606561, -3.9213128821171543],
          [-46.80306767606561, -4.71007932684004],
          [-49.74740361356561, -4.929028854573604],
          [-49.70345830106561, -7.941792881548083],
          [-51.10970830106561, -7.941792881548083],
          [-51.24154423856561, -9.895323616178654],
          [-52.60384892606561, -9.808729128530096],
          [-52.77963017606561, -12.739003501720807],
          [-53.74642705106561, -12.824716139788876],
          [-53.87826298856561, -11.579119417971635],
          [-56.64681767606561, -11.277601666007175],
          [-56.95443486356561, -10.803151195310797],
          [-58.27279423856561, -10.93262256539388],
          [-57.61361455106561, -19.116163259223285],
          [-55.24056767606561, -19.199186408966142]]])

def geometryExternalAma():
    return ee.Geometry.Polygon(
        [[[-62.8278615219761, -11.598354813417137],
          [-64.64315837440823, -11.094060987267374],
          [-65.39551947949316, -9.635067331236675],
          [-67.68614456745048, -10.328279302298704],
          [-70.34227350586215, -10.482354238298376],
          [-70.5886535509369, -9.26534779179196],
          [-71.51270960060039, -9.485924677085004],
          [-72.41082048556592, -8.859165159087917],
          [-73.21618024946886, -7.02509960744601],
          [-72.75554218449692, -5.496090316974604],
          [-69.98700530621984, -4.642610362926727],
          [-69.15323111420759, -0.7711822628194441],
          [-69.99592972871083, 0.44803013844904316],
          [-69.16684184358706, 0.40545149433729394],
          [-68.98040118857374, 1.063346052262655],
          [-69.84876354683702, 1.5454526613944732],
          [-67.97062995743454, 1.922732810551719],
          [-66.41191557611569, 0.3928018797320693],
          [-64.83512838395987, 0.7096485800822485],
          [-63.69726927510233, 1.641088329806571],
          [-63.36774945611459, 2.588462914329745],
          [-63.829298928916664, 2.8333340108870586],
          [-64.15826003971712, 4.089225328626829],
          [-62.56374325526195, 3.579252018068477],
          [-60.533397749765925, 4.964556083289729],
          [-60.042400580051385, 5.414887636662277],
          [-59.10785867469882, 7.3506293358829655],
          [-63.89520484002552, 4.640061936377268],
          [-65.65299874764857, 4.201257109451596],
          [-68.11284143647022, 3.9383289510355017],
          [-70.88068706357622, 5.252119112356416],
          [-72.37454435130296, 4.376738212849503],
          [-74.00057428326625, 2.62294730439625],
          [-73.69349661365288, 0.5145939602841381],
          [-74.08940745893075, -1.1112272904363107],
          [-74.96896045341356, -3.0873076133879978],
          [-75.62888530961428, -4.972383483023832],
          [-75.54240296522862, -9.203128511869672],
          [-73.25639927066975, -10.460080739801521],
          [-71.98156567213434, -12.442663431308832],
          [-69.69489815414279, -13.556908937245245],
          [-68.72735677230047, -14.580721918843718],
          [-66.79207021026184, -15.6844100684799],
          [-64.41647873542148, -17.496476794778488],
          [-63.44332203487761, -15.30456562040141],
          [-63.21688430731632, -14.71148237666636]]])

def Ama_To_Inc(cur_point):
    print (cur_point)
    points = [ee.Geometry.Point(-53.11312026833297, -6.624082850909348),
        ee.Geometry.Point(-53.10607126201212, -6.297039549260922),
        ee.Geometry.Point(-53.24935008684166, -6.460776978764056),
        ee.Geometry.Point(-53.49640485365901, -6.256003883690083),
        ee.Geometry.Point(-53.23862300617458, -6.487400081744558),
        ee.Geometry.Point(-53.33491287646091, -6.378228771303218),
        ee.Geometry.Point(-53.56178859885768, -6.491562938843558),
        ee.Geometry.Point(-53.39789144100922, -6.040352940144524),
        ee.Geometry.Point(-53.613343207976584, -6.605977829421326),
        ee.Geometry.Point(-53.533007947428224, -6.3171031099356311)
    ]
    return ee.FeatureCollection([points[cur_point]])

def AmazonToCerrado1():
    return ee.Geometry.Polygon(
        [[[-56.035388775013914, -15.644746533620214],
          [-56.233142681263914, -15.951315126171277],
          [-56.524280376576414, -15.977721650106872],
          [-56.628650493763914, -16.17302129785542],
          [-56.985706157826414, -16.194123257533917],
          [-57.090076275013914, -16.62621234209656],
          [-56.996692485951414, -17.382639957133545],
          [-56.447376079701414, -18.84445368059467],
          [-55.974963970326414, -19.010730236265722],
          [-55.321277446888914, -18.938004607201464],
          [-54.892810650013914, -18.704029558257215],
          [-54.574207134388914, -18.318567881724555],
          [-54.502796001576414, -17.681205696547163],
          [-54.656604595326414, -16.920741017387325],
          [-55.266345806263914, -16.062199087620122]]])


def AmazonToCerrado2():
    return ee.Geometry.Polygon(
        [[[-54.01196353425838, -5.51495556717548],
          [-55.64343326082088, -6.350900267184527],
          [-55.36328189363338, -7.359856734625503],
          [-54.85241763582088, -7.098281025495162],
          [-53.22644107332088, -7.147337966653623],
          [-52.91882388582088, -6.85291856334284],
          [-52.62768619050838, -5.831997971967348],
          [-53.07812564363338, -5.810138618249126],
          [-54.02294986238338, -5.569630410070629]]])

def AreaofDeforestation():
    return ee.Geometry.Polygon(
        [[[-52.837665893635375, -6.653386470812877],
          [-53.2620128174635, -5.806982587778027],
          [-53.553150512776, -5.7632610934905655],
          [-53.805836059651, -6.009149970781929],
          [-53.64928088386975, -6.7420411786553744]]])

def LargeScaleExample():
    return ee.Geometry.Polygon(
        [[[-62.019319827521905, -10.914328014636668],
          [-62.019319827521905, -16.578389129130294],
          [-54.55960303064691, -16.578389129130294],
          [-54.55960303064691, -10.914328014636668]]])

def ProjectPresentation_GridPoints(cur_point):
    print (cur_point)
    points = [
        ee.Geometry.Point(-53.6396, -6.2661),
        ee.Geometry.Point(-53.6396, -6.3002199999999995),
        ee.Geometry.Point(-53.6396, -6.334339999999999),
        ee.Geometry.Point(-53.6396, -6.368459999999999),
        ee.Geometry.Point(-53.6396, -6.402579999999999),
        ee.Geometry.Point(-53.6396, -6.436699999999998),
        ee.Geometry.Point(-53.6396, -6.470819999999998),
        ee.Geometry.Point(-53.6396, -6.504939999999998),
        ee.Geometry.Point(-53.6396, -6.539059999999997),
        ee.Geometry.Point(-53.6396, -6.573179999999997),
        ee.Geometry.Point(-53.6396, -6.607299999999997),
        ee.Geometry.Point(-53.60018, -6.2661),
        ee.Geometry.Point(-53.60018, -6.3002199999999995),
        ee.Geometry.Point(-53.60018, -6.334339999999999),
        ee.Geometry.Point(-53.60018, -6.368459999999999),
        ee.Geometry.Point(-53.60018, -6.402579999999999),
        ee.Geometry.Point(-53.60018, -6.436699999999998),
        ee.Geometry.Point(-53.60018, -6.470819999999998),
        ee.Geometry.Point(-53.60018, -6.504939999999998),
        ee.Geometry.Point(-53.60018, -6.539059999999997),
        ee.Geometry.Point(-53.60018, -6.573179999999997),
        ee.Geometry.Point(-53.60018, -6.607299999999997),
        ee.Geometry.Point(-53.56076, -6.2661),
        ee.Geometry.Point(-53.56076, -6.3002199999999995),
        ee.Geometry.Point(-53.56076, -6.334339999999999),
        ee.Geometry.Point(-53.56076, -6.368459999999999),
        ee.Geometry.Point(-53.56076, -6.402579999999999),
        ee.Geometry.Point(-53.56076, -6.436699999999998),
        ee.Geometry.Point(-53.56076, -6.470819999999998),
        ee.Geometry.Point(-53.56076, -6.504939999999998),
        ee.Geometry.Point(-53.56076, -6.539059999999997),
        ee.Geometry.Point(-53.56076, -6.573179999999997),
        ee.Geometry.Point(-53.56076, -6.607299999999997),
        ee.Geometry.Point(-53.52134, -6.2661),
        ee.Geometry.Point(-53.52134, -6.3002199999999995),
        ee.Geometry.Point(-53.52134, -6.334339999999999),
        ee.Geometry.Point(-53.52134, -6.368459999999999),
        ee.Geometry.Point(-53.52134, -6.402579999999999),
        ee.Geometry.Point(-53.52134, -6.436699999999998),
        ee.Geometry.Point(-53.52134, -6.470819999999998),
        ee.Geometry.Point(-53.52134, -6.504939999999998),
        ee.Geometry.Point(-53.52134, -6.539059999999997),
        ee.Geometry.Point(-53.52134, -6.573179999999997),
        ee.Geometry.Point(-53.52134, -6.607299999999997),
        ee.Geometry.Point(-53.48192, -6.2661),
        ee.Geometry.Point(-53.48192, -6.3002199999999995),
        ee.Geometry.Point(-53.48192, -6.334339999999999),
        ee.Geometry.Point(-53.48192, -6.368459999999999),
        ee.Geometry.Point(-53.48192, -6.402579999999999),
        ee.Geometry.Point(-53.48192, -6.436699999999998),
        ee.Geometry.Point(-53.48192, -6.470819999999998),
        ee.Geometry.Point(-53.48192, -6.504939999999998),
        ee.Geometry.Point(-53.48192, -6.539059999999997),
        ee.Geometry.Point(-53.48192, -6.573179999999997),
        ee.Geometry.Point(-53.48192, -6.607299999999997),
        ee.Geometry.Point(-53.4425, -6.2661),
        ee.Geometry.Point(-53.4425, -6.3002199999999995),
        ee.Geometry.Point(-53.4425, -6.334339999999999),
        ee.Geometry.Point(-53.4425, -6.368459999999999),
        ee.Geometry.Point(-53.4425, -6.402579999999999),
        ee.Geometry.Point(-53.4425, -6.436699999999998),
        ee.Geometry.Point(-53.4425, -6.470819999999998),
        ee.Geometry.Point(-53.4425, -6.504939999999998),
        ee.Geometry.Point(-53.4425, -6.539059999999997),
        ee.Geometry.Point(-53.4425, -6.573179999999997),
        ee.Geometry.Point(-53.4425, -6.607299999999997),
        ee.Geometry.Point(-53.40308, -6.2661),
        ee.Geometry.Point(-53.40308, -6.3002199999999995),
        ee.Geometry.Point(-53.40308, -6.334339999999999),
        ee.Geometry.Point(-53.40308, -6.368459999999999),
        ee.Geometry.Point(-53.40308, -6.402579999999999),
        ee.Geometry.Point(-53.40308, -6.436699999999998),
        ee.Geometry.Point(-53.40308, -6.470819999999998),
        ee.Geometry.Point(-53.40308, -6.504939999999998),
        ee.Geometry.Point(-53.40308, -6.539059999999997),
        ee.Geometry.Point(-53.40308, -6.573179999999997),
        ee.Geometry.Point(-53.40308, -6.607299999999997),
        ee.Geometry.Point(-53.36366, -6.2661),
        ee.Geometry.Point(-53.36366, -6.3002199999999995),
        ee.Geometry.Point(-53.36366, -6.334339999999999),
        ee.Geometry.Point(-53.36366, -6.368459999999999),
        ee.Geometry.Point(-53.36366, -6.402579999999999),
        ee.Geometry.Point(-53.36366, -6.436699999999998),
        ee.Geometry.Point(-53.36366, -6.470819999999998),
        ee.Geometry.Point(-53.36366, -6.504939999999998),
        ee.Geometry.Point(-53.36366, -6.539059999999997),
        ee.Geometry.Point(-53.36366, -6.573179999999997),
        ee.Geometry.Point(-53.36366, -6.607299999999997),
        ee.Geometry.Point(-53.32424, -6.2661),
        ee.Geometry.Point(-53.32424, -6.3002199999999995),
        ee.Geometry.Point(-53.32424, -6.334339999999999),
        ee.Geometry.Point(-53.32424, -6.368459999999999),
        ee.Geometry.Point(-53.32424, -6.402579999999999),
        ee.Geometry.Point(-53.32424, -6.436699999999998),
        ee.Geometry.Point(-53.32424, -6.470819999999998),
        ee.Geometry.Point(-53.32424, -6.504939999999998),
        ee.Geometry.Point(-53.32424, -6.539059999999997),
        ee.Geometry.Point(-53.32424, -6.573179999999997),
        ee.Geometry.Point(-53.32424, -6.607299999999997),
        ee.Geometry.Point(-53.28482, -6.2661),
        ee.Geometry.Point(-53.28482, -6.3002199999999995),
        ee.Geometry.Point(-53.28482, -6.334339999999999),
        ee.Geometry.Point(-53.28482, -6.368459999999999),
        ee.Geometry.Point(-53.28482, -6.402579999999999),
        ee.Geometry.Point(-53.28482, -6.436699999999998),
        ee.Geometry.Point(-53.28482, -6.470819999999998),
        ee.Geometry.Point(-53.28482, -6.504939999999998),
        ee.Geometry.Point(-53.28482, -6.539059999999997),
        ee.Geometry.Point(-53.28482, -6.573179999999997),
        ee.Geometry.Point(-53.28482, -6.607299999999997),
        ee.Geometry.Point(-53.245400000000004, -6.2661),
        ee.Geometry.Point(-53.245400000000004, -6.3002199999999995),
        ee.Geometry.Point(-53.245400000000004, -6.334339999999999),
        ee.Geometry.Point(-53.245400000000004, -6.368459999999999),
        ee.Geometry.Point(-53.245400000000004, -6.402579999999999),
        ee.Geometry.Point(-53.245400000000004, -6.436699999999998),
        ee.Geometry.Point(-53.245400000000004, -6.470819999999998),
        ee.Geometry.Point(-53.245400000000004, -6.504939999999998),
        ee.Geometry.Point(-53.245400000000004, -6.539059999999997),
        ee.Geometry.Point(-53.245400000000004, -6.573179999999997),
        ee.Geometry.Point(-53.245400000000004, -6.607299999999997)
    ]
    return ee.FeatureCollection([points[cur_point]])

if __name__ == "__main__":

    #ee.Authenticate()

    ee.Initialize()

    main()
