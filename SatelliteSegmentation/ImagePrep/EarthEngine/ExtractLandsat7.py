from geetools import batch
import ee
import time

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

  image = ee.ImageCollection('LANDSAT/LE07/C01/T1').filterDate('2001-10-01', '2002-03-31').filterBounds(geo).sort('CLOUD_COVER', True).first(); # September -> April

  image = image.clip(geo)

  return ee.Image(image)


# Returns an image with only the Blue, Green and Red bands
def getBGR (image):
  return image.select(['B1', 'B2', 'B3'])


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
def getBiomeImage(biomeGeometry, geometry_colour, biome, BGR_images, seed, num_points):

    # Add geomtry area to Google Earth Engine
    #Map.addLayer(biomeGeometry, {color: geometry_colour}, biome);

    # Get Feature Collection of random points within biome geometry
    random_points_FC = ee.FeatureCollection.randomPoints(biomeGeometry, num_points, seed, 10)

    #Map.addLayer(random_points_FC);

    geometries = random_points_FC.map(getCoord)

    images = ee.ImageCollection(geometries.map(getImages, True))

    # If BGR_images flag is false then extract NDVI data from each image tile
    if (not BGR_images):

        withNDVI = images.map(addNDVI);

        """ TO ADD EACH NDVI IMAGE TO THE MAP
        ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']}

        Map.addLayer(withNDVI.select('NDVI'),
                 {min: -1, max: 1, palette: ['blue', 'white', 'green']},
                 'NDVI classification');
        """

        meanNDVIs = ee.FeatureCollection(withNDVI.map(meanNDVI))

        return meanNDVIs

    else:

        imagesBGR = images.map(getBGR);

        return imagesBGR.first();


# Export the given image to the Google drive in the
# folder named folder with the given file name
def export(image, folder, fileName):

    task = ee.batch.Export.image.toDrive(image=image,
                                     folder=folder,
                                     fileNamePrefix=str(fileName))
    task.start()

    status = task.status()['state']

    while (status != 'COMPLETED' and status != 'FAILED'):

        status = task.status()['state']

        print ("BIOME: " + folder + "\tIMAGE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))

        time.sleep(10)


# Export the given image to the Google drive in the
# folder named folder with the given file name
def exportTable(table, folder, fileName):

    task = ee.batch.Export.table.toDrive(collection=table,
                                        folder=folder,
                                        fileNamePrefix=str(fileName),
                                        fileFormat='CSV')
    task.start()

    status = task.status()['state']

    while (status != 'COMPLETED' and status != 'FAILED'):

        status = task.status()['state']

        print ("BIOME: " + folder + "\tTABLE: "+ str(fileName) + "\tSTATUS: " + str(status) + '\tTIME: ' + str(time.strftime("%H:%M:%S", time.localtime())))

        time.sleep(10)


# Creates N many images for each biome and exports them
# to the google drive
def main():

    season = "Summer"
    imgType = "RGB"

    # Create Amazonia train and test geometries
    # Train set
    geometryTrainAMA = geometryAMAQuad1().union(geometryAMAQuad2());
    geometryTrainAMA = geometryTrainAMA.union(geometryAMAQuad3());

    # Test set
    geometryTestAMA = geometryAMAQuad4();

    # Create Cerrado train and test geometries
    # Train set
    geometryTrainCER = geometryCERQuad1().union(geometryCERQuad2());
    geometryTrainCER = geometryTrainCER.union(geometryCERQuad3());

    # Test set
    geometryTestCER = geometryCERQuad4();

    # Create Caatinga train and test geometries
    # Train set
    geometryTrainCAT = geometryCATQuad1().union(geometryCATQuad2());
    geometryTrainCAT = geometryTrainCAT.union(geometryCATQuad3());

    # Test set
    geometryTestCAT = geometryCATQuad4();

    # Extract RGB
    if imgType == "RGB":

        # Number of total tiles to extract per biome
        N = 2

        # Only extract 1 tile for each biome at a time
        num_points = 1

        for i in range(0,N):


            biomeCaatinga = getBiomeImage(geometryTrainCAT, '32a83a', 'Caatinga', True, i, num_points)
            biomeCerrado = getBiomeImage(geometryTrainCER, '324ca8', 'Cerrado', True, i, num_points)
            biomeAmazonia = getBiomeImage(geometryTrainAMA, 'FF0000', 'Amazonia', True, i, num_points)

            normCaatinga = scale(biomeCaatinga)
            normCerrado  = scale(biomeCerrado)
            normAmazonia = scale(biomeAmazonia)

            #help(batch.Export.imagecollection.toDrive)

            export(normCaatinga, 'CaatingaT', i)
            export(normCerrado, 'CerradoT', i)
            export(normAmazonia, 'AmazoniaT', i)

    # Extract NDVI
    else:

        # Total number of tiles that will be extracted
        N = 950

        # Extract 1 tile for each biome at a time
        num_points = 1

        mainCaatingaFC = getBiomeImage(geometryTrainCAT, '32a83a', 'Caatinga', False, 0, num_points)
        mainCerradoFC = getBiomeImage(geometryTrainCER, '324ca8', 'Cerrado', False, 0, num_points)
        mainAmazoniaFC = getBiomeImage(geometryTrainAMA, 'FF0000', 'Amazonia', False, 0, num_points)

        for i in range(0,N):

            biomeCaatingaFC = getBiomeImage(geometryTrainCAT, '32a83a', 'Caatinga', False, i, num_points)
            biomeCerradoFC = getBiomeImage(geometryTrainCER, '324ca8', 'Cerrado', False, i, num_points)
            biomeAmazoniaFC = getBiomeImage(geometryTrainAMA, 'FF0000', 'Amazonia', False, i, num_points)

            mainCaatingaFC = mainCaatingaFC.merge(biomeCaatingaFC);
            mainCerradoFC = mainCerradoFC.merge(biomeCerradoFC);
            mainAmazoniaFC = mainAmazoniaFC.merge(biomeAmazoniaFC);


        exportTable(mainCaatingaFC,  'NDVI_Scores', 'CaatingaNDVI'+str(N)+season)
        exportTable(mainCerradoFC,   'NDVI_Scores', 'CerradoNDVI'+str(N)+season)
        exportTable(mainAmazoniaFC,  'NDVI_Scores', 'AmazoniaNDVI'+str(N)+season)

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

if __name__ == "__main__":

    #ee.Authenticate()

    ee.Initialize()

    main()
