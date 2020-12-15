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

  image = ee.ImageCollection('LANDSAT/LE07/C01/T1').filterDate('2000-09-30', '2001-04-01').filterBounds(geo).sort('CLOUD_COVER', True).first(); # September -> April

  image = image.clip(geo)

  return ee.Image(image)

def getBGR (image):
  return image.select(['B1', 'B2', 'B3'])

# Given a biomes geometry this function will extract either the NDVI
# data or the BGR images - based on the flag BGR_images - for each
# tile in the image
def getBiomeImage(biomeGeometry, geometry_colour, biome, BGR_images, i):

    # Add geomtry area to Google Earth Engine
    #Map.addLayer(biomeGeometry, {color: geometry_colour}, biome);

    random_points = 1;

    # Get Feature Collection of random points within biome geometry
    random_points_FC = ee.FeatureCollection.randomPoints(biomeGeometry, random_points, i, 10)

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

        print ("BIOME: " + folder + "\tIMAGE: "+ str(fileName) + "\tSTATUS: " + str(status))

        time.sleep(10)


# Creates N many images for each biome and exports them
# to the google drive
def main():

    N = 2

    for i in range(0,N):

        biomeCaatinga = getBiomeImage(geometryCAT(), '32a83a', 'Caatinga', True, i)
        biomeCerrado = getBiomeImage(geometryCER(), '324ca8', 'Cerrado', True, i)
        biomeAmazonia = getBiomeImage(geometryAMA(), 'FF0000', 'Amazonia', True, i)

        #help(batch.Export.imagecollection.toDrive)

        export(biomeCaatinga, 'Caatinga', i)
        export(biomeCerrado, 'Cerrado', i)
        export(biomeAmazonia, 'Caatinga', i)

# Geometry for Caatinga biome
def geometryCAT():
    return ee.Geometry.Polygon(
        [[[-42.05826377771528, -15.854757940174075],
          [-40.65201377771528, -15.89702702328183],
          [-40.69595909021528, -14.79520738192892],
          [-39.11392784021528, -14.752714929835435],
          [-39.59732627771528, -12.446375440802163],
          [-38.36685752771528, -12.489284406524968],
          [-37.18033409021528, -10.702856490573197],
          [-36.30142784021528, -10.09773308477402],
          [-36.21353721521528, -8.884131303332929],
          [-35.19180869959028, -8.884131303332929],
          [-34.92813682459028, -7.666498211221274],
          [-35.12589073084028, -6.598185200158663],
          [-35.56534385584028, -5.275999997841332],
          [-37.05948448084028, -5.133766672625146],
          [-37.76260948084028, -4.663081322808359],
          [-39.82803916834028, -3.095685481006172],
          [-42.21207237146528, -3.084715128060662]]]);

# Geometry for Cerrado biome
def geometryCER():
    return ee.Geometry.Polygon(
        [[[-43.253135518651106, -5.021111684508034],
          [-44.439658956151106, -5.371228868634882],
          [-44.835166768651106, -6.507654397388285],
          [-45.714073018651106, -6.507654397388285],
          [-45.758018331151106, -8.555153211290122],
          [-47.164268331151106, -8.511694397794136],
          [-47.032432393651106, -10.418870149845777],
          [-48.922080831151106, -10.591704899583325],
          [-48.746299581151106, -11.54051384001206],
          [-50.416221456151106, -11.712688302848527],
          [-50.767783956151106, -14.32420258612824],
          [-55.030479268651106, -14.409344617736894],
          [-55.030479268651106, -16.86362641591061],
          [-55.118369893651106, -19.244969065102573],
          [-57.579307393651106, -19.20347426484414],
          [-57.491416768651106, -21.95954148057466],
          [-56.348838643651106, -21.673957079992284],
          [-55.338096456151106, -22.000292631391275],
          [-55.294151143651106, -22.893825804598848],
          [-50.690879659276106, -22.863459691178388],
          [-50.602989034276106, -24.912533777822745],
          [-48.087119893651106, -24.892604026445238],
          [-48.12271690271528, -23.58325374834716],
          [-46.80435752771528, -23.58325374834716],
          [-46.76041221521528, -21.51352125209363],
          [-45.22232627771528, -21.67696382469707],
          [-45.22232627771528, -20.611328719824495],
          [-43.77213096521528, -20.611328719824495],
          [-43.77213096521528, -19.5795967364699],
          [-42.80533409021528, -19.538187162485723],
          [-42.89322471521528, -3.1395657527972243]]]);

# Geometry for Amazonia biome
def geometryAMA():
    return ee.Geometry.Polygon(
        [[[-64.5666120811511, -11.066493499251933],
          [-60.523643331151106, -11.15273707517923],
          [-59.644737081151106, -10.807611138266022],
          [-56.832237081151106, -10.807611138266022],
          [-56.700401143651106, -11.238955065551924],
          [-54.854698018651106, -11.497453648969081],
          [-53.843955831151106, -11.583567424204649],
          [-53.800010518651106, -12.657692002652746],
          [-52.745323018651106, -12.743432145058614],
          [-52.701377706151106, -9.81320317254805],
          [-51.295127706151106, -9.856502671467135],
          [-51.251182393651106, -7.902764098698903],
          [-49.800987081151106, -7.859233851221025],
          [-49.757041768651106, -4.889768555297202],
          [-46.812705831151106, -4.802192113984373],
          [-46.812705831151106, -3.9696837263465303],
          [-45.186729268651106, -3.7504557622598402],
          [-45.274619893651106, -2.2144994155120714],
          [-50.767783956151106, -0.19356623187742772],
          [-51.339073018651106, 3.583391914910464],
          [-52.613487081151106, 1.7836842831804605],
          [-55.250205831151106, 1.827607774007858],
          [-56.876182393651106, 1.4322607731300514],
          [-58.502158956151106, 1.080783362408126],
          [-59.600791768651106, 1.3443959573873827],
          [-60.128135518651106, 2.162163647164861],
          [-60.128135518651106, 3.040155691349771],
          [-59.864463643651106, 4.136617699689834],
          [-60.347862081151106, 4.399558110676168],
          [-60.303916768651106, 5.012715322006051],
          [-60.875205831151106, 4.530993857471111],
          [-62.149619893651106, 3.96127525211753],
          [-62.633018331151106, 3.6104804939931237],
          [-64.0832136436511, 3.8735899888115704],
          [-63.907432393651106, 2.776823698318261],
          [-63.248252706151106, 2.6451353368034516],
          [-63.336143331151106, 1.7668890829160742],
          [-65.3576277061511, 0.4048611400838265],
          [-66.3683698936511, 0.5366925982296809],
          [-67.6427839561511, 1.9425781718249597],
          [-69.4884870811511, 1.6790382027792001],
          [-69.4884870811511, 1.1518572250745795],
          [-68.7853620811511, 1.0639830399459438],
          [-68.8293073936511, 0.31697226743362433],
          [-69.8400495811511, 0.3609168099149286],
          [-69.0050886436511, -0.825575671604701],
          [-69.8839948936511, -4.687553688980184],
          [-72.6086042686511, -5.387958122859675],
          [-73.2238386436511, -7.091607643909578],
          [-72.4328230186511, -9.136248182452052],
          [-71.6418073936511, -9.699834262497767],
          [-70.3673933311511, -9.136248182452052],
          [-70.3673933311511, -10.478611364805767],
          [-68.4337995811511, -10.478611364805767],
          [-65.2257917686511, -9.353120640758235]]]);
if __name__ == "__main__":

    #ee.Authenticate()

    ee.Initialize()

    main()
