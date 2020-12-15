from geetools import batch
import ee
import time

# Given a geometry point this returns a 1km (polygon) tile
# with the given point being the bottom left corner
def getCoord(feature):

  x1 = feature.geometry().coordinates().get(0)
  y1 = feature.geometry().coordinates().get(1)


  x2 = ee.Number(0.01).add(x1)
  y2 = ee.Number(0.01).add(y1)

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
def ExtractBiome(biomeGeometry, geometry_colour, biome, BGR_images, i):

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

def clip(image):
    geo = ee.Geometry.Polygon([
      [[-69.3, -10], [-69.3, -9.9], [-69.2, -9.9], [-69.2, -10]]
    ])
    return image.clip(geo)




def main():

    # Geometry for Caatinga biome
    geometryCAT = ee.Geometry.Polygon([[
    [-43, -16],
    [-35.5, -9],
    [-43, -3],
    ]]);

    # Geometry for Cerrado biome
    geometryCER = ee.Geometry.Polygon([[
    [-58, -15],
    [-52, -17],
    [-58, -20],
    [-42, -20],
    [-42, -3],
    [-52, -13.5],
    ]]);

    # Geometry for Amazonia biome
    geometryAMA = ee.Geometry.Polygon([[
    [-68, -9],
    [-66, -9],
    [-58, -15],
    [-50, -9],
    [-49, -1],
    [-68, -1],
    [-68, -9]
    ]]);

    for i in range(0,10):
        biomeCaatinga = ExtractBiome(geometryCAT, '32a83a', 'Caatinga', True, i)

        #help(batch.Export.imagecollection.toDrive)

        geometry = ee.Geometry.Polygon([
          [[-69.3, -10], [-69.3, -9.9], [-69.2, -9.9], [-69.2, -10]]
        ])

        setLandsatImage = ee.ImageCollection('LANDSAT/LE07/C01/T1').filterBounds(geometry).filterDate('2001-04-01', '2001-05-30').sort('CLOUD_COVER', False)

        setLandsatImage = setLandsatImage.select('B3', 'B2', 'B1')
        #setLandsatImage = setLandsatImage.map(clip)

        #filtered = setLandsatImage.filter(setLandsatImage.first().not())

        #print (len(setLandsatImage))
        #print (len(filtered))
        #biomeImage = biomeCaatinga.first()

        #tasks = batch.Export.imagecollection.toDrive(biomeCaatinga, 'PythonTest', scale= 30)

        #tasks[0].start()
        #tasks = batch.Export.imagecollection.toDrive(setLandsatImage, 'PythonTest')

        task = ee.batch.Export.image.toDrive(image=biomeCaatinga,  # an ee.Image object.
                                         folder='PythonTest',
                                         fileNamePrefix=str(i))
        task.start()
        status = task.status()['state']
        print ("IMAGE: "+ str(i) + "\tSTATUS: " + str(status))
        while (status != 'COMPLETED' and status != 'FAILED'):
            status = task.status()['state']
            print ("IMAGE: "+ str(i) + "\tSTATUS: " + str(status))
            time.sleep(5)

        # Batch Export
        #tasks = batch.Export.imagecollection.toDrive(setLandsat, 'PythonTest', region=geo)

if __name__ == "__main__":

    #ee.Authenticate()

    ee.Initialize()

    main()
