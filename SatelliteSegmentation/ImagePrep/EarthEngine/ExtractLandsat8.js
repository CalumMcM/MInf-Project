// GENERATES A NORMALISED IMAGE FOR A GIVEN NUMBER
// OF PATHS AND ROWS
// PATH = horizontal (longitude)
// ROW = vertical (latitude)

// Takes in an image and calculates the NDVI values for that image
// then overlays a palette so that green = high NDVI and blue = low NDVI
function ndvi(image){

  var nir = image.select(['B4']);
  var red = image.select(['B3']);

  // NDVI = (NIR-RED)/(NIR+RED)
  var result = image.normalizedDifference(['B4', 'B3']).rename('NDVI');

  var ndviMean = result.reduceRegion(ee.Reducer.mean()).get('NDVI')


  //cerradoNDVIs.push(ndviMean) //UNCOMMENTED FOR MAPPING

  var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};

  // Blue = low NDVI
  // Green = high NDVI.
  var blended = result.visualize(ndviParams)

  return blended;
}
// Takes in a row and path and returns the normalised
// version of the image with the least amount of cloud cover
function processImage(geometry, biome, type, scale, maxPixels)
{
  try
  {
    type = type || "float";
    scale = scale || 1000;
    maxPixels = maxPixels || 1e10;

    // Get Landsat image for those co-ordinates
    var dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1')
                      .filterDate('1999-04-01', '2001-09-30') // April -> September
                      .filterBounds(geometry)
                      .sort('CLOUD_COVER'); // Sort by cloud cover (lowest to highest)


    // Extract first image from collection
    var image = dataset.first();


    // calculate the min and max value of an image
    var minMax = image.reduceRegion({
      reducer: ee.Reducer.minMax(),
      geometry: image.geometry(),
      scale: 30,
      maxPixels: 10e9,
      // tileScale: 16
    });

    // use unit scale to normalize the pixel values
    var unitScale = ee.ImageCollection.fromImages(
      image.bandNames().map(function(name){
        name = ee.String(name);
        var band = image.select(name);
        return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
                    // eventually multiply by 100 to get range 0-100
                    //.multiply(100);
    })).toBands().rename(image.bandNames());

    unitScale = unitScale.select(['B3', 'B2', 'B1']).clip(geometry)

    var imgtype = {"float":unitScale.toFloat(),
                     "byte":unitScale.toByte(),
                     "int":unitScale.toInt(),
                     "double":unitScale.toDouble()
                    }

    image = image.clip(geometry);

    // For processing an NDVI image
    image = ndvi(image);

    //imageRGB = image.select(['B3', 'B2', 'B1'])
    /*
    Export.image.toDrive({
      image: image,
      description: tileCount.toString(),
      folder: biome,
      scale: 30,
      region: geometry
    });
    */

    //Map.addLayer(image)

    print (tileCount);

    tileCount ++;

    return (image)

  }
  catch(err)
  {
    print ("_____ERROR OCCURED_____");
    print (String(err));
  }
}

function ExtractAmazonia(){

  var amazoniaCollection = [];

  var geometryAMA = ee.Geometry.Polygon([[
    [-68, -9],  // BL point
    [-66, -9],
    [-58, -15],
    [-50, -9],  // BR point
    [-49, -1],  // TR point
    [-68, -1],  // TL point
    [-68, -9]   // BL point
    ]]);

  var random_points = 5;

  var random_points_FC = ee.FeatureCollection.randomPoints(geometryCAT,random_points, 0, 10);

  var geometries = random_points_FC.map(getCoord)

  var images = ee.ImageCollection(geometries.map(getImages, true))

  var withNDVI = images.map(addNDVI);

  // Visualise:
  /*
  var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};

  var blended = withNDVI.first().select('NDVI').visualize(ndviParams)

  Map.centerObject(withNDVI.first(), 10)
  Map.addLayer(blended)
  */

  var meanNDVIs = ee.FeatureCollection(withNDVI.map(meanNDVI));

  print ("MEAN NDVI'S:")
  print (meanNDVIs)

  return meanNDVIs
}


function ExtractCaatinga(){

  var caatingaCollection = [];

  // Caatinga
  var geometryCAT = ee.Geometry.Polygon([[
    [-43, -16],  // BL point
    [-35.5, -9],  // TL point
    [-43, -3],  // TR point
    ]]);

  var random_points = 5;

  var random_points_FC = ee.FeatureCollection.randomPoints(geometryCAT,random_points, 0, 10);

  var geometries = random_points_FC.map(getCoord)

  var images = ee.ImageCollection(geometries.map(getImages, true))

  var withNDVI = images.map(addNDVI);

  // Visualise:
  /*
  var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};

  var blended = withNDVI.first().select('NDVI').visualize(ndviParams)

  Map.centerObject(withNDVI.first(), 10)
  Map.addLayer(blended)
  */

  var meanNDVIs = ee.FeatureCollection(withNDVI.map(meanNDVI));

  print ("MEAN NDVI'S:")
  print (meanNDVIs)

  return meanNDVIs
}


function ExtractCerrado(){

  var cerradoCollection = [];

  //Cerrado
  var geometryCER = ee.Geometry.Polygon([[
    [-58, -15],  //  point 1
    [-52, -17],  //  point2
    [-58, -20],  //  point2
    [-42, -20],
    [-42, -3],  // TR point
    [-52, -13.5],  // TL point
    ]]);

  var random_points = 1000;

  var random_points_FC = ee.FeatureCollection.randomPoints(geometryCER,random_points, 0, 10);

  var geometries = random_points_FC.map(getCoord)

  var images = ee.ImageCollection(geometries.map(getImages, true))

  var withNDVI = images.map(addNDVI);

  // Visualise:
  /*
  var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};

  var blended = withNDVI.first().select('NDVI').visualize(ndviParams)

  Map.centerObject(withNDVI.first(), 10)
  Map.addLayer(blended)
  */

  var meanNDVIs = ee.FeatureCollection(withNDVI.map(meanNDVI));

  print ("MEAN NDVI'S:")
  print (meanNDVIs)

  return meanNDVIs
}

// Given a geometry point this returns a 1km (polygon) tile
// with the given point being the bottom left corner
var getCoord = function(feature, list) {
  var x1 = feature.geometry().coordinates().get(0)
  var y1 = feature.geometry().coordinates().get(1)


  var x2 = ee.Number(0.01).add(x1)
  var y2 = ee.Number(0.01).add(y1)

  var sliding_geometry = ee.Geometry.Polygon([[
  [x1, y1],  // P1
  [x2, y1],  // P2
  [x2, y2],  // P3
  [x1, y2],  // P4
  ]]);

  return ee.Feature(sliding_geometry)
};

// Given a polygon geometry this returns a clipped
// Landsat 7 image of that geometry
var getImages = function(feature) {
  var geo = ee.Geometry.Polygon(feature.geometry().coordinates().get(0))
  var centroid = feature.geometry().centroid();

  var image = ee.ImageCollection('LANDSAT/LE07/C01/T1')
                    .filterDate('1999-04-01', '2001-09-30') // April -> September
                    .filterBounds(geo)
                    .sort('CLOUD_COVER').first();

  image = image.clip(geo)

  return ee.Image(image);
}

// Extract the red and near-infrared bands from an image,
// computes the NDVI band and appends it to the image
var addNDVI = function(image) {
  var nir = image.select(['B4']);
  var red = image.select(['B3']);

  // NDVI = (NIR-RED)/(NIR+RED)
  var result = image.normalizedDifference(['B4', 'B3']).rename('NDVI');
  // Blue = low NDVI  &&  Green = high NDVI
  return image.addBands(result);
};

// Returns the mean NDVI for an image
var meanNDVI = function(image) {
    var ndvi = image.select('NDVI')

    var ndviMean = ndvi.reduceRegion(ee.Reducer.mean()).get('NDVI')

    return ee.Feature(null, {'NDVI': ndviMean})
}



// Extract mean NDVIs for cerrado
var meanNDVIs = ExtractCerrado();

Export.table.toDrive({
  collection: meanNDVIs,
  description: 'CaatingaBatch1000',
  fileFormat: 'CSV',
  folder: 'NDVI_Scores'
});

/*
// Export a collection of images
var batch = require('users/fitoprincipe/geetools:batch')
batch.Download.ImageCollection.toDrive(caatingaBatch, 'caatingaBatch',
                {scale: 100,
                 type: 'float'})
*/
