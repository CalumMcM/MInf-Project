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

  //print ('NDVI Mean: ', parseInt(JSON.parse(ndviMean)))

  caatingaNDVIs.push(ndviMean)

  var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};

  // Blue = low NDVI
  // Green = high NDVI.
  var blended = result.visualize(ndviParams)

  return blended;
}
// Takes in a row and path and returns the normalised
// version of the image with the least amount of cloud cover
function processImage(geometry, biome, tileCount, type, scale, maxPixels)
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

  Map.addLayer(geometryAMA);

  var tileCount = 1;

  // Make a list of features using a for loop
  var features = [];
  var images = [];
  // Get Landsat image for those co-ordinates

  Map.setCenter(-54, -2, 10);

  var x1 = -54
  var x2 = -53.99


  for (var i = -6; i < 4; i++) // columns
  {
    var y1 = -2
    var y2 = -1.99

    for (var j = 2; j < 6; j++) // path
    {
      /*
      P4: [x1, y2] P3: [x2, y2]
      P1: [x1, y1] P2: [x2, y1]
      */
      var sliding_geometry = ee.Geometry.Polygon([[
      [x1, y1],  // P1
      [x2, y1],  // P2
      [x2, y2],  // P3
      [x1, y2],  // P4
      ]]);

      Map.addLayer(sliding_geometry, {color: 'FF0000'});

      var image = processImage(sliding_geometry, "AmazoniaNDVI", tileCount);

      amazoniaCollection.push(image)

      tileCount ++;
      y1 -= 1.25;
      y2 -= 1.25;

    }
    x1 -= 1.35;
    x2 -= 1.35;
  }

  return (ee.ImageCollection(amazoniaCollection));
}

function ExtractCaatinga(){

  var caatingaCollection = [];

  // Caatinga
  var geometryCAT = ee.Geometry.Polygon([[
    [-43, -16],  // BL point
    [-35.5, -9],  // TL point
    [-43, -3],  // TR point
    ]]);

  Map.addLayer(geometryCAT);

  var tileCount = 1;

  // Make a list of features using a for loop
  var features = [];
  var images = [];
  // Get Landsat image for those co-ordinates

  Map.setCenter(-38, -9, 6);

  var x1 = -42
  var x2 = -41.99

  var num_rows = 10;
  for (var i = 0; i < 5; i++) // columns
  {

    var y1 = -3.8 - (i*1.25);
    var y2 = -3.79 - (i*1.25);

    for (var j = 0; j < num_rows; j++) // rows
    {
      /*
      P4: [x1, y2] P3: [x2, y2]
      P1: [x1, y1] P2: [x2, y1]
      */
      var sliding_geometry = ee.Geometry.Polygon([[
      [x1, y1],  // P1
      [x2, y1],  // P2
      [x2, y2],  // P3
      [x1, y2],  // P4
      ]]);

      Map.addLayer(sliding_geometry, {color: 'FF0000'});

      var image = processImage(sliding_geometry, "CaatingaNDVI", tileCount);

      caatingaCollection.push(image);

      tileCount ++;
      y1 -= 1.25;
      y2 -= 1.25;

    }
    x1 += 0.5;
    x2 += 0.5;

    num_rows -= 2;
  }

  return (ee.ImageCollection(caatingaCollection));
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

  Map.addLayer(geometryCER);

  var tileCount = 1;

  // Make a list of features using a for loop
  var features = [];
  var images = [];
  // Get Landsat image for those co-ordinates

  Map.setCenter(-49, -14, 5);

  var x1 = -44
  var x2 = -43.99

  var num_rows = 10;
  for (var i = 0; i < 5; i++) // columns
  {

    var y1 = -7.5 - (i*1.25);
    var y2 = -7.49 - (i*1.25);

    for (var j = 0; j < num_rows; j++) // rows
    {
      /*
      P4: [x1, y2] P3: [x2, y2]
      P1: [x1, y1] P2: [x2, y1]
      */
      var sliding_geometry = ee.Geometry.Polygon([[
      [x1, y1],  // P1
      [x2, y1],  // P2
      [x2, y2],  // P3
      [x1, y2],  // P4
      ]]);

      Map.addLayer(sliding_geometry, {color: 'FF0000'});

      var image = processImage(sliding_geometry, "CerradoNDVI", tileCount);

      cerradoCollection.push(image);

      tileCount ++;
      y1 -= 1.25;
      y2 -= 1.25;

    }
    x1 -= 0.5;
    x2 -= 0.5;

    num_rows -= 2

  }

  return (ee.ImageCollection(cerradoCollection));
}

var caatingaNDVIs = []

var caatingaBatch = ExtractCaatinga();

var myFeatures = ee.FeatureCollection(caatingaBatch.map(function(el){
  el = ee.List(el); // cast every element of the list
  var geom = null;
  return ee.Feature(geom, {
    'NDVI': ee.Number(el.get(0))
  });
}));

var caatingaNDVIsFC = ee.FeatureCollection([ee.Feature(null, {'NDVI': caatingaNDVIs})])


// (caatingaNDVIs)

//print (caatingaNDVIsFC)

Map.addLayer(caatingaBatch)

Export.table.toDrive({
  collection: caatingaNDVIsFC,
  description: 'Caatinga',
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
