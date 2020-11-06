// GENERATES A NORMALISED IMAGE FOR A GIVEN NUMBER
// OF PATHS AND ROWS
// PATH = horizontal (longitude)
// ROW = vertical (latitude)

var row = 60
var path = 231

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

// Make a list of features using a for loop
var features = [];
var images = [];
// Get Landsat image for those co-ordinates

// Takes in a row and path and returns the normalised
// version of the image with the least amount of cloud cover
function addImage(geometry, biome, type, scale, maxPixels)
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

    //image = image.clip(geometry)

    print (image)
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

    var id = unitScale.id().getInfo();

    Export.image.toDrive({
      image: unitScale,
      description: id,
      folder: biome,
      scale: 30,
      region: geometry
    });

    Map.addLayer(unitScale, {min: 0, max: 1}, 'unitscaled')

  }
  catch(err)
  {
    print ("_____ERROR OCCURED_____");
    print (String(err));
  }
}



Map.setCenter(-54, -2, 10);

var x1 = -54
var x2 = -53.99


for (var i = -6; i < 4; i++) // rows
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

    addImage(sliding_geometry, "Amazonia");

    y1 -= 1.25;
    y2 -= 1.25;

    break;
  }
  x1 -= 1.35;
  x2 -= 1.35
  break;
}
