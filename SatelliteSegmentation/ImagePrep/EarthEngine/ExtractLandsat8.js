// GENERATES A NORMALISED IMAGE FOR A GIVEN NUMBER
// OF PATHS AND ROWS
// PATH = horizontal (longitude)
// ROW = vertical (latitude)

Map.setCenter(-60.86524, -7.33678, 6);

// Make a list of features using a for loop
var features = [];
var images = [];
// Get Landsat image for those co-ordinates

var row = 60
var path = 231

// Takes in a row and path and returns the normalised
// version of the image with the least amount of cloud cover 
function addImage(path, row)
{
  try
  {
    // Get Landsat image for those co-ordinates
    var dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1')
                      .filterDate('1999-01-08', '2000-01-01')
                      .filter(ee.Filter.eq('WRS_PATH', path))
                      .filter(ee.Filter.eq('WRS_ROW', row))
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
    
    
    Map.addLayer(unitScale.select(['B3', 'B2', 'B1']), {min: 0, max: 1}, 'unitscaled')
    
  }
  catch(err)
  {
    print ("_____ERROR OCCURED_____")
    print (err)
  }
}


// Loop through and add each image to the map
for (var i = 0; i < 6; i++)
{
  for (var j = 0; j < 6; j++)
  {
    // temp_path catches the wrap around from 233 -> 1
    var temp_path = ((path+i) % 234)
    print ("TEMP " + temp_path)
    if (temp_path < 100){
       temp_path ++
    }
    addImage(temp_path, row+j)
  }
}

