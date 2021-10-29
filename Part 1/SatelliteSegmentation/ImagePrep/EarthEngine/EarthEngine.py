import ee

row = 60
path = 231

# Takes in a row and path and returns the normalised
# version of the image with the least amount of cloud cover
def addImage(path, row):
  try:
    # Get Landsat image for those co-ordinates
    dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1').filterDate('1999-01-08', '2000-01-01').filter(ee.Filter.eq('WRS_PATH', path)).filter(ee.Filter.eq('WRS_ROW', row)).sort('CLOUD_COVER'); # Sort by cloud cover (lowest to highest)

    # Extract first image from collection
    image = dataset.first();

    print (image)
    # calculate the min and max value of an image
    minMax = image.reduceRegion({
      reducer: ee.Reducer.minMax(),
      geometry: image.geometry(),
      scale: 30,
      maxPixels: 10e9});

    # use unit scale to normalize the pixel values
    unitScale = ee.ImageCollection.fromImages(
      image.bandNames().map(function(name){
        name = ee.String(name);
        var band = image.select(name);
        return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max')))) .multiply(100);
    })).toBands().rename(image.bandNames());


    Map.addLayer(unitScale.select(['B3', 'B2', 'B1']), {min: 0, max: 1}, 'unitscaled')

  catch(err):
    print ("_____ERROR OCCURED_____")
    print (err)

def view_image():
    # Load a Landsat image.
    img = ee.Image('LANDSAT/LT05/C01/T1_SR/LT05_034033_20000913')

    # Print image object WITHOUT call to getInfo(); prints serialized request instructions.
    print(img)

    # Print image object WITH call to getInfo(); prints image metadata.
    print(img.getInfo())

if __name__ == "__main__":

    # ee.Authenticate() # One time command

    ee.Initialize() # Need run everytime

    view_image()
