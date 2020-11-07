// HOW TO USE
// 1. Choose the Biome (line 15)
// 2. Choose the year (line 18)
// 3. Set the geometry at "geometry" button on the upleft side of the map
// 4. RUN the script
// 5. Run the task of export on the upright panel
//
// Obs: If you define a big geometry, maybe your data get splitted
// on Google Drive. You will need to make a local mosaic using ArcGis, QGis, ENVI...

// Do not change the mosaics asset
var asset = 'projects/mapbiomas-workspace/MOSAICOS/workspace-c3';

// Choose the biome name: 'AMAZONIA', 'CAATINGA', 'CERRADO', 'MATAATLANTICA', 'PAMPA' or 'PANTANAL'
var biome = 'AMAZONIA';

// Define the year of your interest
var year = 2019;

// Output file name
var fileName = 'mosaic-' + String(year);

// Look the available band list at the console
var exportBands = [
      "median_blue",
      "median_gree",
      "median_red",
      "median_bir",
      "median_swir1",
      "median_swir2",
  ];

// Amazonia
var geometryAMA = ee.Geometry.Polygon([[
  [-68, -9],  // BL point
  [-66, -9],
  [-58, -15],
  [-50, -9],  // BR point
  [-49, -1],  // TR point
  [-68, -1],  // TL point
  [-68, -9]   // BL point
  ]]);

// Caatinga
var geometryCAT = ee.Geometry.Polygon([[
  [-43, -16],  // BL point
  [-35.5, -9],  // TR point
  [-43, -3],  // TL point
  ]]);

//Cerrado
var geometryCER = ee.Geometry.Polygon([[
  [-60.80975401310168, -30.519994257183307],  // BL point
  [-20.93634092716418, -30.519994257183307],  // BR point
  [-20.93634092716418, 10.617795520325061],  // TL point
  [-60.80975401310168, 10.617795520325061],  // TR point
  [-60.80975401310168, -30.519994257183307]   // BL point
  ]]);

// Get the moisac
var mosaicCER = ee.ImageCollection(asset)
               .filterMetadata('biome', 'equals', 'CERRADO')
               .filterMetadata('year', 'equals', year)
               .filterBounds(geometryCER)
               .mosaic();

var mosaicAMA = ee.ImageCollection(asset)
               .filterMetadata('biome', 'equals', 'AMAZONIA')
               .filterMetadata('year', 'equals', year)
               .mosaic();

var mosaicCAT = ee.ImageCollection(asset)
               .filterMetadata('biome', 'equals', 'CAATINGA')
               .filterMetadata('year', 'equals', year)
               .filterBounds(geometryCAT)
               .mosaic();

// prints all bands available to download
print(mosaicCER.bandNames());
print(mosaicAMA.bandNames());
print(mosaicCAT.bandNames());
/*
// Shows the mosaic on map
Map.addLayer(mosaicCER.clip(geometryCER),
    {
        bands: 'median_swir1,median_nir,median_red',
        gain: '0.08,0.06,0.2',
        gamma: 0.75
    },

    'mapbiomas mosaic CER'
);
*/

// Shows the mosaic on map
Map.addLayer(geometryAMA, {color: 'FF0000'}, 'colored');


Map.addLayer(mosaicAMA,
    {
        bands: 'median_swir1,median_nir,median_red', // CONVERT TO RGB
        gain: '0.08,0.06,0.2',
        gamma: 0.75
    },

    'mapbiomas mosaic AMA'
);
/*
// Shows the mosaic on map
Map.addLayer(mosaicCAT.clip(geometryCAT),
    {
        bands: 'median_swir1,median_nir,median_red',
        gain: '0.08,0.06,0.2',
        gamma: 0.75
    },

    'mapbiomas mosaic CAT'
);

//Map.addLayer(geometryAMA, {color: '323aa8'}, 'colorblue');
//Map.addLayer(geometryCAT, {color: '43ed00'}, 'colorgreen');
// Exports the data to MAPBIOMAS-EXPORT folder in your Google Drive
/*
Export.image.toDrive(
      {
        'image': mosaic.int32(),
        'description': fileName,
        'folder': 'MAPBIOMAS-EXPORT',
        'fileNamePrefix': fileName,
        'region': geometry,
        'scale': 30,
        'maxPixels': 1e13,
        'fileFormat': 'GeoTIFF'
      }
);
*/
