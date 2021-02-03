## Motivations

In January 2020, 11,000 scientists endorsed a BioScience article which stated “the climate crisis has arrived and is accelerating faster than most scientists expected” [13]. From land use changes to deforestation it appears that the influence we, as humans, have on the environment is the main source of climate change [6]. Since 1990 the rate at which deforestation occurs in the Amazon is only getting greater with every passing year [9]. Deforestation such as this is does not only have a destructive impact on the environment but is the primary threat to the survival of many species such as the orangutan [15]. In some parts of the world rainforests are illegally being cut down to make way for agriculture, [15] while in other parts of the world wildfires consume and eradicate the landscape leaving little more than ash behind [4].

Fighting this illegal deforestation is a challenge due to the sheer scale at which it occurs and the massive land area it occurs in. On top of this loggers will often camou- flage their equipment as well as create hidden roads to specific deforestation sites [17]. This makes it hard for conservationists to pin point exactly where the deforestation is ongoing as it requires a lot of manpower to monitor and spot it as it happens.

Using machine learning in combination with satellite imagery it may be possible to identify different biomes and with this information be able to remotely monitor the rate at which biomes are changing in size. In the process this would reduce response times to areas that are under heavy illegal deforestation.

## Possible techniques:

- Pre-training on ImageNet

    - https://arxiv.org/pdf/1805.02855.pdf
    - Apparently a de facto standard (worth looking into)
    - Drastically reduces the amount of training data needed for new tasks
    - Do not perform well and cannot take advantage of additional spectral bands
    - Fewer occlusions, permutations of object placement and changes of scale to content with
        - provides powerful signal for learning representations

- Tile2Vec

    - https://arxiv.org/pdf/1805.02855.pdf
    - A counter to pre-training on ImageNet
    - Relies on the assumption that tiles close together have similar semantics 
    - An unsupervised feature learning algorithm for spatially distributed data on tasks from land cover classification to poverty prediction.
        - Outperforms supervised CNNs trained on 50k labeled examples.
        - Was trained on the National Agriculture Imagery Program (NAIP)
            - Provides aerial imagery for public use that has four spectral bands — red (R), green (G), blue (B), and infrared (N) — at 0.6 m ground resolution
    - How it works:
        - A convolutional neural network is trained on *triplets* of tiles. 
            - Each triplet consists of an anchor tile (t<sub>a</sub>), a neighbour tile (t<sub>n</sub>) - who's centre is within the chosen neighbourhood distance of t<sub>a</sub> - and a distance tile (t<sub>d</sub>)
            - There are two parameters you can change:
                - **Tile size**: pixel width and height of single tile
                - **Neighbourhood**: region around the anchor tile from which to sample the neighbour tile

    ![Screenshot 2020-05-19 at 09.42.15](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-19 at 09.42.15.png)

    ![Screenshot 2020-05-19 at 10.03.42](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-19 at 10.03.42.png)

- Fine Pixel Spectral Analysis (Hyperspectral analysis)
    - Hyperspectral images have a high price
    - Given how much of the light sprecturm this uses it is believed that any given object should have a unique spectral signature
    
- Look for research in the tropics and then move it across

- Semantic segmentation 

- Pre Processing the data

    - Normalise the images (Very important with few training samples)
        - Histogram Equalisation: Increase brightness of all channels to high value

## Tile2Vec

- https://arxiv.org/pdf/1805.02855.pdf
- A counter to pre-training on ImageNet
- Relies on the assumption that tiles close together have similar semantics 
- An unsupervised feature learning algorithm for spatially distributed data on tasks from land cover classification to poverty prediction.
    - Outperforms supervised CNNs trained on 50k labeled examples.
    - Was trained on the National Agriculture Imagery Program (NAIP)
        - Provides aerial imagery for public use that has four spectral bands — red (R), green (G), blue (B), and infrared (N) — at 0.6 m ground resolution
- How it works:
    - ![IMG_5399](/Users/calummcmeekin/Downloads/IMG_5399.JPG)
        1. Images
            1. Must have double the number of images to triplets that will be sampled
        2. Tile2Vec
            1. Takes an input of triplets and assumes that triplets taken close to each other will be similar when compared to a triplet taken from far away or from another image
            2. The model is then trained with weights that are specific to the semantics of the images the triplets were created from
            3. Once a model has sufficiently been trained (fine tuning the weights) on a large amount of triplets it can be used to accuratly create a function which maps from the tile space to the model space (diagram on bottom right of whiteboard)
        3. Classifier
            1. Uses a fresh set of tiles different to the ones the model was trained on
            2. These tiles are then split into a training and testing set
            3. If classifier being used is a random forest then you provide the number of classes
            4. Classifier uses these encoded tiles to classify (see diagram bottom right of whiteboard)
    - The Triplet Idea
        - Each triplet consists of an anchor tile (t<sub>a</sub>), a neighbour tile (t<sub>n</sub>) - who's centre is within the chosen neighbourhood distance of t<sub>a</sub> - and a distance tile (t<sub>d</sub>)
        - There are two parameters you can change:
            - **Tile size**: pixel width and height of single tile
            - **Neighbourhood**: region around the anchor tile from which to sample the neighbour tile

![Screenshot 2020-05-19 at 09.42.15](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-19 at 09.42.15.png)

![Screenshot 2020-05-19 at 10.03.42](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-19 at 10.03.42.png)

- Uses .npy (NumPy array) file format for the images 
- List of required libraries (installed in following order):

```bash
pip install matplotlib
pip install numpy
pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
pip install torch
```

- Currently works with images in .tif format

- Takes a small image - splits into small tiles - maps tiles to 100 dimension vectors - with these vectors can take clusters - means of clusters - use standard machine learning approaches on these means

  ​    

## Datasets

- Landsat Imagery
    - https://www-sciencedirect-com.ezproxy.is.ed.ac.uk/science/article/pii/S0034425713002204 
        - Mainly used Landsat 7 due to limited option
            - Landsat 7 had failures in 2003 meaning there is lost data
        - Selected images with lowest cloud cover that were available in time frame of ± 1 year
        - Created mosaic images for each data and each site by simple overlay, with the least cloudy image at the top of the mosaic![Screenshot 2020-05-20 at 09.17.43](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-20 at 09.17.43.png)
    - EarthExplorer
        - Quick bulk download access for Landsat 8 imagery
            - Landsat 7/8 level 2
                - Can specify cloud cover but not day/night
                - Downloads an image with multiple bands, need to figure out how to brighten the image once channels merged
            - Landsat 7/8 level 1
                - Can specify day/night but not cloud cover
    - Google EarthEngine
      - Easy to use interface
      - Pre-processing handled by super computers
      - Stores images on the cloud

## Estimating deforestation

- Perhaps best just to take data from two periods e.g. 2005 and 2010 and compare the difference between the area of forest in these two time zones
- https://www-sciencedirect-com.ezproxy.is.ed.ac.uk/science/article/pii/S0034425713002204
    - ![Screenshot 2020-05-20 at 09.21.36](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-20 at 09.21.36.png)

## Amazon Basin Biomes

A biome is a geographical region containing diverse combinations of fauna and flora. The difference between biomes depends on their temperature, humidity and how fertile the soil is. The main biomes in South America (where this paper is focused) are the Amazonia, Caatinga, Cerrado, Atlantic Forest, Pampas and the Pantanal. Each of these biomes are made up of unique combinations of smaller ecoregions.

- Amazonia
    - 22-34°C [constant & year round]
- Cerrado
    - Dry winter: 
        - April - September
        - 0°C -> 46°C
    - Rainy summer:  
        - October - March
        - 20°C -> 26°C
- Caatinga
    - 32-36°C

### Ecoregions

- Forest (78% [Wiki])
    - Tropical broadleaf forest (majority)
        - Broad green canopy hiding any view of the ground below
        - ![Screenshot 2020-05-21 at 10.25.48](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-21 at 10.25.48.png)
- Floodplain forests (3-4% [WWF] or 5.83% [Wiki])
    - **Varzea forests** - feb by muddy rivers
    - **Igapo forests** - blackwater and clearwater tributaries
    - **Tidal forests** - located in the estuary
    - These biomes generally occur in the northern amazon basin (above the equator)
    - Created by heavy seasonal rainfall which raises the level of the river level fluctuations 
- Savannas (12.75% [Wiki])
    - Campina
        - Open forest on sandy soil where sunlight reaches the ground
        - Mainly in the south
        - Vegetation stunted
    - Campinarana
        - Savannah, scrub and forests make up campinarana which can all be found on leached white sands around circular swampy depressions in lowland tropical moist forest
        - In the Rio Negro and Rio Branco basins in the north of Brazil
        - ![Screenshot 2020-05-21 at 10.24.52](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-21 at 10.24.52.png)
- Other Biomes within the amazon basin (4%)
    - grasslands
    - swamps
    - bamboos
    - Palm forests
    
    
### Results

Random Forest Classifier:

| Model                              | Accuracy ± Std.Dev. | Macro Precision ± Std.Dev. | Macro Recall ± Std.Dev. | Macro F1-Score ± Std.Dev. |
| ---------------------------------- | ------------------- | -------------------------- | ----------------------- | ------------------------- |
| Tile2vec                           | 0.7878 ± 0.0060     | 0.7863 ± 0.0057            | 0.7874 ± 0.0057         | 0.7899 ± 0.0058           |
| Tile2Vec<br /> Pre-trained on NAIP | 0.8686 ± 0.0053     | 0.8702 ± 0.0053            | 0.8685 ± 0.0052         | 0.8689 ± 0.0052           |
| ResNet 18                          | 0.8812 ± 0.0043     | 0.8824 ± 0.0049            | 0.8811 ± 0.0048         | 0.8814 ± 0.0048           |
| AlexNet                            | 0.9042 ± 0.0043     | 0.9056 ± 0.0042            | 0.9041 ± 0.0043         | 0.9043 ± 0.0043           |



KNN where K is picked as optimal for all models

| Model                              | Accuracy ± Std.Dev. | Macro Precision ± Std.Dev. | Macro Recall ± Std.Dev. | Macro F1-Score ± Std.Dev. |
| ---------------------------------- | ------------------- | -------------------------- | ----------------------- | ------------------------- |
| Tile2vec                           | 0.7878 ± 0.0060     | 0.7863 ± 0.0057            | 0.7874 ± 0.0057         | 0.7899 ± 0.0058           |
| Tile2Vec<br /> Pre-trained on NAIP | 0.8453 ± 0.0049     | 0.8478± 0.0049             | 0.8451± 0.0049          | 0.8457± 0.0049            |
| ResNet 18                          | 0.8812 ± 0.0043     | 0.8824 ± 0.0049            | 0.8811 ± 0.0048         | 0.8814 ± 0.0048           |
| AlexNet                            | 0.9042 ± 0.0043     | 0.9056 ± 0.0042            | 0.9041 ± 0.0043         | 0.9043 ± 0.0043           |



### TODO

- [ ] Get Images (lots of them)
  
    - [ ] **[EarthExplorer Bulk Download Application (BDA)](https://earthexplorer.usgs.gov/bulk)**
    - [ ] Google Earth Engine
    
- [ ] Construct the tile2vec trainer in a single python file

- [ ] Explore to see if I can use a pretrained tile2vec model

- [ ] Reserve space on the edi cluster 

- [ ] Run tile2vec python script on edi cluster 

- [x] Check project web page for information

- [ ] Create background information

- [ ] Collecting ground truth where you can find out what biome is at what co-ordinate

- [ ] For a better mark develop an algorithm (2nd year (?))
