##Motivations

- Indicators are needed:
    - To study and mitigate against further environmental degradation, as well as monitor progress to sustainable development, indicators of environmental status are required (Hall 2001) [https://www.tandfonline.com/doi/abs/10.1080/0143116031000103853#aHR0cHM6Ly93d3cudGFuZGZvbmxpbmUuY29tL2RvaS9wZGYvMTAuMTA4MC8wMTQzMTE2MDMxMDAwMTAzODUzP25lZWRBY2Nlc3M9dHJ1ZUBAQDA=]
-  Remote sensing can provide a wealth of environmental data over a range of spatial and temporal scales and so may play a major role in the provision of indicators of environmental condition that may inform sustainable development and associated decision-making (Schultink 1992, Rao 2001, Rao and Pant 2001, Chen 2002) [https://www.tandfonline.com/doi/abs/10.1080/0143116031000103853#aHR0cHM6Ly93d3cudGFuZGZvbmxpbmUuY29tL2RvaS9wZGYvMTAuMTA4MC8wMTQzMTE2MDMxMDAwMTAzODUzP25lZWRBY2Nlc3M9dHJ1ZUBAQDA=]
- Land cover change is one of the most important variables of environmental change and represents the largest threat to ecological systems for at least the next 100 years (Skole 1994, Chapinet al. 2000) [https://www.tandfonline.com/doi/abs/10.1080/0143116031000103853#aHR0cHM6Ly93d3cudGFuZGZvbmxpbmUuY29tL2RvaS9wZGYvMTAuMTA4MC8wMTQzMTE2MDMxMDAwMTAzODUzP25lZWRBY2Nlc3M9dHJ1ZUBAQDA=]

##Possible techniques:

- Create what an ideal final product would be

- Pixel comparison
    - Take a picture and label all the pixels that have the wanted biome in them
    - When a new picture comes in cross-check the image with those previously taken and look to see if a previously marked pixel is no longer the same?
    - https://www.sciencedirect.com/science/article/pii/S0034425718305534 (Section 3)

- PLANETLAB 

    - A distributed system to allow you to run large programs on a slice of the network
    - Has been retired (no longer in operation)

- K-Means

    - 

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

##Datasets

- Radiant
    - email ITO to find out if i need money for AWS
    - Check to see if someone has compared an unsupervised method to this method - they haven't
    - With low training data - UNet
    - Give Sohan a pres on what the data is and what it looks like
        - **Possible issue**: This data does not encompass multi images for the same area meaning it could help in providing training data but it would not help with monitoring changes in biomes over the years
        - **ANother issue** The data is in 'requester pays' mode so I am apprehensive to use the data as it does not state anywhere how much the price per image is. (could be from £0.01 to £10) 
- Landsat Imagery
    - 
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
                - 

## Estimating deforestation

- Perhaps best just to take data from two periods e.g. 2005 and 2010 and compare the difference between the area of forest in these two time zones
- https://www-sciencedirect-com.ezproxy.is.ed.ac.uk/science/article/pii/S0034425713002204
    - ![Screenshot 2020-05-20 at 09.21.36](/Users/calummcmeekin/Library/Application Support/typora-user-images/Screenshot 2020-05-20 at 09.21.36.png)

## Amazon Basin Biomes

- Biomes change characterestics depending on the season -> Try get images in same timezone
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



Courses that are good for machine learning:

- MLPR
- 

 Image and Vision Computing, computer human interaction, software testing, software design and modelling, algorithmic foundations of data science n

Heindrinken task thingy





### TODO

- [ ] Get Images (lots of them)
  
    - [ ] **[EarthExplorer Bulk Download Application (BDA)](https://earthexplorer.usgs.gov/bulk)**
    
- [ ] Construct the tile2vec trainer in a single python file

- [ ] Explore to see if I can use a pretrained tile2vec model

- [ ] Reserve space on the edi cluster 

- [ ] Run tile2vec python script on edi cluster 

- [x] Check project web page for information

- [ ] Create a background information thing

- [ ] Collecting ground truth where you can find out what biome is at what co-ordinate

- [ ] Get a hypothesis: E.g. Do places with a certain biome have a certain quality of life

- [ ] For a better mark develop an algorithm (2nd year (?))

- [ ] Send Sohan reports every week and a half

    - [x] Use LaTeX
    - [x] Download thesis format from informatics page
    - [x] Make it as short as possible while getting the point across (bullet points)

    - [x] Include diagrams (where data is coming from, how it is being merged what methods you are using, output you are getting)
    - [x] Can I provide one line impact statement which says the impact of the project. 
    - [ ] Have a never ending report with a bullet point list at the start detailing what is being done (with some figures)
    - [ ] Include lots of figures

- [ ] Create a TEAMs project page and put all the files there

- [ ] Run the scripts on the university Remote Desktop

- [ ] 

