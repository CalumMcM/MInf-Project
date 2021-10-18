import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar

# Load image
im = Image.open("/Users/calummcmeekin/Downloads/AmazonALLLS7.png")

width, height = im.size


new_res = width/0.15
scale = new_res/30

# Create subplot
fig, ax = plt.subplots()
ax.axis("off")

# Plot image
ax.imshow(im, cmap="gray")

# Create scale bar
scalebar = ScaleBar(scale, location='lower right')
ax.add_artist(scalebar)

# Show
plt.tight_layout()
plt.savefig('/Users/calummcmeekin/Downloads/AmazonALLLS7_ScaleBar.png', bbox_inches='tight')

plt.show()
