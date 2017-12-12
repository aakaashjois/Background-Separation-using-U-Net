# Background-Separation-using-U-Net
Using a U-Net model to perform Image Segmentation on Carvana Image Masking Challenge Dataset

# Run
The file `run.py` can be run using the command line.
```
python3 run.py
````

The program takes an input for scale factor of the original image.
Once finished, the best fit model and history of the run is exported to the current directory.
The model can be loaded using Keras to further improve or for making predictions. The history can be used for visualization.
