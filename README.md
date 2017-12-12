# Background-Separation-using-U-Net
Using a U-Net model to perform Image Segmentation on Carvana Image Masking Challenge Dataset

## Setup

This project requires Python 3 to be installed on the machine. The python libraries needed are in the `requirements.txt` file.
They can be installed using `pip`.

`tensorflow-gpu` is the default library. In case a compatible GPU is not available, change `tensorflow-gpu` to `tensorflow` in `requirements.txt` before installing using `pip`.

The following command can be used to install all the dependencies.
````
pip install requirements.txt
````

## Run
The file `run.py` can be run using the command line.
```
python run.py
````

The program takes an input for scale factor of the original image.
Once finished, the best fit model and history of the run is exported to the current directory.
The model can be loaded using Keras to further improve or for making predictions. The history can be used for visualization.

> If Python 3 is not default on the system, then replace all `pip` with `pip3` and `python` with `python3`.
