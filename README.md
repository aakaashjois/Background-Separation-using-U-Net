# Background-Separation-using-U-Net

Using a U-Net model to perform Image Segmentation on Carvana Image Masking Challenge Dataset

## Setup

This project requires Python 3 to be installed. 

### If Python 3 is not default on the system, replace all `pip` with `pip3` and `python` with `python3`

The python libraries needed are in the [`requirements.txt`](./requirements.txt) file. They can be installed using `pip`.

`tensorflow-gpu` is the default library. In case a compatible GPU is not available, change `tensorflow-gpu` to `tensorflow` in [`requirements.txt`](./requirements.txt) before installing using `pip`.

The following command can be used to install all the dependencies.
````
pip install requirements.txt
````

Once the dependencies are installed, the image data needs to be downloaded. The instructions to download the images are available in the respective folders in [`data`](./data) folder.

## Run

The file [`run.py`](./run.py) can be run using the following command.
```
python run.py
````

The program takes an input for scale factor of the original image.
Once finished, the best fit model and history of the run is exported to the current directory.
The model can be loaded using Keras to further improve or for making predictions. The history can be used for visualization.

## Report

The report of this project is available [here](./report/report.pdf).

## License

This project is licensed under Apache License 2.0. The details can be found [here](./LICENSE).

## Credits

All the contributions to this project is made by [Aakaash Jois](https://github.com/aakaashjois) and [Alp Aygar](https://github.com/alpombeo).
