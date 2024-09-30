# class-activation-map
Class activation maps (CAM) output for the batch using PyTorch.

## Examples
ResNet50 CAM output examples:
| Original  | CAM |
| ------------- | ------------- |
|   <img src="https://github.com/georgechaikin/class-activation-map/blob/main/data/images/Golden_Retriever_with_tennis_ball.jpg?raw=true" width="224" height="224"/>  | <img src="https://github.com/georgechaikin/class-activation-map/blob/main/data/cam-images/Golden_Retriever_with_tennis_ball_207.jpg?raw=true" />  |
|   <img src="https://github.com/georgechaikin/class-activation-map/blob/main/data/images/200px-Dalmatian_puppy,_four_months.png?raw=true" width="224" height="224"/>  | <img src="https://github.com/georgechaikin/class-activation-map/blob/main/data/cam-images/200px-Dalmatian_puppy,_four_months_251.png?raw=true" />  |


## Prerequisites
* Python (>=3.10)
* Poetry (>=1.8.3) (for package building)

## Requirements for data.
The script was tested with the following image formats:
- JPEG (.jpeg, .jpg)
- PNG (.png)
- TIFF (.tif, .tiff)

## How to build and use the package.
```shell
# Build the package using Poetry.
poetry install --without dev
poetry build --format wheel
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate # There is another command for Windows: venv/Scripts/activate
# Run the script.
save_heatmaps data/images data/cam-images -c 207 -c 251
```
Also, you can use Python functions instead of console scripts.
There is ```save_heatmaps``` script implementation in [main.py](https://github.com/georgechaikin/class-activation-map/blob/main/class_activation_map/main.py).

## Available scripts
- save_heatmaps:
```shell
Usage: save_heatmaps [OPTIONS] IMG_DIR SAVE_DIR

  Saves heatmaps for images in IMG_DIR to SAVE_DIR (with ResNet50 model).

Options:
  -c, --class-index INTEGER  Class index.
  --batch-size INTEGER       Batch size.
  -v, --verbose              Progress bar on/off.
  --help                     Show this message and exit.
```

## TODO
- [ ] MIME-type support for images.

