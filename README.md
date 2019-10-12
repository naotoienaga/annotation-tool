# annotation-tool
Semi automatic annotation tool


## Requirements
The code was tested with python 3.6.8, opencv 4.1.1, lightgbm 2.2.4, and anaconda.

## Usage
- Use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect key points of the video and annotate part of the input video. Annotate at least one rest and gesture (Any tire name is ok other than PREDICTED and QUERY).
- Edit "config.ini" (Please specify path to json file, path to annotation file path etc.)
- Run `python anno_tool.py`
- An elan file with predicted annotation will be generated. If you are not satisfied with the prediction results, add more annotations and run again (delete PREDICTED tire and QUERY tire)
