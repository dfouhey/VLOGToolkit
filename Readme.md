# Welcome to the VLOG Dataset!

This repository contains useful tools for working with the VLOG dataset. We will be updating
this with additional tools as time goes on. For the dataset itself, please see [http://people.eecs.berkeley.edu/~dfouhey/2017/VLOG/]

Dependencies:

* numpy
* scipy
* sklearn

You have to specify paths to files in the following files (although all the scripts will prompt you for paths):

* `DATA_ROOT` should contain the path of the labels file
* `CLIP_ROOT` should contain the path of the clips, dumped in the /X/Y/Z/v_TUVWXYZ/001/clip.mp4 format described in the data folder
* `FRAME_ROOT` should contain the path of the frames, dumped in the /X/Y/Z/v_TUVWXZ/001/frame000001.jpg format described in the data folder

## Evaluation

`eval.py` contains evaluation scripts that will let you reproduce the numbers reported in the paper given
the precomputed predictions provided with the dataset. 

You always have to specify:
* `--benchmark` (`hand_object`, `hand_state`, `scene_proxemic`, `scene_category`)
* `--predictions` a file specifying the predictions, either a .txt or .npy file

The following options always apply:
* `--dobci` This compute 95% bootstrapped confidence intervals using the bias correction method, boostrapping the uploader (i.e., not 
 treating the videos as independently). They seem pretty tight and not worth worrying too much about. 
* `--dobreakdown` (`scene_proxemic`, `scene_category`) This runs the evaluation for each of the proxemic/semantic categories for scenes.

## Demo Reading Code

`demo.py` contains demo code that loads and displays the data. This will open a series of windows using `eog` and `cvlc`, so it may
or may not work depending on your computer. It also displays hand bounding boxes by using opencv.



