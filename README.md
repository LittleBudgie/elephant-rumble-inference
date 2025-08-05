# Elephant Rumble Inference Updated Version

This repository is a revised version of the elephant rumble inference model created by FruitPunchAI (can be found at this link: https://github.com/ramayer/elephant-rumble-inference). 

### Notable Version Updates Include:
* Can now accept a directory of audio files as an input, with new parameter "--merge-raven-tables"
    * If true, merges all selection tables into one selection table to output.
         * Each audio file's individual detections can be found in the Raven selection table through the file path column.
    * If false, goes through all audio files in a directory and runs model on them, outputting Raven selection tables for each of them individually
    * Default is set to false
* Changed defaults for some parameters (automatically save raven file)

## Installing This Version of the Package
You can use the same reference video shared, however, when providing the github link in the pip install, use this line instead:
```
pip install "git+https://github.com/LittleBudgie/elephant-rumble-inference.git"
```

## List of Parameters
This is a list taken from the original developer's main.py. All descriptions of parameters written by the original developers (except --merge-files-in-dir)

Example of code to put into terminal to run with a specified parameter value: 
```
elephant-rumble-inference "test.wav" --merge-files-in-dir=True
```

* --model
    * Specify the model name. Full list here: http://0ape.com/pretrained_models/
* --save-dir
    * Directory to save outputs
    * Default = "output" 
* --visualizations-per-audio-file
    * Default: 1
* --model-name
    * Default: 2024-07-03.pth
    * Name of the model.  Recommended 2024-06-29.pth, 2024-06-30.pth, 2024-07-03.pth, or 2024-07-09.pth. Full list of available models here: http://0ape.com/pretrained_models/ )
* --duration-of-visualizations
    * Default: 1
    * Minutes of audio for a visualization. 15 is nice for wide monitor, 60 is interesting if you don't mind horizontal scrolling
* --save-scores
    * Default: True
    * Save classification scores to a file
* --save-raven
    * Default: True
    * Save a raven file with found labels
* --load-labels-from-raven-file-folder
    * Show labels from existing raven files
* --limit-audio-hours
    * Default limit 24
    * Limit audio hours for file
* --merge-raven-tables (**NEW**)
    * Merges all audio files in each subdirectory of input path. Outputs merged raven selection table for merged audio file with all file paths in table.
    * Default: False
