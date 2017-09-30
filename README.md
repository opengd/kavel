# kavel
Audio manipulation tool

## Installation

First be sure you have a running Python 3 up and running.

Then start by clone this git project:
```
git clone https://github.com/opengd/kavel.git
```

Install dependencies using pip:
```
pip install ciffi
pip install numpy
```

You will need to install PySoundFile to get kavel working
http://pysoundfile.readthedocs.io/en/0.9.0/#installation

Install PySoundFile using pip:
```
pip install pysoundfile
```

On linux you also need to install libsndfile:
```
sudo apt-get install libsndfile1
```

## How to run

```
python3 kavel.py
```

Example:
```
python3 kavel.py -r my_audio_file.flac
```
This will reverse the input audio file and output a new flac file named "my_audio_file_r_k.flac".

To use paulstretch on a audio file:
```
python3 kavel.py -s --stretch_amount=2.0 --window_size=1 my_audio_file.flac
```
This will create a new flac file called "my_audio_file_k_s2.0_w1.0.flac" stretched using paulstretch.

For more info about paulstretch:
https://github.com/paulnasca/paulstretch_python

To see command help text for kavel:
```
python3 kavel.py --help
```

List all supported audio formats:
```
python3 kavel.py -l
```