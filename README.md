# spectrobrowse
Browse through audio clips by viewing their spectrograms

# Current Status
* Usable as a program. Currently a CLI program that dumps image files to a directory. Use `--help` to get more details.

# Current Bugs
* Graphs are truncated instead of downscaled

# Current Todos
* ~Generate FFTs in log scale instead of linear scale~
* CLI Ergonomics - command line args (`clap`) and progress bars and feedback (`indicatif`)

# Project Goals
* Load a directory of audio files (ogg vorbis is a requirement here)
* Turn each audio file into a waveform(?)
* Generate the spectrogram from that waveform
* Collect all the spectrograms somewhere
* Put them on a (static, offline) webpage

# Ideas/Todos
## Libraries
* ~Will [`sonogram`](https://github.com/psiphi75/sonogram) be fast enough or do I have to roll my own with [`RustFFT`](https://github.com/ejmahler/RustFFT)?~
* Sonogram takes WAV files. Do I have to use something like [`rodio`](https://github.com/RustAudio/rodio) to convert OGG files?
* How do I generate the webpage (seems like the easiest UI option to start out with)?  [`askama`](https://github.com/djc/askama) or `liquid` or something else?
* How messed up is that web page going to be?
