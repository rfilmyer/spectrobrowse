mod lib;

use std::{cmp::max, error::Error, ffi::OsStr, fs::read_dir, path::PathBuf};

use rayon::{iter::IntoParallelRefIterator, prelude::*};

use plotters::prelude::*;

use clap::{App, Arg};

fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("Spectrobrowse")
        .version("0.1.0")
        .author("Roger Filmyer <spectrobrowse@synolect.com>")
        .about("Generates spectrograms from audio files in a directory")
        .arg(Arg::with_name("INPUT_DIR")
            .help("Input Directory")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("output")
            .help("Output directory. Defaults to the current directory.")
            .short("o")
            .default_value(".")
            .takes_value(true))
        .arg(Arg::with_name("window-size")
            .help("Window size (# of frequencies for FFT). Default 2048")
            .short("w")
            .takes_value(true)
            .default_value("2048"))
        .arg(Arg::with_name("overlap-ratio")
            .help("Window Overlap Ratio (default 0.75). Must be between 0 and 1, exclusive.")
            .short("r")
            .takes_value(true)
            .default_value("0.75"))
        .get_matches();

    // parsing command line args
    let directory_path = matches.value_of_os("INPUT_DIR").unwrap();
    let output_path = matches.value_of_os("output").unwrap();
    let window_size = matches.value_of("window-size")
        .unwrap_or_else(|| panic!("A value must be specified for `window-size` or a default must be configured"))
        .parse()
        .unwrap();
    
    let overlap = matches.value_of("overlap-ratio")
        .unwrap_or_else(|| panic!("A value must be specified for `overlap-ratio` or a default must be configured"))
        .parse()?;
    if overlap >= 1.0 || overlap <= 0.0 {
        panic!("Invalid value for `overlap`: {} (must be greater than 0, less than 1)", overlap);
    }
    // let directory_path = "sound_files_smaller";
    let directory = read_dir(directory_path)?;

    let filepaths = directory
        .filter_map(|de| {
            match de {
                Ok(de) => Some(de),
                Err(e) => {eprintln!("Error when listing directory: {}", e); None}
            }
        })
        .map(|de| de.path());
    
    // load waveforms
    let waveforms = filepaths
        .map(move |fp| (fp.clone(), lib::load_soundfile_from_path(fp)))
        .filter_map(|(fp, wf)| {
            match wf {
                Ok(wf) => Some((fp, wf)),
                Err(e) => {
                    let filepath = fp.to_string_lossy();
                    eprintln!("Problem reading {}: {}", filepath, e);
                    None
                }
            }
        })
        .collect::<Vec<(PathBuf, Vec<i16>)>>(); // I would *love* to not have to collect here but I'll deal with that later
    
    // Multiprocessing begins here:
    // FFT calculation
    let spectrograms = waveforms
        .par_iter()
        .map(move |(filepath, waveform)| (filepath.to_owned(), lib::compute_spectrogram(waveform.to_owned(), window_size, overlap)))
        .filter_map(|(filepath, spectrogram)| {
            match spectrogram {
                Ok(s) => Some((filepath, s)),
                Err(e) => {
                    let filepath = filepath.to_string_lossy();
                    eprintln!("Problem calculating spectrogram for {}: {}", filepath, e);
                    None
                }
            }
        });
    
    // Graphing
    spectrograms.for_each(move |(audio_filepath, spectrogram)| {
        let output_path = std::path::Path::new(output_path)
            .join(
                audio_filepath
                .with_extension("png")
                .file_name()
                .unwrap_or(OsStr::new("output.png"))
            );

        let (num_samples, num_freq_bins) = match spectrogram.shape() {
            &[num_rows, num_columns] => (num_rows, num_columns),
            _ => panic!("Windows is a {}D array, expected a 2D array", spectrogram.ndim())
        };
        let image_height = num_freq_bins;
        let image_width = max(image_height, num_samples);
        // Eventually I want to replace with this function below, currently I end up truncating the spectrogram
        // let image_width = max(image_height, (f32::sqrt(image_height as f32) * f32::sqrt(num_samples as f32)).round() as usize);
        println!("Generating a {} wide x {} high image at {}", image_width, image_height, output_path.to_string_lossy());
        
        let image_dimensions: (u32, u32) = (image_width as u32, image_height as u32);
        let root = 
            BitMapBackend::new(
                &output_path, 
                image_dimensions, // width x height. Worth it if we ever want to resize the graph.
            ).into_drawing_area();
    
        lib::plot_spectrogram(&spectrogram, &root);
    });

    Ok(())
}
