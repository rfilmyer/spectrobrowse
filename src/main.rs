mod lib;

use std::{
    cmp::max,
    error::Error,
    ffi::{OsStr, OsString},
    fs::read_dir,
    path::PathBuf,
};

use ndarray::Array2;
use rayon::{iter::IntoParallelRefIterator, prelude::*};

use clap::{App, Arg};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};

struct DecodedAudioFile {
    filepath: PathBuf,
    file_stem: OsString,
    waveform: Vec<i16>,
}

struct Spectrogram {
    audio_filepath: PathBuf,
    audio_file_stem: OsString,
    spectrogram: Array2<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("Spectrobrowse")
        .version("0.1.0")
        .author("Roger Filmyer <spectrobrowse@synolect.com>")
        .about("Generates spectrograms from audio files in a directory")
        .arg(
            Arg::with_name("INPUT_DIR")
                .help("Input Directory")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .help("Output directory. Defaults to the current directory.")
                .short("o")
                .long("output")
                .default_value(".")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("window-size")
                .help("Window size (# of frequencies for FFT).")
                .short("w")
                .long("window-size")
                .takes_value(true)
                .default_value("2048"),
        )
        .arg(
            Arg::with_name("overlap-ratio")
                .help("Window Overlap Ratio. Must be between 0 and 1, exclusive.")
                .short("r")
                .long("overlap-ratio")
                .takes_value(true)
                .default_value("0.75"),
        )
        .arg(
            Arg::with_name("linear")
                .help("Graph frequencies on a linear scale (default is logarithmic)")
                .short("l")
                .long("linear"),
        )
        .get_matches();

    // parsing command line args
    let input_directory_path = matches.value_of_os("INPUT_DIR").unwrap();
    let input_directory_path = PathBuf::from(input_directory_path);

    let output_directory_path = matches.value_of_os("output").unwrap();
    let output_directory_path = PathBuf::from(output_directory_path);

    let window_size = matches
        .value_of("window-size")
        .unwrap_or_else(|| {
            panic!("A value must be specified for `window-size` or a default must be configured")
        })
        .parse()
        .unwrap();

    let overlap = matches
        .value_of("overlap-ratio")
        .unwrap_or_else(|| {
            panic!("A value must be specified for `overlap-ratio` or a default must be configured")
        })
        .parse()?;

    let use_linear_scale = matches.is_present("linear");

    // validate values for the arguments
    assert!(
        input_directory_path.exists(),
        "Input directory does not exist: {}",
        input_directory_path.to_string_lossy()
    );
    assert!(
        input_directory_path.is_dir(),
        "Input directory is not a directory: {}",
        input_directory_path.to_string_lossy()
    );

    assert!(
        output_directory_path.exists(),
        "Output directory does not exist: {}",
        output_directory_path.to_string_lossy()
    );
    assert!(
        output_directory_path.is_dir(),
        "Output directory is not a directory: {}",
        output_directory_path.to_string_lossy()
    );
    let output_directory_path_metadata = output_directory_path.metadata().expect(
        format!(
            "Output directory could not be accessed: {}",
            output_directory_path.to_string_lossy()
        )
        .as_str(),
    );
    assert!(
        !output_directory_path_metadata.permissions().readonly(),
        "Output directory is read-only: {}",
        output_directory_path.to_string_lossy()
    );

    assert!(
        (overlap > 0.0) & (overlap < 1.0),
        "Invalid value for `overlap`: {} (must be greater than 0, less than 1)",
        overlap
    );

    // Now reading the directory and looking for files
    let directory = read_dir(input_directory_path)?;

    let input_filepaths = directory
        .filter_map(|de| match de {
            Ok(de) => Some(de),
            Err(e) => {
                eprintln!("Error when listing directory: {}", e);
                None
            }
        })
        .map(|de| de.path());

    // load waveforms
    let waveforms = input_filepaths
        .progress()
        .map(move |filepath| {
            // Decode the audio files
            let file_stem = filepath
                .file_name()
                .unwrap_or_else(|| {
                    eprintln!(
                        "Could not find filename for {}, using \"output\" instead.",
                        filepath.to_string_lossy()
                    );
                    OsStr::new("output")
                })
                .to_os_string();
            let waveform = lib::load_soundfile_from_path(&filepath)
                .map_err(|error| {
                    eprintln!(
                        "Problem reading file {}: {}",
                        filepath.to_string_lossy(),
                        error
                    );
                    error
                })
                .ok();
            return match waveform {
                Some(waveform) => Some(DecodedAudioFile {
                    filepath,
                    file_stem,
                    waveform,
                }),
                None => None,
            };
        })
        .filter_map(|x| x)
        .collect::<Vec<DecodedAudioFile>>(); // I would *love* to not have to collect here but I'll deal with that later

    let num_files = waveforms.len() as u64;
    let fft_progress_bar = ProgressBar::new(num_files);
    let graphing_progress_bar = ProgressBar::new(num_files);
    // Multiprocessing begins here:
    // FFT calculation
    let spectrograms = waveforms
        .par_iter()
        .progress_with(fft_progress_bar)
        .filter_map(move |a| {
            let audio_file_stem = a.file_stem.clone();
            let audio_filepath = a.filepath.clone();
            let computed_spectrogram = lib::compute_spectrogram(
                a.waveform.to_owned(),
                window_size,
                overlap,
                !use_linear_scale,
            );

            match computed_spectrogram {
                Ok(spectrogram) => Some(Spectrogram {
                    audio_file_stem,
                    audio_filepath,
                    spectrogram,
                }),
                Err(e) => {
                    let audio_filepath = audio_filepath.to_string_lossy();
                    eprintln!(
                        "Problem calculating spectrogram for {}: {}",
                        audio_filepath, e
                    );
                    None
                }
            }
        });

    // Graphing
    spectrograms
        .progress_with(graphing_progress_bar)
        .map(|s| {
            // determine output filename
            let output_file_path =
                output_directory_path.join(PathBuf::from(s.audio_file_stem).with_extension("png"));
            (output_file_path, s.spectrogram)
        })
        .map(|(output_file_path, spectrogram)| {
            // determine graph height and width
            let (num_samples, num_freq_bins) = spectrogram.dim();

            let image_height = num_freq_bins;
            let image_width = max(
                image_height,
                ((image_height as f32).sqrt() * (num_samples as f32).sqrt()).round() as usize,
            );
            (output_file_path, image_height, image_width, spectrogram)
        })
        .map(
            |(output_file_path, image_height, image_width, spectrogram)| {
                let image_buffer =
                    lib::render_spectrogram(&spectrogram, image_width as u32, image_height as u32);
                (output_file_path, image_buffer)
            },
        )
        .progress()
        .for_each(|(output_file_path, image_buffer)| {
            image_buffer
                .save(&output_file_path)
                .unwrap_or_else(|error| {
                    eprintln!(
                        "Error saving {}: {}",
                        output_file_path.to_string_lossy(),
                        error
                    )
                });
        });

    Ok(())
}
