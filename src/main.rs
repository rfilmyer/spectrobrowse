mod lib;

use std::{error::Error};

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let waveform =  lib::load_soundfile_from_path("sound_files_nodistrib/Roundstart_MAIN-sharedassets2.assets-54.wav")?;
    let spectrogram = lib::compute_spectrogram(waveform, 2048, 0.75)?;

    let (num_samples, num_freq_bins) = match spectrogram.shape() {
        &[num_rows, num_columns] => (num_rows, num_columns),
        _ => panic!("Windows is a {}D array, expected a 2D array", spectrogram.ndim())
    };
    println!("Generating a {} wide x {} high image", num_samples, num_freq_bins);
    let image_dimensions: (u32, u32) = (num_samples as u32, num_freq_bins as u32);
    let root = 
        BitMapBackend::new(
            "output/Roundstart_MAIN-sharedassets2.assets-54.png", 
            image_dimensions, // width x height. Worth it if we ever want to resize the graph.
        ).into_drawing_area();

    lib::plot_spectrogram(&spectrogram, &root);

    Ok(())
}
