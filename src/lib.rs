use std::path::Path;
use std::io::BufReader;
use std::fs::File;
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;
use plotters::{coord::Shift, prelude::*};
use rodio::{self, Decoder, source::Spatial};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{error::Error};
use colorous;

pub fn load_soundfile_from_path<P: AsRef<Path>>(path: P) -> Result<Vec<i16>, Box<dyn Error>> {
    let file = File::open(path)?;

    let source = Decoder::new(BufReader::new(file))?;

    let spatial_filter = Spatial::new(
        source,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    );

    Ok(spatial_filter.into_iter().collect())

}

pub fn compute_spectrogram(waveform: Vec<i16>, window_size: usize, overlap: f64) -> Result<Array2<f32>, Box<dyn Error>> {
    let skip_size = (window_size as f64 * (1f64 - overlap)) as usize;

    let waveform = Array::from(waveform);
    let windows = waveform
    .windows(ndarray::Dim(window_size))
    .into_iter()
    .step_by(skip_size)
    .collect::<Vec<_>>()
    ;
    let windows = ndarray::stack(Axis(0), &windows)?;
    
    // So to perform the FFT on each window we need a Complex<f32>, and right now we have i16s, so first let's convert
    let mut windows = windows.map(|i| Complex::from(*i as f32));

    // get the FFT up and running
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    // Since we have a 2-D array of our windows with shape [window_size, (num_samples / window_size) - 1], we can run an FFT on every row.
    windows.axis_iter_mut(Axis(0))
        .for_each(|mut frame| { fft.process(frame.as_slice_mut().unwrap()); });
    
    // Get the real component of those complex numbers we get back from the FFT
    let windows = windows.map(|i| i.re);

    // And finally, only look at the first half of the spectrogram - the first (n/2)+1 points of each FFT
    // https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
    let windows = windows.slice_move(ndarray::s![.., ..((window_size / 2) + 1)]);

    Ok(windows)
}

pub fn plot_spectrogram<DB: DrawingBackend>(spectrogram: &Array2<f32>, drawing_area: &DrawingArea<DB, Shift>) {
    // let mut chart = ChartBuilder::on(drawing_area);

    // get some dimensions for drawing
    // The shape is in [nrows, ncols], meaning [n_samples, n_freqbins], but we want to transpose this
    // so that in our graph, x = time, y = frequency.
    let (num_samples, num_freq_bins) = match spectrogram.shape() {
        &[num_rows, num_columns] => (num_rows, num_columns),
        _ => panic!("Spectrogram is a {}D array, expected a 2D array.
                     This should never happen (should not be possible to call function with anything but a 2d array)", spectrogram.ndim())
    };

    println!("Generating a {} wide x {} high image", num_samples, num_freq_bins);

    let spectrogram_cells = drawing_area.split_evenly((num_freq_bins, num_samples));

    // Scaling values
    let windows_scaled = spectrogram.map(|i| i.abs()/(num_freq_bins as f32));
    let highest_spectral_density = windows_scaled.max_skipnan();
    
    // transpose and flip around to prepare for graphing
    /* the array is currently oriented like this:
        t = 0 |
              |
              |
              |
              |
        t = n +-------------------
            f = 0              f = m

        so it needs to be flipped...
        t = 0 |
              |
              |
              |
              |
        t = n +-------------------
            f = m              f = 0

        ...and transposed...
        f = m |
              |
              |
              |
              |
        f = 0 +-------------------
            t = 0              t = n
        
        ... in order to look like a proper spectrogram
    */
    let windows_flipped = windows_scaled.slice(ndarray::s![.., ..; -1]); // flips the
    let windows_flipped = windows_flipped.t();

    // Finally add a color scale
    let color_scale = colorous::MAGMA;

    for (cell, spectral_density) in spectrogram_cells.iter().zip(windows_flipped.iter()) {
            let spectral_density_scaled = spectral_density.sqrt() / highest_spectral_density.sqrt();
            let color = color_scale.eval_continuous(spectral_density_scaled as f64);
            cell.fill(&RGBColor(color.r, color.g, color.b)).unwrap();
        };
}