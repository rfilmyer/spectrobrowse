use std::{fmt::{Debug}, iter::Sum, path::Path};
use std::io::BufReader;
use std::fs::File;
use ndarray::{Array, Array2, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use plotters::{coord::Shift, prelude::*};
use rodio::{self, Decoder, source::Spatial};
use rustfft::{FftPlanner, num_complex::Complex, num_traits::{FromPrimitive, Num}};
use colorous;

use eyre::Result;

pub fn load_soundfile_from_path<P: AsRef<Path>>(path: P) -> Result<Vec<i16>, eyre::Report> {
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

pub fn compute_spectrogram(waveform: Vec<i16>, window_size: usize, overlap: f64) -> Result<Array2<f32>, eyre::Report> {
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
    let mut scratch_buffer = vec![Complex::new(0f32, 0f32); window_size];
    windows.axis_iter_mut(Axis(0))
        .for_each(|mut frame| { 
            if let Some(buffer) = frame.as_slice_mut() {
                fft.process_with_scratch(buffer, scratch_buffer.as_mut_slice()); 
            };
        });
    
    // Get the real component of those complex numbers we get back from the FFT
    let windows = windows.map(|i| i.re);

    // And finally, only look at the first half of the spectrogram - the first (n/2)+1 points of each FFT
    // https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
    let windows = windows.slice_move(ndarray::s![.., ..((window_size / 2) + 1)]);
    let windows_shape = windows.shape();

    let log_transformed = windows.axis_iter(Axis(0))
        .map(log_distort_vector)
        .flatten()
        .collect::<Vec<_>>();
    
    let log_transformed = Array::from_shape_vec(windows_shape, log_transformed)?;
    let log_transformed = log_transformed.into_dimensionality()?;
    Ok(log_transformed)
}

fn log_distort_vector<'a, T: Num + Sum<&'a T> + FromPrimitive + Copy + Debug>(input: ArrayView1<'a, T>) -> Vec<T> {
    let length = input.len();
    let log_length = f32::log2(length as f32);
    let log_bin_width = log_length / length as f32;

    let log_transformed = (1..(length + 1))
        .map(|i| {
            let lower = log_bin_width * (i as f32 - 1f32);
            let lower = lower
                .exp2()
                .floor() as usize;
            let higher = log_bin_width * (i as f32);
            let higher = higher
                .exp2()
                .floor() as usize;

            // eprintln!("slicing between {} (= {:?}) and {} (= {:?}) on an array {} long", lower, input.get(lower), higher, input.get(higher), length);
            let average_spectral_density = input.slice(ndarray::s![lower..higher])
                .mean()
                .unwrap_or_else(|| {*input.get(lower).expect(format!("index {} out of bounds for array with length {}", lower, length).as_str())})
                ;
            average_spectral_density
        })
        .collect::<Vec<_>>();

    log_transformed
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
    println!("...from a spectrogram with {} samples x {} frequency bins.", num_samples, num_freq_bins);

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
    let windows_scaled = windows_flipped.map(|x| { x.sqrt() / highest_spectral_density.sqrt() });

    eprintln!("Splitting drawing area...");

    let spectrogram_cells = drawing_area.split_evenly((num_freq_bins, num_samples));

    // Finally add a color scale
    let color_scale = colorous::MAGMA;

    eprintln!("Drawing Cells");

    for (cell, &spectral_density) in spectrogram_cells.iter().zip(windows_scaled.iter()) {
            let color = color_scale.eval_continuous(spectral_density as f64);
            cell.fill(&RGBColor(color.r, color.g, color.b)).unwrap();
        };
}