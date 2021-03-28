use colorous;
use eyre::Result;
use image::{self, DynamicImage, GrayImage, RgbImage};
use ndarray::{Array, Array2, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use rodio::{self, source::Spatial, Decoder};
use rustfft::{
    num_complex::Complex,
    num_traits::{FromPrimitive, Num},
    FftPlanner,
};
use std::fs::File;
use std::io::BufReader;
use std::{fmt::Debug, iter::Sum, path::Path};

/// Loads an audio file as a waveform and mixes down to mono if necessary.
pub fn load_soundfile_from_path<P: AsRef<Path>>(path: P) -> Result<Vec<i16>, eyre::Report> {
    let file = File::open(path)?;

    let source = Decoder::new(BufReader::new(file))?;

    let spatial_filter = Spatial::new(source, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);

    Ok(spatial_filter.into_iter().collect())
}

/// Computes a spectrogram from a waveform and "log-transforms" the spectrogram frequencies, if desired
pub fn compute_spectrogram(
    waveform: Vec<i16>,
    window_size: usize,
    overlap: f64,
    log_transform: bool,
) -> Result<Array2<f32>, eyre::Report> {
    let skip_size = (window_size as f64 * (1f64 - overlap)) as usize;

    let waveform = Array::from(waveform);
    let windows = waveform
        .windows(ndarray::Dim(window_size))
        .into_iter()
        .step_by(skip_size)
        .collect::<Vec<_>>();
    let windows = ndarray::stack(Axis(0), &windows)?;

    // So to perform the FFT on each window we need a Complex<f32>, and right now we have i16s, so first let's convert
    let mut windows = windows.map(|i| Complex::from(*i as f32));

    // get the FFT up and running
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    // Since we have a 2-D array of our windows with shape [window_size, (num_samples / window_size) - 1], we can run an FFT on every row.
    let mut scratch_buffer = vec![Complex::new(0f32, 0f32); window_size];
    windows.axis_iter_mut(Axis(0)).for_each(|mut frame| {
        if let Some(buffer) = frame.as_slice_mut() {
            fft.process_with_scratch(buffer, scratch_buffer.as_mut_slice());
        };
    });

    // Get the real component of those complex numbers we get back from the FFT
    let windows = windows.map(|i| i.re);

    // And finally, only look at the first half of the spectrogram - the first (n/2)+1 points of each FFT
    // https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
    let windows = windows.slice_move(ndarray::s![.., ..((window_size / 2) + 1)]);

    if log_transform {
        let windows_shape = windows.shape();
        let log_transformed = windows
            .axis_iter(Axis(0))
            .map(log_distort_vector)
            .flatten()
            .collect::<Vec<_>>();

        let log_transformed = Array::from_shape_vec(windows_shape, log_transformed)?;
        let log_transformed = log_transformed.into_dimensionality()?;
        Ok(log_transformed)
    } else {
        Ok(windows)
    }
}

/// The log-distortion function, effectively changing the frequency windows from linear to log scale.
/// The API for this currently is not great - it returns a new `Vec<T>` that must be then reassembled into an array.
/// Actual testing shows that this is not close to being the constraint, though, so I'm not going to try to make it better.
fn log_distort_vector<'a, T: Num + Sum<&'a T> + FromPrimitive + Copy + Debug>(
    input: ArrayView1<'a, T>,
) -> Vec<T> {
    let length = input.len();
    let log_length = f32::log2(length as f32);
    let log_bin_width = log_length / length as f32;

    let log_transformed = (1..(length + 1))
        .map(|i| {
            let lower = log_bin_width * (i as f32 - 1f32);
            let lower = lower.exp2().floor() as usize;
            let higher = log_bin_width * (i as f32);
            let higher = higher.exp2().floor() as usize;

            // eprintln!("slicing between {} (= {:?}) and {} (= {:?}) on an array {} long", lower, input.get(lower), higher, input.get(higher), length);
            let average_spectral_density = input
                .slice(ndarray::s![lower..higher])
                .mean()
                .unwrap_or_else(|| {
                    *input.get(lower).expect(
                        format!(
                            "index {} out of bounds for array with length {}",
                            lower, length
                        )
                        .as_str(),
                    )
                });
            average_spectral_density
        })
        .collect::<Vec<_>>();

    log_transformed
}

/// Renders a spectrogram as an RGB image, resizing into a `width` x `height` image.
pub fn render_spectrogram(spectrogram: &Array2<f32>, width: u32, height: u32) -> RgbImage {
    // get some dimensions for drawing
    // The shape is in [nrows, ncols], meaning [n_samples, n_freqbins], but we want to transpose this
    // so that in our graph, x = time, y = frequency.
    let (num_samples, num_freq_bins) = match spectrogram.shape() {
        &[num_rows, num_columns] => (num_rows, num_columns),
        _ => panic!("Spectrogram is a {}D array, expected a 2D array.
                     This should never happen (should not be possible to call function with anything but a 2d array)", spectrogram.ndim())
    };
    println!(
        "...from a spectrogram with {} samples x {} frequency bins.",
        num_samples, num_freq_bins
    );

    // Scaling values
    let windows_scaled = spectrogram.map(|i| i.abs() / (num_freq_bins as f32));
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
    println!("Vector transformation operations...");
    let windows_flipped = windows_scaled.slice(ndarray::s![.., ..; -1]); // flips the
    let windows_flipped = windows_flipped.t();
    let windows_scaled = windows_flipped.map(|x| x.sqrt() / highest_spectral_density.sqrt()); // values are now between 0 and 1

    println!("Resizing spectrogram...");
    let image_buffer = GrayImage::from_raw(
        num_samples as u32,
        num_freq_bins as u32,
        windows_scaled
            .iter()
            .map(|x| (x * u8::MAX as f32).round() as u8) // Convert [0.0, 1.0] ratio to [0-256] number
            .collect(),
    )
    .unwrap();

    let image_buffer = image::imageops::resize(
        &image_buffer,
        width,
        height,
        image::imageops::FilterType::CatmullRom,
    );

    // Finally add a color scale
    let color_scale = colorous::MAGMA;

    let mut image_buffer = DynamicImage::ImageLuma8(image_buffer).to_rgb8();

    eprintln!("Drawing image...");
    image_buffer.pixels_mut().for_each(|x| {
        *x = image::Rgb(
            color_scale
                .eval_rational(x.0[0] as usize, u8::MAX as usize)
                .as_array(),
        )
    });

    image_buffer
}
