use std::{error::Error, fs::File};
use std::io::BufReader;
use rodio::{self, Source};
use sonogram::SpecOptionsBuilder;



fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("Aphex Twin - Windowlicker.mp3")?;

    let source = rodio::Decoder::new(BufReader::new(file))?;
    println!("Read file");

    let source = rodio::source::Spatial::new(
        source,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    );

    let sample_rate = source.sample_rate();

    let waveform: Vec<i16> = source.into_iter().collect();
    println!("Converted file into waveform");
    
    let mut spectrograph = SpecOptionsBuilder::new(1024 * 8,256)
        .load_data_from_memory(waveform, sample_rate)
        .build();

    spectrograph.compute(2048, 0.8);
    println!("Computed spectrograph");

    let png_file = std::path::Path::new("windowlicker.png");
    spectrograph.save_as_png(&png_file, true)?;
    println!("Saved as PNG");

    Ok(())
}
