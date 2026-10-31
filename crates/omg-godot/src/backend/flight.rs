//! The flight recorder: the last minute of what Godot played, at hand
//! whenever the player marks a moment. The audio thread copies every
//! frame it takes from the engine into a ring of its own (`push`, beside
//! `record::push`); the game freezes it (`freeze`) when the player marks
//! a moment and reads it back while nothing is written — a WAV of any
//! stretch (`save`), its spectrogram (`spectrogram`). Frames are numbered
//! from the first one kept (`frames`): the clock the game's own log of the
//! same minute (where the ears were, the sources, the paths) is stamped
//! with.

use std::cell::UnsafeCell;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use omg_dsp::fft::{RealFft, C};

/// Seconds kept: the minute asked for.
pub const KEEP_S: f64 = 60.0;
/// And a few more the audio thread may be writing while the game reads
/// the minute (a push already under way when it froze).
const SLACK_S: f64 = 4.0;

struct Flight {
    buf: Box<[UnsafeCell<(f32, f32)>]>,
    rate: u32,
    written: AtomicU64,
    on: AtomicBool,
}

unsafe impl Sync for Flight {}

static FLIGHT: OnceLock<Flight> = OnceLock::new();

/// Keep the minute (again) at `rate` Hz; the first call sizes the ring.
pub fn start(rate: u32) {
    let f = FLIGHT.get_or_init(|| {
        let cap = ((KEEP_S + SLACK_S) * rate as f64) as usize;
        Flight { buf: (0..cap).map(|_| UnsafeCell::new((0.0, 0.0))).collect(), rate, written: AtomicU64::new(0), on: AtomicBool::new(false) }
    });
    f.on.store(true, Ordering::Release);
}

/// Stop writing: the minute holds still to be read.
pub fn freeze() {
    if let Some(f) = FLIGHT.get() {
        f.on.store(false, Ordering::Release);
    }
}

/// Frames kept so far (the minute ends here).
pub fn frames() -> u64 {
    FLIGHT.get().map_or(0, |f| f.written.load(Ordering::Acquire))
}

pub fn rate() -> u32 {
    FLIGHT.get().map_or(0, |f| f.rate)
}

/// The audio thread: the frames it just played.
pub fn push(xs: &[(f32, f32)]) {
    let Some(f) = FLIGHT.get() else { return };
    if !f.on.load(Ordering::Acquire) {
        return;
    }
    let cap = f.buf.len() as u64;
    let w = f.written.load(Ordering::Relaxed);
    for (i, x) in xs.iter().enumerate() {
        unsafe { *f.buf[((w + i as u64) % cap) as usize].get() = *x };
    }
    f.written.store(w + xs.len() as u64, Ordering::Release);
}

/// Frames `from..to` that are still kept (the last `KEEP_S` s).
fn span(from: u64, to: u64) -> (u64, u64) {
    let (Some(f), end) = (FLIGHT.get(), frames()) else { return (0, 0) };
    let first = end.saturating_sub((KEEP_S * f.rate as f64) as u64);
    let a = from.clamp(first, end);
    (a, to.clamp(a, end))
}

fn at(f: &Flight, i: u64) -> (f32, f32) {
    unsafe { *f.buf[(i % f.buf.len() as u64) as usize].get() }
}

/// Frames `from..to` (clamped to what is kept) as a float stereo WAV.
pub fn save(path: &str, from: u64, to: u64) -> std::io::Result<u64> {
    let Some(f) = FLIGHT.get() else { return Ok(0) };
    let (a, b) = span(from, to);
    let n = b - a;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    let bytes = (n * 8) as u32;
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + bytes).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&3u16.to_le_bytes())?; // IEEE float
    w.write_all(&2u16.to_le_bytes())?;
    w.write_all(&f.rate.to_le_bytes())?;
    w.write_all(&(f.rate * 8).to_le_bytes())?;
    w.write_all(&8u16.to_le_bytes())?;
    w.write_all(&32u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&bytes.to_le_bytes())?;
    for i in a..b {
        let (l, r) = at(f, i);
        w.write_all(&l.to_le_bytes())?;
        w.write_all(&r.to_le_bytes())?;
    }
    w.flush()?;
    Ok(n)
}

/// The spectrogram of frames `from..to` as an RGBA8 image `width` wide:
/// the left ear's `bands` log bands (40 Hz – 16 kHz, low at the bottom)
/// over the right's, 4 rows apart — each column the loudest of its bands
/// in a 4096-point Hann window (85 ms at 48 kHz) at the column's time,
/// coloured as the overlay's live one (inferno, −100 … −30 dB of a
/// sinusoid's amplitude). Height: 2 · bands + 4.
pub fn spectrogram(from: u64, to: u64, width: usize, bands: usize) -> Vec<u8> {
    const N: usize = 4096;
    const GAP: usize = 4;
    let h = 2 * bands + GAP;
    let mut img = vec![0u8; width * h * 4];
    let Some(f) = FLIGHT.get() else { return img };
    let (a, b) = span(from, to);
    if b <= a || width == 0 || bands == 0 {
        return img;
    }
    let lut = inferno();
    let win: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (2.0 * core::f32::consts::PI * i as f32 / N as f32).cos()).collect();
    let norm = 2.0 / win.iter().sum::<f32>();
    let (lo, hi) = (40.0f32, 16000.0f32);
    let bin_hz = f.rate as f32 / N as f32;
    // each band's bins (at least the nearest one)
    let edges: Vec<(usize, usize)> = (0..bands)
        .map(|i| {
            let f0 = lo * (hi / lo).powf(i as f32 / bands as f32);
            let f1 = lo * (hi / lo).powf((i + 1) as f32 / bands as f32);
            let k0 = (f0 / bin_hz).round() as usize;
            let k1 = ((f1 / bin_hz).round() as usize).max(k0 + 1);
            (k0.min(N / 2), k1.min(N / 2 + 1))
        })
        .collect();
    let mut fft = RealFft::new(N);
    let mut x = vec![0.0f32; N];
    let mut spec = vec![C::default(); N / 2 + 1];
    for col in 0..width {
        let centre = a as f64 + (col as f64 + 0.5) * (b - a) as f64 / width as f64;
        let start = centre as i64 - (N / 2) as i64;
        for ear in 0..2 {
            for (i, v) in x.iter_mut().enumerate() {
                let k = start + i as i64;
                *v = if k >= a as i64 && k < b as i64 {
                    let s = at(f, k as u64);
                    (if ear == 0 { s.0 } else { s.1 }) * win[i]
                } else {
                    0.0
                };
            }
            fft.forward(&x, &mut spec);
            for (bi, &(k0, k1)) in edges.iter().enumerate() {
                let mut m = 0.0f32;
                for c in &spec[k0..k1] {
                    m = m.max((c.re * c.re + c.im * c.im).sqrt());
                }
                let db = 20.0 * (m * norm + 1e-9).log10();
                let t = ((db + 100.0) / 70.0).clamp(0.0, 1.0).powf(1.25);
                let rgb = lut[(t * 255.0).round() as usize];
                let row = ear * (bands + GAP) + (bands - 1 - bi);
                let p = (row * width + col) * 4;
                img[p..p + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
    }
    // the gap between the ears: dark grey
    for row in bands..bands + GAP {
        for col in 0..width {
            let p = (row * width + col) * 4;
            img[p..p + 4].copy_from_slice(&[40, 40, 48, 255]);
        }
    }
    img
}

/// Inferno (black → purple → crimson → orange → yellow), the overlay's
/// and the web's stops, as 256 colours.
fn inferno() -> Vec<[u8; 3]> {
    const S: [(f32, [f32; 3]); 10] = [
        (0.0, [0.0, 0.0, 4.0]),
        (0.125, [22.0, 11.0, 57.0]),
        (0.25, [66.0, 10.0, 104.0]),
        (0.375, [106.0, 23.0, 110.0]),
        (0.5, [147.0, 38.0, 103.0]),
        (0.625, [188.0, 55.0, 84.0]),
        (0.75, [221.0, 81.0, 58.0]),
        (0.875, [243.0, 120.0, 25.0]),
        (0.9375, [249.0, 189.0, 38.0]),
        (1.0, [252.0, 255.0, 164.0]),
    ];
    (0..256)
        .map(|i| {
            let x = i as f32 / 255.0;
            let k = (0..S.len() - 1).find(|&k| S[k + 1].0 >= x).unwrap_or(S.len() - 2);
            let u = (x - S[k].0) / (S[k + 1].0 - S[k].0);
            core::array::from_fn(|c| (S[k].1[c] + u * (S[k + 1].1[c] - S[k].1[c])).round() as u8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 kHz tone in the left ear comes back where it was played: the
    /// brightest band of the left half is the one holding 1 kHz, the right
    /// half stays dark, and a stretch of it saves as that many frames.
    #[test]
    fn the_minute_reads_back_what_played() {
        start(48_000);
        let f0 = frames();
        let tone: Vec<(f32, f32)> = (0..48_000).map(|i| ((2.0 * core::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.5, 0.0)).collect();
        for c in tone.chunks(512) {
            push(c);
        }
        freeze();
        let f1 = frames();
        assert_eq!(f1 - f0, 48_000);
        let bands = 64;
        let img = spectrogram(f0, f1, 8, bands);
        let lum = |row: usize, col: usize| {
            let p = (row * 8 + col) * 4;
            img[p] as u32 + img[p + 1] as u32 + img[p + 2] as u32
        };
        let brightest = (0..bands).max_by_key(|&r| lum(r, 4)).unwrap();
        // row r holds band bands − 1 − r: its range 40 Hz · 400^(i/bands)
        let band = bands - 1 - brightest;
        let (lo, hi) = (40.0 * 400f32.powf(band as f32 / bands as f32), 40.0 * 400f32.powf((band + 1) as f32 / bands as f32));
        assert!(lo <= 1000.0 && 1000.0 < hi, "brightest band {band}: {lo:.0}–{hi:.0} Hz");
        assert!((0..bands).all(|r| lum(bands + 4 + r, 4) < 30), "the right ear lit up");
        let path = std::env::temp_dir().join("omg_flight_test.wav");
        let n = save(path.to_str().unwrap(), f0 + 1000, f0 + 25_000).unwrap();
        assert_eq!(n, 24_000);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 24_000 * 8);
        let _ = std::fs::remove_file(path);
    }
}
