//! A recording of what Godot plays: every frame its audio thread takes from
//! the engine (`OmgOutputPlayback::mix`), written to a WAV by a thread of
//! its own, and each callback's timing beside it in a CSV — when it ran,
//! how many frames, how long the take took, how many frames the output
//! thread hadn't rendered in time (rendered late, on the audio thread).
//! The audio thread only copies into a lock-free ring; nothing it does
//! waits on the disk. What the headless tools measure is the engine; this
//! is the engine as the game plays it.

use std::cell::UnsafeCell;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

/// One audio callback.
#[derive(Clone, Copy, Default)]
struct Callback {
    /// Since the recording started (µs).
    at_us: u64,
    frames: u32,
    /// How long the take from the engine took (µs).
    take_us: u32,
    /// Frames the output thread hadn't rendered: rendered in the take.
    late: u32,
}

/// A single-producer single-consumer ring (the audio thread writes, the
/// writer thread reads).
struct Ring<T: Copy + Default> {
    buf: Box<[UnsafeCell<T>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Copy + Default> Sync for Ring<T> {}

impl<T: Copy + Default> Ring<T> {
    fn new(cap: usize) -> Self {
        Self { buf: (0..cap).map(|_| UnsafeCell::new(T::default())).collect(), head: AtomicUsize::new(0), tail: AtomicUsize::new(0) }
    }

    /// Producer: as many of `xs` as fit; how many did.
    fn push(&self, xs: &[T]) -> usize {
        let cap = self.buf.len();
        let (h, t) = (self.head.load(Ordering::Relaxed), self.tail.load(Ordering::Acquire));
        let n = xs.len().min(cap - (h - t));
        for (i, x) in xs[..n].iter().enumerate() {
            unsafe { *self.buf[(h + i) % cap].get() = *x };
        }
        self.head.store(h + n, Ordering::Release);
        n
    }

    /// Consumer: everything pushed so far, onto `out`.
    fn drain(&self, out: &mut Vec<T>) {
        let cap = self.buf.len();
        let (h, t) = (self.head.load(Ordering::Acquire), self.tail.load(Ordering::Relaxed));
        for i in t..h {
            out.push(unsafe { *self.buf[i % cap].get() });
        }
        self.tail.store(h, Ordering::Release);
    }
}

struct Recorder {
    frames: Ring<(f32, f32)>,
    calls: Ring<Callback>,
    active: AtomicBool,
    /// Frames that didn't fit the ring (the writer fell behind).
    dropped: AtomicUsize,
    start: Mutex<Option<Instant>>,
    writer: Mutex<Option<JoinHandle<()>>>,
}

/// ~10 s of stereo at 48 kHz between two writer passes is plenty.
const FRAMES: usize = 1 << 19;
const CALLS: usize = 1 << 14;

fn recorder() -> &'static Recorder {
    static R: OnceLock<Recorder> = OnceLock::new();
    R.get_or_init(|| Recorder {
        frames: Ring::new(FRAMES),
        calls: Ring::new(CALLS),
        active: AtomicBool::new(false),
        dropped: AtomicUsize::new(0),
        start: Mutex::new(None),
        writer: Mutex::new(None),
    })
}

/// Start recording to `path` (a .wav; the callbacks go to `path`.csv).
/// A recording already running is stopped first.
pub fn start(path: &str, sample_rate: u32) -> std::io::Result<()> {
    stop();
    let r = recorder();
    // what is in the rings from before is not this recording's
    r.frames.drain(&mut Vec::new());
    r.calls.drain(&mut Vec::new());
    r.dropped.store(0, Ordering::Relaxed);
    let mut wav = std::fs::File::create(path)?;
    write_header(&mut wav, sample_rate, 0)?;
    let mut csv = std::io::BufWriter::new(std::fs::File::create(format!("{path}.csv"))?);
    writeln!(csv, "at_ms,frames,take_us,late")?;
    *r.start.lock().unwrap() = Some(Instant::now());
    r.active.store(true, Ordering::Release);
    let handle = std::thread::Builder::new().name("omg-record".into()).spawn(move || {
        let r = recorder();
        let (mut fr, mut cl) = (Vec::new(), Vec::new());
        let mut bytes: u32 = 0;
        loop {
            let running = r.active.load(Ordering::Acquire);
            fr.clear();
            cl.clear();
            r.frames.drain(&mut fr);
            r.calls.drain(&mut cl);
            let mut buf = Vec::with_capacity(fr.len() * 8);
            for (a, b) in &fr {
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
            }
            let _ = wav.write_all(&buf);
            bytes = bytes.saturating_add(buf.len() as u32);
            for c in &cl {
                let _ = writeln!(csv, "{:.3},{},{},{}", c.at_us as f64 / 1000.0, c.frames, c.take_us, c.late);
            }
            if !running {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = csv.flush();
        let _ = wav.seek(SeekFrom::Start(0));
        let _ = write_header(&mut wav, sample_rate, bytes);
    })?;
    *r.writer.lock().unwrap() = Some(handle);
    Ok(())
}

/// Stop and finish the files (the WAV's sizes written). Frames dropped
/// because the writer fell behind: their count (0 normally).
pub fn stop() -> usize {
    let r = recorder();
    r.active.store(false, Ordering::Release);
    if let Some(h) = r.writer.lock().unwrap().take() {
        let _ = h.join();
    }
    r.dropped.load(Ordering::Relaxed)
}

pub fn recording() -> bool {
    recorder().active.load(Ordering::Acquire)
}

/// The audio thread: the frames it just played, how long taking them
/// took, and how many of them were rendered late.
pub fn push(frames: &[(f32, f32)], take_us: u32, late: u32) {
    let r = recorder();
    if !r.active.load(Ordering::Acquire) {
        return;
    }
    let at_us = r.start.try_lock().ok().and_then(|s| s.map(|s| s.elapsed().as_micros() as u64)).unwrap_or(0);
    let n = r.frames.push(frames);
    if n < frames.len() {
        r.dropped.fetch_add(frames.len() - n, Ordering::Relaxed);
    }
    r.calls.push(&[Callback { at_us, frames: frames.len() as u32, take_us, late }]);
}

/// A float-32 stereo WAV header for `bytes` of data.
fn write_header(w: &mut impl Write, sample_rate: u32, bytes: u32) -> std::io::Result<()> {
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + bytes).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&3u16.to_le_bytes())?; // IEEE float
    w.write_all(&2u16.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 8).to_le_bytes())?;
    w.write_all(&8u16.to_le_bytes())?;
    w.write_all(&32u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&bytes.to_le_bytes())?;
    Ok(())
}
