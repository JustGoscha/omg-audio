//! The clip bank: imported recordings by id, decoded on background
//! threads so a level starts at once (Opus in pure Rust is ~10× real time;
//! a long ambience bed takes seconds). A clip id is valid immediately;
//! `get` returns None until it has decoded.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use omg_dsp::playlog::Clip;

use super::decode::{self, ClipOptions};

#[derive(Clone)]
enum Entry {
    Loading,
    Ready(Clip),
    Failed(String),
}

#[derive(Default)]
struct Bank {
    entries: Vec<Entry>,
    keys: HashMap<String, usize>,
}

fn bank() -> &'static (Mutex<Bank>, Condvar) {
    static B: OnceLock<(Mutex<Bank>, Condvar)> = OnceLock::new();
    B.get_or_init(|| (Mutex::new(Bank::default()), Condvar::new()))
}

/// Import options plus whether to RMS-normalise (sources' loops are
/// import-normalised to the engine's reference level, as the web's
/// eng_source_commit; fx clips keep their own calibration).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Import {
    pub opts: ClipOptions,
    pub normalize_rms: bool,
}

/// Where a clip's bytes come from: a file on disk, or bytes the host
/// already read (the web build reads res:// through Godot — there is no
/// file system for std::fs to see).
pub enum Source {
    Path(String),
    Bytes { name: String, bytes: Vec<u8> },
}

/// Queue a decode of `src` at engine rate `sr`; returns its id (the same
/// id for the same file + options).
pub fn load(src: Source, sr: f32, imp: Import) -> usize {
    let name = match &src {
        Source::Path(p) => p.as_str(),
        Source::Bytes { name, .. } => name.as_str(),
    };
    let key = format!("{name}|{sr}|{imp:?}");
    let (m, _) = bank();
    let id = {
        let mut b = m.lock().unwrap();
        if let Some(&id) = b.keys.get(&key) {
            return id;
        }
        b.entries.push(Entry::Loading);
        let id = b.entries.len() - 1;
        b.keys.insert(key, id);
        id
    };
    spawn_decode(Box::new(move || {
        let o = &imp.opts;
        let limit = if o.max_seconds > 0.0 { o.max_seconds * o.rate.max(1.0) + 3.0 } else { 0.0 };
        let pcm = match src {
            Source::Path(path) => decode::decode_file(&path, limit),
            Source::Bytes { name, bytes } => decode::decode_bytes(bytes, limit).map_err(|e| format!("{name}: {e}")),
        };
        let entry = match pcm {
            Ok(pcm) => {
                let mut x = decode::import(&pcm, sr, o);
                if imp.normalize_rms && !o.stereo {
                    omg_dsp::level::normalize_rms(&mut x, omg_dsp::level::REF_CLIP_RMS);
                }
                Entry::Ready(Arc::from(x))
            }
            Err(e) => Entry::Failed(e),
        };
        let (m, cv) = bank();
        m.lock().unwrap().entries[id] = entry;
        cv.notify_all();
    }));
    id
}

type Job = Box<dyn FnOnce() + Send>;

/// Natively: a thread per clip, all decoding at once.
#[cfg(not(target_os = "emscripten"))]
fn spawn_decode(job: Job) {
    std::thread::spawn(job);
}

/// Web: two decode workers fed from a queue. Every thread is a Web Worker
/// from the page's small pthread pool (the engine's sim and render
/// threads need theirs); a thread per clip would exhaust it.
#[cfg(target_os = "emscripten")]
fn spawn_decode(job: Job) {
    use std::sync::mpsc::{channel, Sender};
    static QUEUE: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();
    let q = QUEUE.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..2 {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                match next {
                    Ok(job) => job(),
                    Err(_) => return,
                }
            });
        }
        Mutex::new(tx)
    });
    let _ = q.lock().unwrap().send(job);
}

/// Add samples made elsewhere (synthesised in a script) as a clip.
pub fn add(samples: Vec<f32>) -> usize {
    let (m, _) = bank();
    let mut b = m.lock().unwrap();
    b.entries.push(Entry::Ready(Arc::from(samples)));
    b.entries.len() - 1
}

/// The clip, once decoded.
pub fn get(id: usize) -> Option<Clip> {
    match bank().0.lock().unwrap().entries.get(id) {
        Some(Entry::Ready(c)) => Some(c.clone()),
        _ => None,
    }
}

/// 0 loading, 1 ready, -1 failed / unknown id; with the error text.
pub fn state(id: usize) -> (i32, String) {
    match bank().0.lock().unwrap().entries.get(id) {
        Some(Entry::Ready(..)) => (1, String::new()),
        Some(Entry::Loading) => (0, String::new()),
        Some(Entry::Failed(e)) => (-1, e.clone()),
        None => (-1, "no such clip".into()),
    }
}

/// Block until nothing is loading (or `timeout_s` passes); true if done.
pub fn wait_all(timeout_s: f32) -> bool {
    let (m, cv) = bank();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f32(timeout_s.max(0.0));
    let mut b = m.lock().unwrap();
    loop {
        if !b.entries.iter().any(|e| matches!(e, Entry::Loading)) {
            return true;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return false;
        }
        b = cv.wait_timeout(b, deadline - now).unwrap().0;
    }
}
