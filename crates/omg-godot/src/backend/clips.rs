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

/// Queue a decode of `path` at engine rate `sr`; returns its id (the same
/// id for the same path + options).
pub fn load(path: &str, sr: f32, imp: Import) -> usize {
    let key = format!("{path}|{sr}|{imp:?}");
    let (m, cv) = bank();
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
    let path = path.to_string();
    std::thread::spawn(move || {
        let o = &imp.opts;
        let limit = if o.max_seconds > 0.0 { o.max_seconds * o.rate.max(1.0) + 3.0 } else { 0.0 };
        let entry = match decode::decode_file(&path, limit) {
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
    });
    let _ = cv;
    id
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
