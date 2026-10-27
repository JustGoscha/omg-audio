//! Clip import: decode an audio file to f32 at the engine rate, then shape
//! it with a few generic import options (peak / RMS normalisation,
//! high-pass, silence trim, length cap, fades, varispeed). These are the
//! operations the web demo applies in JavaScript on decode; here they are
//! plain options so a game can import its own clips the same way.
//!
//! Formats: Ogg Vorbis, Ogg FLAC, FLAC and WAV through symphonia; Ogg
//! Opus through `opus-decoder` (symphonia demuxes it, but has no codec).

use std::io::Cursor;

use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

/// Decoded audio: interleaved samples.
pub struct Pcm {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub rate: u32,
}

/// Decode a file; `limit_s` > 0 stops after that many seconds (the
/// pure-Rust Opus decoder runs ~10× real time, so a five-minute bed is
/// worth cutting short).
pub fn decode_file(path: &str, limit_s: f32) -> Result<Pcm, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    decode_bytes(bytes, limit_s).map_err(|e| format!("{path}: {e}"))
}

pub fn decode_bytes(bytes: Vec<u8>, limit_s: f32) -> Result<Pcm, String> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), MediaSourceStreamOptions::default());
    let mut format = symphonia::default::get_probe()
        .probe(&Hint::new(), mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| format!("probe: {e}"))?;
    let track = format.default_track(TrackType::Audio).ok_or("no audio track")?;
    let track_id = track.id;
    let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
        return Err("not an audio track".into());
    };
    let channels = params.channels.as_ref().map_or(1, |c| c.count()).max(1);
    let mut out = Vec::new();

    if params.codec == CODEC_ID_OPUS {
        // Opus always decodes at 48 kHz; the OpusHead pre-skip (encoder
        // delay) is dropped so loops start on their first real sample.
        let pre_skip = params
            .extra_data
            .as_ref()
            .filter(|h| h.len() >= 12 && &h[..8] == b"OpusHead")
            .map_or(312, |h| u16::from_le_bytes([h[10], h[11]]) as usize);
        let mut dec = opus_decoder::OpusDecoder::new(48_000, channels).map_err(|e| format!("opus: {e:?}"))?;
        let mut pcm = vec![0.0f32; 5760 * channels];
        let cap = if limit_s > 0.0 { (limit_s * 48_000.0) as usize * channels } else { usize::MAX };
        while let Ok(Some(packet)) = format.next_packet() {
            if out.len() >= cap {
                break;
            }
            if packet.track_id != track_id {
                continue;
            }
            match dec.decode_float(&packet.data, &mut pcm, false) {
                Ok(n) => out.extend_from_slice(&pcm[..n * channels]),
                Err(_) => continue,
            }
        }
        let skip = (pre_skip * channels).min(out.len());
        out.drain(..skip);
        return Ok(Pcm { samples: out, channels, rate: 48_000 });
    }

    let rate = params.sample_rate.unwrap_or(48_000);
    let mut dec = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("codec: {e}"))?;
    let mut tmp: Vec<f32> = Vec::new();
    let cap = if limit_s > 0.0 { (limit_s * rate as f32) as usize * channels } else { usize::MAX };
    while let Ok(Some(packet)) = format.next_packet() {
        if out.len() >= cap {
            break;
        }
        if packet.track_id != track_id {
            continue;
        }
        match dec.decode(&packet) {
            Ok(buf) => {
                tmp.clear();
                buf.copy_to_vec_interleaved(&mut tmp);
                out.extend_from_slice(&tmp);
            }
            Err(_) => continue,
        }
    }
    Ok(Pcm { samples: out, channels, rate })
}

/// Import options (all optional; defaults leave the clip untouched).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipOptions {
    /// Keep interleaved stereo (ambience beds) instead of taking channel 0.
    pub stereo: bool,
    /// Varispeed: play `rate`× faster (resampled; pitch moves with it).
    pub rate: f32,
    /// Normalise the peak to this level (0 = off).
    pub peak: f32,
    /// One-pole high-pass corner (Hz, 0 = off), applied after `peak`.
    pub highpass_hz: f32,
    /// Drop leading samples below this level (0 = off), keeping 10 ms.
    pub trim_start: f32,
    /// Drop trailing samples below this level (0 = off), keeping 10 ms.
    pub trim_end: f32,
    /// Keep at most this many seconds (0 = all).
    pub max_seconds: f32,
    /// Linear fade-out over the last seconds (0 = off).
    pub fade_out_s: f32,
    /// Linear fade-in over the first seconds (0 = off).
    pub fade_in_s: f32,
}

/// Linear resample of interleaved audio from `from` Hz to `to` Hz.
fn resample(x: &[f32], ch: usize, from: f32, to: f32) -> Vec<f32> {
    if (from - to).abs() < 0.5 || x.is_empty() {
        return x.to_vec();
    }
    let frames = x.len() / ch;
    let step = from as f64 / to as f64;
    let n = ((frames as f64) / step).floor() as usize;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let p = i as f64 * step;
        let j = p as usize;
        let f = (p - j as f64) as f32;
        let j1 = (j + 1).min(frames - 1);
        for c in 0..ch {
            out.push(x[j * ch + c] * (1.0 - f) + x[j1 * ch + c] * f);
        }
    }
    out
}

/// Decode + shape a clip for an engine running at `sr`.
pub fn import(pcm: &Pcm, sr: f32, o: &ClipOptions) -> Vec<f32> {
    let ch = pcm.channels.max(1);
    // channel pick: mono = channel 0 (as the web's getChannelData(0));
    // stereo = interleaved L/R, a mono file duplicated with a phase offset
    let mut x: Vec<f32> = if o.stereo {
        if ch >= 2 {
            pcm.samples.chunks(ch).flat_map(|f| [f[0], f[1]]).collect()
        } else {
            let n = pcm.samples.len();
            let off = n / 3;
            (0..n).flat_map(|i| [pcm.samples[i], pcm.samples[(i + off) % n.max(1)]]).collect()
        }
    } else {
        pcm.samples.chunks(ch).map(|f| f[0]).collect()
    };
    let och = if o.stereo { 2 } else { 1 };
    let rate = if o.rate > 0.0 { o.rate } else { 1.0 };
    x = resample(&x, och, pcm.rate as f32 * rate, sr);

    if o.peak > 0.0 {
        let p = x.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let g = o.peak / p;
        x.iter_mut().for_each(|v| *v *= g);
    }
    if o.highpass_hz > 0.0 && och == 1 {
        let k = (-2.0 * core::f32::consts::PI * o.highpass_hz / sr).exp();
        let mut lp = 0.0f32;
        for v in &mut x {
            lp += (1.0 - k) * (*v - lp);
            *v -= lp;
        }
    }
    let frames = x.len() / och;
    let level = |i: usize| (0..och).fold(0.0f32, |m, c| m.max(x[i * och + c].abs()));
    let keep = (0.010 * sr) as usize;
    let mut a = 0usize;
    if o.trim_start > 0.0 {
        while a < frames && level(a) < o.trim_start {
            a += 1;
        }
        a = a.saturating_sub(keep);
    }
    let mut b = frames;
    if o.trim_end > 0.0 {
        while b > a + 1 && level(b - 1) < o.trim_end {
            b -= 1;
        }
        b = (b + keep).min(frames);
    }
    if o.max_seconds > 0.0 {
        b = b.min(a + (o.max_seconds * sr) as usize);
    }
    let mut y = x[a * och..b * och].to_vec();
    let n = y.len() / och;
    let fi = ((o.fade_in_s * sr) as usize).min(n);
    for i in 0..fi {
        let g = i as f32 / fi as f32;
        for c in 0..och {
            y[i * och + c] *= g;
        }
    }
    let fo = ((o.fade_out_s * sr) as usize).min(n);
    for i in 0..fo {
        let g = i as f32 / fo as f32;
        for c in 0..och {
            y[(n - 1 - i) * och + c] *= g;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(p: &str) -> String {
        format!("{}/../../demo/assets/{p}", env!("CARGO_MANIFEST_DIR"))
    }

    /// Every container/codec the demo ships decodes to real audio.
    #[test]
    fn decodes_ogg_flac_vorbis_and_opus() {
        for (f, codec) in [
            ("birds/owl_0.ogg", "ogg flac"),
            ("steps/wood_0.ogg", "ogg vorbis"),
            ("motor048.ogg", "ogg opus"),
            ("night-nature48.ogg", "ogg opus stereo"),
        ] {
            let pcm = decode_file(&asset(f), 3.0).unwrap_or_else(|e| panic!("{codec}: {e}"));
            let peak = pcm.samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(pcm.samples.len() > 1000 && peak > 0.01, "{codec} ({f}) decoded silent");
            assert!(pcm.samples.len() <= (3.1 * pcm.rate as f32) as usize * pcm.channels, "limit ignored");
            assert!(peak <= 1.5, "{codec} decoded out of range: {peak}");
        }
    }

    #[test]
    fn shaping_options_apply() {
        let pcm = Pcm { samples: [vec![0.0; 4800], vec![0.5; 4800], vec![0.0; 4800]].concat(), channels: 1, rate: 48_000 };
        let o = ClipOptions { peak: 0.25, trim_start: 0.01, trim_end: 0.01, ..Default::default() };
        let y = import(&pcm, 48_000.0, &o);
        assert!(y.len() < 4800 + 2 * 480 + 2, "silence not trimmed: {}", y.len());
        assert!((y.iter().fold(0.0f32, |m, v| m.max(*v)) - 0.25).abs() < 1e-4);
        let half = import(&pcm, 24_000.0, &ClipOptions::default());
        assert_eq!(half.len(), 7200);
    }
}
