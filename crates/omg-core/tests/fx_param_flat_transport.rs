//! Effect: the simulation → renderer transport — `ParamBlock::write_flat`
//! / `ParamBlock::read_flat`, the flat f32 layout every ParamBlock crosses
//! the worker / SharedArrayBuffer boundary in (omg-web).
//!
//! Physical claim guaranteed: the transport is LOSSLESS for everything
//! the renderer consumes — tap identity (key), delay, direction, per-band
//! gains, late reverb (rt60, level), and the coupled-room remote reverb
//! (present or absent) — bit-exact, in tap order; a reused buffer never
//! leaks stale data from a previous, larger block; the documented layout
//! (header 24, 9 floats per tap) holds.
//!
//! Combined in the demo: omg-web `tick` writes every source's block with
//! `write_flat` and the audio worklet side rebuilds it with `read_flat`
//! — the whole web demo hears the world only through this.

use omg_core::params::{ParamBlock, RemoteReverb, ReverbParams, Tap, FLAT_HEADER, FLAT_PER_TAP};
use omg_core::rng::Rng;
use omg_core::NBANDS;

fn tap(rng: &mut Rng, key: u32) -> Tap {
    let mut r = || rng.next_f32();
    Tap {
        key,
        delay_s: r() * 0.5,
        dir: [r() * 2.0 - 1.0, r() * 2.0 - 1.0, r() * 2.0 - 1.0],
        // span many decades, including denormal-ish small gains
        gains: [r(), r() * 1e-3, r() * 1e-7],
    }
}

fn block(n_taps: usize, remote: bool, version: u64, seed: u64) -> ParamBlock {
    let mut rng = Rng::new(seed);
    let taps = (0..n_taps).map(|i| tap(&mut rng, (i as u32) * 37 + 5)).collect();
    let reverb = ReverbParams { rt60: [1.7, 1.2, 0.8], level: [0.31, 0.22, 0.05] };
    let remote = remote.then(|| RemoteReverb {
        rt60: [2.5, 2.0, 1.1],
        send: [0.12, 0.06, 0.01],
        sh: core::array::from_fn(|i| (i as f32 - 4.0) * 0.123),
    });
    ParamBlock { taps, reverb, remote, version, onset_s: 0.0137 * seed as f32 }
}

fn bits3(a: [f32; NBANDS]) -> [u32; NBANDS] {
    a.map(f32::to_bits)
}

fn assert_same(a: &ParamBlock, b: &ParamBlock) {
    assert_eq!(a.version, b.version, "version");
    assert_eq!(a.onset_s.to_bits(), b.onset_s.to_bits(), "onset");
    assert_eq!(a.taps.len(), b.taps.len(), "tap count");
    for (i, (x, y)) in a.taps.iter().zip(&b.taps).enumerate() {
        assert_eq!(x.key, y.key, "tap {i} key");
        assert_eq!(x.delay_s.to_bits(), y.delay_s.to_bits(), "tap {i} delay");
        assert_eq!(x.dir.map(f32::to_bits), y.dir.map(f32::to_bits), "tap {i} dir");
        assert_eq!(bits3(x.gains), bits3(y.gains), "tap {i} gains");
    }
    assert_eq!(bits3(a.reverb.rt60), bits3(b.reverb.rt60), "reverb rt60");
    assert_eq!(bits3(a.reverb.level), bits3(b.reverb.level), "reverb level");
    match (&a.remote, &b.remote) {
        (None, None) => {}
        (Some(x), Some(y)) => {
            assert_eq!(bits3(x.rt60), bits3(y.rt60), "remote rt60");
            assert_eq!(bits3(x.send), bits3(y.send), "remote send");
            assert_eq!(x.sh.map(f32::to_bits), y.sh.map(f32::to_bits), "remote sh");
        }
        _ => panic!("remote presence changed: {:?} → {:?}", a.remote.is_some(), b.remote.is_some()),
    }
}

fn roundtrip(pb: &ParamBlock) -> (Vec<f32>, ParamBlock) {
    let mut flat = Vec::new();
    pb.write_flat(&mut flat);
    let back = ParamBlock::read_flat(&flat);
    (flat, back)
}

#[test]
fn roundtrip_with_taps_and_remote() {
    let pb = block(40, true, 12_345, 1);
    let (flat, back) = roundtrip(&pb);
    assert_eq!(flat.len(), FLAT_HEADER + 40 * FLAT_PER_TAP);
    assert_eq!(flat.len(), pb.flat_len());
    assert_same(&pb, &back);
}

#[test]
fn roundtrip_without_remote() {
    let pb = block(7, false, 3, 2);
    let (flat, back) = roundtrip(&pb);
    assert_eq!(flat[8], 0.0, "remote flag must be 0 when absent");
    assert!(back.remote.is_none());
    assert_same(&pb, &back);
}

#[test]
fn roundtrip_empty_and_default_blocks() {
    // an idle source ships Default: silent reverb level must survive
    let pb = ParamBlock::default();
    let (flat, back) = roundtrip(&pb);
    assert_eq!(flat.len(), FLAT_HEADER);
    assert_same(&pb, &back);
    assert_eq!(back.reverb.level, [0.0; NBANDS], "idle block must stay silent");
    // remote only, no taps
    let pb = block(0, true, 0, 3);
    assert_same(&pb, &roundtrip(&pb).1);
}

#[test]
fn reused_buffer_does_not_leak_stale_data() {
    let mut flat = Vec::new();
    // big block with a remote reverb first...
    block(64, true, 1, 4).write_flat(&mut flat);
    // ...then a small one without: nothing of the first may survive
    let small = block(3, false, 2, 5);
    small.write_flat(&mut flat);
    assert_eq!(flat.len(), small.flat_len());
    assert!(flat[9..24].iter().all(|&x| x == 0.0), "stale remote payload in header");
    let back = ParamBlock::read_flat(&flat);
    assert_same(&small, &back);
}

#[test]
fn realistic_engine_keys_survive() {
    // key spaces the engine actually ships below 2^24: ISM indices,
    // world-sim fixed keys (7000.., 9000), PT box keys (100_000 + base-7)
    let mut rng = Rng::new(6);
    let keys = [0u32, 1, 62, 7000, 7799, 9000, 100_000, 100_000 + 342, (1 << 24) - 1];
    let pb = ParamBlock {
        taps: keys.iter().map(|&k| tap(&mut rng, k)).collect(),
        ..Default::default()
    };
    assert_same(&pb, &roundtrip(&pb).1);
}

/// Tap keys are IDENTITY ("same key ⇒ same path, glide; new key ⇒
/// crossfade"). The flat layout stores them as `key as f32`, which is
/// exact only below 2^24. omg-scene ships keys far above that:
/// `PT_KEY_BASE + 0x0800_0000 + edge_key` for bent paths (≈1.34e8, where
/// the f32 step is 16 — sixteen consecutive edge keys collapse into ONE)
/// and `PT_KEY_BASE + MeshRecord::key()` (a 32-bit FNV hash: up to 256
/// apart collapse). Distinct paths then share a key after transport.
/// Minimal fix: the per-tap slot has a spare float (FLAT_PER_TAP = 9,
/// only 8 used) — carry key bits 16..31 in `o + 8` and bits 0..15 in `o`
/// (both exact small integers in f32, NaN-free unlike a raw bit-cast).
#[test]
fn large_keys_stay_distinct_and_exact() {
    let mut rng = Rng::new(7);
    let base = 100_000u32 + 0x0800_0000;
    let keys = [base + 1, base + 2, base + 3, 0x9E37_79B9, 0xFFFF_FFFF];
    let pb = ParamBlock {
        taps: keys.iter().map(|&k| tap(&mut rng, k)).collect(),
        ..Default::default()
    };
    let back = roundtrip(&pb).1;
    let got: Vec<u32> = back.taps.iter().map(|t| t.key).collect();
    assert_eq!(got, keys, "tap keys corrupted by the flat transport");
}
