//! The built-in bus decoder (`hrir/hrir_sh2.bin`): a plane wave on the
//! order-2 bus decodes as the HRTF of its direction — so the bus and the
//! point-rendered paths (`HRIR_GRID`, nearest HRIR) play a sound alike.
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::hrtf::{BinauralDecoder, HrirGrid};

fn decode(dir: [f32; 3]) -> (Vec<f32>, Vec<f32>) {
    let mut dec = BinauralDecoder::from_bytes(omg_audio::HRIR_SPEAKERS);
    let enc = encode_gains(dir);
    let (mut l, mut r) = (Vec::new(), Vec::new());
    for i in 0..256 {
        let sh: [f32; NCH] = core::array::from_fn(|n| if i == 0 { enc[n] } else { 0.0 });
        let (a, b) = dec.process(&sh);
        l.push(a);
        r.push(b);
    }
    (l, r)
}

fn energy(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

#[test]
fn a_plane_wave_decodes_as_its_hrtf() {
    let grid = HrirGrid::from_bytes(omg_audio::HRIR_GRID);
    for (name, dir) in [("front", [1.0f32, 0.0, 0.0]), ("left", [0.0, 1.0, 0.0]), ("below", [0.0, 0.0, -1.0]), ("up-right", [0.5, -0.5, 0.707])] {
        let (l, r) = decode(dir);
        let (hl, hr) = grid.hrir(grid.nearest(dir));
        let db = |a: f32, b: f32| 10.0 * (a / b).log10();
        let (el, er) = (db(energy(&l), energy(hl)), db(energy(&r), energy(hr)));
        eprintln!("{name}: decoded vs HRIR energy, left {el:+.1} dB, right {er:+.1} dB");
        // order 2 can't hold every notch: within a few dB, per ear
        assert!(el.abs() < 4.0 && er.abs() < 4.0, "{name}: {el:+.1} / {er:+.1} dB");
    }
    let (l, r) = decode([1.0, 0.0, 0.0]);
    assert!((energy(&l) / energy(&r) - 1.0).abs() < 0.05, "ahead: both ears alike");
    let (l, r) = decode([0.0, 1.0, 0.0]);
    assert!(energy(&l) > 8.0 * energy(&r), "from the left: the left ear (ILD {:.1} dB)", 10.0 * (energy(&l) / energy(&r)).log10());
}
