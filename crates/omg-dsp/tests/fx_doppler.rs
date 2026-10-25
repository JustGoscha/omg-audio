//! Effect: Doppler by construction (renderer.rs, same-key delay glide).
//!
//! Claim: a tap whose `key` stays the same while its `delay_s` moves is
//! rendered by GLIDING one fractional read head through the source delay
//! line, so the pitch heard is exactly the kinematic Doppler ratio
//! f'/f = 1 − dτ/dt (τ = propagation delay). No pitch-shifter, no
//! separate Doppler parameter — the shift is a by-product of the delay
//! motion. Same-key delay jumps larger than RETARGET_MS (6 ms) are a path
//! identity change in disguise and must CROSSFADE (two slots, both at
//! the original pitch) instead of sliding (which would be a chirp).
//!
//! Isolation: bus-only `Renderer::new` (no HRIR grid), one tap straight
//! ahead, W channel read raw. No scene, no simulation.
//! Combined in the demo: omg-scene tests/early_gate.rs ("glide — not
//! re-key — while walking") pins the key stability that makes this
//! effect fire on real motion; tests/car_drone.rs drives a moving car.

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::playlog::{Clip, PlayLog};
use omg_dsp::Renderer;

const C: f32 = 343.0;
const F0: f32 = 1000.0;

fn block(version: u64, delay_s: f32) -> ParamBlock {
    ParamBlock {
        taps: vec![Tap { key: 7, delay_s, dir: [1.0, 0.0, 0.0], gains: [1.0, 1.0, 1.0] }],
        version,
        ..Default::default()
    }
}

/// Render a 1 kHz tone through one tap whose delay follows `delay_at(t)`,
/// re-sent at 100 Hz (a typical simulation tick). Returns (W channel,
/// max live slot count seen).
fn render(n: usize, delay_at: impl Fn(f32) -> f32) -> (Vec<f32>, u32) {
    let mut r = Renderer::new(SR);
    let mut version = 1;
    r.set_params(&block(version, delay_at(0.0)));
    let mut out = Vec::with_capacity(n);
    let mut max_live = 0;
    for k in 0..n {
        if k % 480 == 0 && k > 0 {
            version += 1;
            r.set_params(&block(version, delay_at(k as f32 / SR)));
        }
        let mut bus = [0.0f32; NCH];
        r.process(sine_at(F0, k), &mut bus);
        out.push(bus[0]);
        max_live = max_live.max(r.debug_stats().0);
    }
    (out, max_live)
}

/// Approaching at 20 m/s: the delay shrinks at v/c per second and the
/// tone must rise by exactly that ratio; receding lowers it. The slot is
/// never re-keyed (one live slot throughout).
#[test]
fn moving_delay_glides_to_the_doppler_ratio() {
    let t_move = 0.3; // fade-in done (~6 τ of the 50 ms slot smoother)
    for v in [20.0f32, -20.0, 0.0] {
        let rate = -v / C; // dτ/dt
        let (y, live) = render(48_000, |t| 0.2 + rate * (t - t_move).max(0.0));
        // measure 0.3 s after motion starts (smoother is in steady glide)
        let f = zc_freq(&y[28_800..43_200]);
        let want = F0 * (1.0 - rate);
        assert!(
            (f - want).abs() < 0.002 * want,
            "v={v} m/s: measured {f:.2} Hz, Doppler predicts {want:.2} Hz"
        );
        assert_eq!(live, 1, "v={v}: motion must glide one slot, never re-key");
        // the glide must not cost level
        let a = rms(&y[28_800..43_200]) * 2f32.sqrt();
        assert!((a - 1.0).abs() < 0.02, "v={v}: amplitude {a}");
    }
}

/// Same key, delay jumps 20.25 ms (> RETARGET_MS): crossfade. Every
/// 10 ms window through the transition stays at the source pitch, and
/// for a while two slots are live (old fading out, new fading in).
/// Control: a 4 ms jump (< RETARGET_MS) glides — a clearly audible
/// pitch excursion, one slot.
#[test]
fn large_same_key_jump_crossfades_small_jump_glides() {
    let t_jump = 0.4;
    let pitch_track = |jump_s: f32| -> (Vec<f32>, u32) {
        let (y, live) =
            render(72_000, |t| if t < t_jump { 0.1 } else { 0.1 + jump_s });
        let j = (t_jump * SR) as usize;
        let f: Vec<f32> = (0..15).map(|w| zc_freq(&y[j + w * 480..j + (w + 1) * 480])).collect();
        (f, live)
    };

    let (f_x, live_x) = pitch_track(0.020_25);
    let worst_x = f_x.iter().map(|f| (f / F0 - 1.0).abs()).fold(0.0, f32::max);
    assert!(worst_x < 0.01, "crossfade must not chirp: windows {f_x:?}");
    assert_eq!(live_x, 2, "a retarget must overlap old and new slot");

    let (f_g, live_g) = pitch_track(0.004);
    let worst_g = f_g.iter().map(|f| (f / F0 - 1.0).abs()).fold(0.0, f32::max);
    assert!(worst_g > 0.05, "a small jump is motion and must glide: {f_g:?}");
    assert_eq!(live_g, 1, "a glide keeps one slot");
}

/// After a retarget crossfade settles, the old ghost is gone and the
/// output is the tone at the NEW delay, sample-exact.
#[test]
fn crossfade_lands_on_the_new_delay() {
    let d2 = 0.120_25f32;
    let (y, _) = render(96_000, |t| if t < 0.4 { 0.1 } else { d2 });
    let ds = d2 * SR; // fractional samples
    let mut err = 0.0f32;
    for k in 86_400..96_000 {
        let t = k as f32 - ds;
        let want = (std::f64::consts::TAU * F0 as f64 * t as f64 / SR as f64).sin() as f32;
        err = err.max((y[k] - want).abs());
    }
    // linear-interp read error at 1 kHz ≤ (ωT)²/8 ≈ 2e-3
    assert!(err < 3e-3, "residual vs. tone at new delay: {err}");
}

/// Walking-speed motion at long delays (up to 0.65 s, where the f32
/// delay smoother's ulp is ~2e-3 samples): the tiny shift is still
/// rendered to within 2 % of its size, steady across 100 ms windows.
#[test]
fn slow_motion_at_long_delay_keeps_its_shift() {
    for (d0, v) in [(0.5f32, 1.4f32), (0.65, 0.3), (0.05, 0.3)] {
        let rate = -v / C;
        let (y, _) = render(96_000, |t| d0 + rate * (t - 0.3).max(0.0));
        let want_shift = -rate * F0;
        for w in 0..5 {
            let f = zc_freq(&y[48_000 + w * 9600..48_000 + (w + 1) * 9600]);
            assert!(
                ((f - F0) / want_shift - 1.0).abs() < 0.02,
                "d0 {d0} s, v {v} m/s, window {w}: shift {:.3} Hz, want {want_shift:.3} Hz",
                f - F0
            );
        }
    }
}

/// Same as `render`, but the source is a looped 1 kHz recording read from
/// a PlayLog (no delay line) and updates arrive every `tick` samples.
fn render_log(n: usize, tick: usize, delay_at: impl Fn(f32) -> f32) -> (Vec<f32>, u32) {
    let mut r = Renderer::new(SR).without_line();
    let mut log = PlayLog::new();
    // 48 samples per cycle at 48 kHz: 100 whole cycles loop seamlessly
    log.set_loop(Clip::from(sine(F0, 4800, 1.0)), 0.0);
    let mut version = 1;
    r.set_params(&block(version, delay_at(0.0)));
    let mut out = Vec::with_capacity(n);
    let mut max_live = 0;
    for k in 0..n {
        if k % tick == 0 && k > 0 {
            version += 1;
            r.set_params(&block(version, delay_at(k as f32 / SR)));
        }
        let mut bus = [0.0f32; NCH];
        r.process_log(&log, k as f64, 1.0, &mut bus);
        out.push(bus[0]);
        max_live = max_live.max(r.debug_stats().0);
    }
    (out, max_live)
}

/// 10 km away (29 s of delay, ~1.4 M samples) and driving off at 20 m/s:
/// the pitch is still exactly Doppler. Delays are f64 end to end — in
/// f32 a 1.4 M-sample delay steps in 1/8 samples and the glide stalls.
#[test]
fn doppler_holds_at_10_km() {
    let t_move = 0.3;
    for v in [20.0f32, -20.0] {
        let rate = -v / C;
        let (y, live) = render_log(48_000, 480, |t| 10_000.0 / C + rate * (t - t_move).max(0.0));
        let f = zc_freq(&y[28_800..43_200]);
        let want = F0 * (1.0 - rate);
        assert!((f - want).abs() < 0.002 * want, "v={v}: measured {f:.2} Hz, Doppler predicts {want:.2} Hz");
        assert_eq!(live, 1, "v={v}: must glide one slot");
    }
}

/// A 150 m/s flyover with updates at 20 Hz moves the delay 21.9 ms per
/// tick — past RETARGET_MS, but it is motion, not a new path: the
/// retarget window widens to half the time between updates (Mach 0.5),
/// so it still glides at the Doppler pitch. A teleport still crossfades.
#[test]
fn fast_motion_at_slow_updates_still_glides() {
    let t_move = 0.3;
    let v = 150.0f32;
    let rate = -v / C;
    let (y, live) = render_log(48_000, 2400, |t| 0.5 + rate * (t - t_move).max(0.0));
    let f = zc_freq(&y[33_600..45_600]);
    let want = F0 * (1.0 - rate);
    assert!((f - want).abs() < 0.01 * want, "measured {f:.2} Hz, Doppler predicts {want:.2} Hz");
    assert_eq!(live, 1, "a fast flyover must glide, not crossfade every tick");
    // a 1 km teleport at the same update rate: crossfade (two slots)
    let (_, live) = render_log(9600, 2400, |t| if t < 0.1 { 0.5 } else { 0.5 + 1000.0 / C });
    assert_eq!(live, 2, "a teleport must crossfade");
}
