use crate::NBANDS;

/// One discrete arrival at the listener: the direct path or an early
/// specular reflection. `gains` are per-band amplitudes including distance,
/// air and wall absorption. `dir` is the arrival direction (unit vector from
/// listener toward the apparent source), listener-relative.
#[derive(Clone, Copy, Debug)]
pub struct Tap {
    /// Stable identity of the propagation path this tap represents.
    /// Same key across updates ⇒ same path ⇒ the renderer may glide its
    /// delay (Doppler). New/vanished key ⇒ the renderer crossfades.
    pub key: u32,
    pub delay_s: f32,
    pub dir: [f32; 3],
    pub gains: [f32; NBANDS],
}

#[derive(Clone, Copy, Debug)]
pub struct ReverbParams {
    /// Per-band decay time of the late field, seconds.
    pub rt60: [f32; NBANDS],
    /// Per-band amplitude of the late field at the listener for a unit
    /// source signal, as measured by the tracer. In a diffuse field this is
    /// (nearly) independent of source–listener distance — the direct path
    /// gets louder as you approach a source, the late field does not.
    /// The renderer feeds the FDN the *raw* source signal scaled by this,
    /// never the distance-attenuated early taps.
    pub level: [f32; NBANDS],
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self { rt60: [0.5; NBANDS], level: [0.05; NBANDS] }
    }
}

/// Coupled-room ("remote") reverb: the reverberant field of the *source's*
/// room, heard through the doorway into the listener's room. Rendered as a
/// directional wet emitter at the portal — this is what makes a hall sound
/// like a hall from the corridor, before you step in.
#[derive(Clone, Copy, Debug)]
pub struct RemoteReverb {
    /// Decay of the source's room (not the listener's).
    pub rt60: [f32; NBANDS],
    /// Per-band send amplitude for a unit source signal: wet level at the
    /// exit door × door muffle × spreading to the listener.
    pub send: [f32; NBANDS],
    /// SH (order-2, ACN/SN3D) encoding gains of the arrival direction
    /// (the entry doorway).
    pub sh: [f32; 9],
}

/// The complete simulation → renderer contract. The simulation thread
/// produces one of these at 10–30 Hz; the audio thread interpolates toward
/// it. This struct is the *entire* coupling between the two clocks — in the
/// web build it will be serialized through a SharedArrayBuffer verbatim.
#[derive(Clone, Debug, Default)]
pub struct ParamBlock {
    pub taps: Vec<Tap>,
    pub reverb: ReverbParams,
    /// Present when the source is in another room: its room's wet field
    /// arriving through the portal.
    pub remote: Option<RemoteReverb>,
    pub version: u64,
}

/// Flat f32 layout so a ParamBlock can cross thread/worker boundaries
/// (postMessage, SharedArrayBuffer) without a serialization library:
///   [0] version  [1] n_taps
///   [2..8]   reverb rt60[3], level[3]
///   [8]      remote flag (0/1)
///   [9..24]  remote rt60[3], send[3], sh[9]
///   then per tap 9 floats: key, delay_s, dir[3], gains[3]
pub const FLAT_HEADER: usize = 24;
pub const FLAT_PER_TAP: usize = 9;

impl ParamBlock {
    pub fn flat_len(&self) -> usize {
        FLAT_HEADER + self.taps.len() * FLAT_PER_TAP
    }

    pub fn write_flat(&self, out: &mut Vec<f32>) {
        out.clear();
        out.resize(self.flat_len(), 0.0);
        out[0] = self.version as f32;
        out[1] = self.taps.len() as f32;
        for b in 0..NBANDS {
            out[2 + b] = self.reverb.rt60[b];
            out[5 + b] = self.reverb.level[b];
        }
        if let Some(r) = &self.remote {
            out[8] = 1.0;
            for b in 0..NBANDS {
                out[9 + b] = r.rt60[b];
                out[12 + b] = r.send[b];
            }
            out[15..24].copy_from_slice(&r.sh);
        }
        for (i, t) in self.taps.iter().enumerate() {
            let o = FLAT_HEADER + i * FLAT_PER_TAP;
            out[o] = t.key as f32;
            out[o + 1] = t.delay_s;
            out[o + 2..o + 5].copy_from_slice(&t.dir);
            out[o + 5..o + 8].copy_from_slice(&t.gains);
        }
    }

    pub fn read_flat(data: &[f32]) -> Self {
        let n_taps = data[1] as usize;
        let mut pb = ParamBlock {
            taps: Vec::with_capacity(n_taps),
            version: data[0] as u64,
            ..Default::default()
        };
        for b in 0..NBANDS {
            pb.reverb.rt60[b] = data[2 + b];
            pb.reverb.level[b] = data[5 + b];
        }
        if data[8] > 0.5 {
            let mut r = RemoteReverb { rt60: [0.0; NBANDS], send: [0.0; NBANDS], sh: [0.0; 9] };
            for b in 0..NBANDS {
                r.rt60[b] = data[9 + b];
                r.send[b] = data[12 + b];
            }
            r.sh.copy_from_slice(&data[15..24]);
            pb.remote = Some(r);
        }
        for i in 0..n_taps {
            let o = FLAT_HEADER + i * FLAT_PER_TAP;
            pb.taps.push(Tap {
                key: data[o] as u32,
                delay_s: data[o + 1],
                dir: [data[o + 2], data[o + 3], data[o + 4]],
                gains: [data[o + 5], data[o + 6], data[o + 7]],
            });
        }
        pb
    }
}
