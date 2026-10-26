//! `Frame`: everything one simulation tick tells the mixer. The ONLY
//! coupling between the two clocks — so it serializes to flat f32s
//! (`write_flat` / `read_flat`) for a worker → AudioWorklet postMessage
//! or a SharedArrayBuffer, without a serialization library.

use omg_core::params::ParamBlock;

/// A sound source in a `Simulation` (and its channel in the `Mixer`).
/// Ids are never reused within one simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u32);

/// The listener's pose. Position in meters (z up); `yaw` turns left
/// (about +z, 0 = facing +x), `pitch` looks up, `roll` tilts the head
/// right — radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Listener {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// One direction cell of the outdoor field: the share of the listener's
/// rays that reached open sky leaving along `dir`, per band, weighted by
/// everything lost on the way (doorways, panes, walls). Feed it to your
/// ambience (wind, rain, birds): outdoors it is large and from above,
/// indoors small and from the windows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyCell {
    pub dir: [f32; 3],
    pub energy: [f32; 3],
}

/// One simulation tick's result.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Tick number (increases by one per `Simulation::tick`).
    pub version: u64,
    pub listener: Listener,
    /// When the listener pose this frame was traced from was current, in
    /// seconds on the host's clock (`Simulation::set_listener_at`); None
    /// when the host gave no time. The mixer times path motion by it.
    pub time: Option<f64>,
    /// Every source's acoustic paths (world frame; the mixer applies the
    /// head orientation). A block keeps its `version` while it is not
    /// re-traced (level of detail), so re-applying it costs nothing.
    pub blocks: Vec<(SourceId, ParamBlock)>,
    /// The outdoor field (see `SkyCell`); empty before the first trace.
    pub sky: Vec<SkyCell>,
}

/// Flat layout tag ("OMGF") and revision; bump on any layout change.
const MAGIC: u32 = 0x4f4d_4746;
const LAYOUT: u32 = 2;
const HEADER: usize = 14;
const SKY_WORDS: usize = 6;

impl Frame {
    /// The block of one source.
    pub fn block(&self, id: SourceId) -> Option<&ParamBlock> {
        self.blocks.iter().find(|(i, _)| *i == id).map(|(_, b)| b)
    }

    /// How much open sky the listener hears, per band: 0 sealed in,
    /// ~1 in the middle of an open plain.
    pub fn outdoor(&self) -> [f32; 3] {
        let mut e = [0.0; 3];
        for c in &self.sky {
            for b in 0..3 {
                e[b] += c.energy[b];
            }
        }
        e
    }

    /// Serialize to flat f32 words (ids and version bit-exact):
    /// ```text
    /// [0] magic  [1] layout  [2..4] version lo, hi (u32 bits)
    /// [4..7] listener position  [7] yaw  [8] pitch  [9] roll
    /// [10] sky cells  [11] blocks  [12..14] time lo, hi (f64 bits; NaN
    ///                                   = none)
    /// sky cells × 6: dir xyz, energy ×3
    /// per block: id (u32 bits), length n, then n words of
    ///            ParamBlock::write_flat
    /// ```
    pub fn write_flat(&self, out: &mut Vec<f32>) {
        let bits = f32::from_bits;
        out.clear();
        out.extend_from_slice(&[
            bits(MAGIC),
            bits(LAYOUT),
            bits(self.version as u32),
            bits((self.version >> 32) as u32),
        ]);
        let l = &self.listener;
        out.extend_from_slice(&l.position);
        out.extend_from_slice(&[l.yaw, l.pitch, l.roll]);
        out.push(bits(self.sky.len() as u32));
        out.push(bits(self.blocks.len() as u32));
        let t = self.time.unwrap_or(f64::NAN).to_bits();
        out.extend_from_slice(&[bits(t as u32), bits((t >> 32) as u32)]);
        for c in &self.sky {
            out.extend_from_slice(&c.dir);
            out.extend_from_slice(&c.energy);
        }
        let mut scratch = Vec::new();
        for (id, pb) in &self.blocks {
            pb.write_flat(&mut scratch);
            out.push(bits(id.0));
            out.push(bits(scratch.len() as u32));
            out.extend_from_slice(&scratch);
        }
    }

    /// Parse `write_flat`'s output. None when `data` is not a frame of
    /// this layout (or is truncated).
    pub fn read_flat(data: &[f32]) -> Option<Frame> {
        let u = |i: usize| data.get(i).map(|x| x.to_bits());
        if u(0)? != MAGIC || u(1)? != LAYOUT || data.len() < HEADER {
            return None;
        }
        let version = u(2)? as u64 | (u(3)? as u64) << 32;
        let listener = Listener {
            position: [data[4], data[5], data[6]],
            yaw: data[7],
            pitch: data[8],
            roll: data[9],
        };
        let (n_sky, n_blocks) = (u(10)? as usize, u(11)? as usize);
        let time = Some(f64::from_bits(u(12)? as u64 | (u(13)? as u64) << 32)).filter(|t| !t.is_nan());
        let mut o = HEADER;
        let sky_end = o.checked_add(n_sky.checked_mul(SKY_WORDS)?)?;
        let cells = data.get(o..sky_end)?;
        let sky = cells
            .chunks_exact(SKY_WORDS)
            .map(|c| SkyCell { dir: [c[0], c[1], c[2]], energy: [c[3], c[4], c[5]] })
            .collect();
        o = sky_end;
        let mut blocks = Vec::with_capacity(n_blocks.min(4096));
        for _ in 0..n_blocks {
            let (id, n) = (u(o)?, u(o + 1)? as usize);
            let body = data.get(o + 2..o + 2 + n)?;
            if body.len() < omg_core::params::FLAT_HEADER
                || body.len() < omg_core::params::FLAT_HEADER + body[1] as usize * omg_core::params::FLAT_PER_TAP
            {
                return None;
            }
            blocks.push((SourceId(id), ParamBlock::read_flat(body)));
            o += 2 + n;
        }
        Some(Frame { version, listener, time, blocks, sky })
    }
}
