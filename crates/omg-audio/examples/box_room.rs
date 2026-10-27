//! A source in a brick room with a doorway, heard from outside while
//! the listener walks in through the door. Renders 6 s to box_room.wav.
//!
//!     cargo run --release -p omg-audio --example box_room

use omg_audio::{clip, Config, Geometry, Hrir, Material, Mixer, Play, Simulation, SourceDesc};

const SR: f32 = 48_000.0;

fn main() -> std::io::Result<()> {
    // 1. geometry: a 8×6×3.5 m brick room with a doorway, on grass
    let mut g = Geometry::new();
    let brick = g.material(Material::BRICK);
    let wood = g.material(Material::WOOD);
    let grass = g.material(Material::GRASS);
    let t = 0.24;
    g.add_box([0.0, 0.0, -0.1], [8.0, 6.0, 0.0], wood); // floor
    g.add_box([-t, -t, 3.5], [8.0 + t, 6.0 + t, 3.5 + t], brick); // roof
    g.add_box([-t, -t, 0.0], [0.0, 6.0 + t, 3.5], brick);
    g.add_box([8.0, -t, 0.0], [8.0 + t, 6.0 + t, 3.5], brick);
    g.add_box([0.0, 6.0, 0.0], [8.0, 6.0 + t, 3.5], brick);
    g.add_box([0.0, -t, 0.0], [3.5, 0.0, 3.5], brick); // front wall, door at x 3.5..4.5
    g.add_box([4.5, -t, 0.0], [8.0, 0.0, 3.5], brick);
    g.add_box([3.5, -t, 2.2], [4.5, 0.0, 3.5], brick);
    g.ground_plane(-0.1, grass);

    // 2. simulation: one source inside, the listener outside
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at([6.0, 4.0, 1.2]));

    // 3. mixer: the engine's built-in HRIRs, a pulsing tone as the clip
    let mut mixer = Mixer::new(SR, Hrir::builtin());
    let tone: Vec<f32> = (0..SR as usize)
        .map(|i| {
            let t = i as f32 / SR;
            let env = (-(t % 0.25) * 18.0).exp();
            0.5 * env * ((t * 440.0 * std::f32::consts::TAU).sin() + 0.5 * (t * 1320.0 * std::f32::consts::TAU).sin())
        })
        .collect();
    mixer.play(src, &clip(&tone), Play::LOOP);

    // 4. walk from 10 m outside, through the door, into the room — a
    //    simulation tick every 50 ms, audio rendered in between
    let secs = 6.0;
    let ticks = (secs * 20.0) as usize;
    let block = (SR / 20.0) as usize;
    let (mut l, mut r) = (vec![0.0f32; block], vec![0.0f32; block]);
    let mut out: Vec<f32> = Vec::with_capacity(2 * ticks * block);
    for k in 0..ticks {
        let y = -10.0 + 13.0 * k as f32 / ticks as f32;
        // facing +y (yaw 90°): into the room
        sim.set_listener([4.0, y, 1.7], std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        mixer.apply(&sim.tick());
        mixer.render(&mut l, &mut r);
        for i in 0..block {
            out.push(l[i]);
            out.push(r[i]);
        }
    }
    write_wav("box_room.wav", &out, SR as u32)?;
    println!("wrote box_room.wav ({secs} s, walking in through the door)");
    Ok(())
}

/// Minimal 16-bit stereo PCM WAV writer.
fn write_wav(path: &str, interleaved: &[f32], sr: u32) -> std::io::Result<()> {
    let data_len = (interleaved.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&2u16.to_le_bytes()); // stereo
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for x in interleaved {
        b.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b)
}
