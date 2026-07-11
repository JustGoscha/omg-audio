//! omg-dsp: the audio-clock side. Everything here runs per-sample inside
//! the real-time callback: no allocation, no locks, no syscalls after init.
//! It renders whatever the latest ParamBlock says, smoothly interpolating
//! parameters so simulation updates never click.

pub mod ambi;
pub mod delay;
pub mod fdn;
pub mod filter;
pub mod hrtf;
pub mod level;
pub mod output;
pub mod rain;
pub mod renderer;
pub mod smooth;

pub use renderer::Renderer;
