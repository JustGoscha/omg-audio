//! The host side behind the Godot nodes — plain Rust, no Godot types:
//! omg-audio's threaded Engine plus what a game host adds around it
//! (clip decoding, the output ring and load governor, material names).

pub mod clips;
pub mod decode;
pub mod flight;
pub mod host;
pub mod materials;
pub mod record;

pub use host::{Host, OutCmd, OutputReader};
