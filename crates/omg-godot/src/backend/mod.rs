//! The host side behind the Godot nodes — plain Rust, no Godot types:
//! omg-audio's threaded Engine plus what a game host adds around it
//! (clip decoding, the output ring and load governor, the outdoor beds,
//! material names).

pub mod clips;
pub mod decode;
pub mod env;
pub mod host;
pub mod materials;

pub use host::{Host, OutCmd, OutputReader};
