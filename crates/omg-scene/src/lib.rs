//! omg-scene: the demo world and its simulation orchestration, shared by
//! the native app (scripted walkthrough path) and the web build
//! (interactive listener). No I/O, no clocks — compiles to wasm unchanged.

pub mod sim;
pub mod walkthrough;
pub mod world;
