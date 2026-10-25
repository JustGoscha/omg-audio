//! omg-scene: the demo world and its simulation orchestration, shared by
//! the native app (scripted walkthrough path) and the web build
//! (interactive listener). No I/O, no clocks — compiles to wasm unchanged.

pub mod ambient;
pub mod dome;
pub mod early;
pub mod early_world;
pub mod environment;
pub mod late;
pub mod quality;
pub mod sim;
pub mod v3walk;
pub mod walkthrough;
pub mod world;
