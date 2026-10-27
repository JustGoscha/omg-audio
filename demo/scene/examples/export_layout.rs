//! Regenerate `assets/demo/layout.json` from `walkthrough::layout` — the
//! one interior layout the tracer, the web demo and the Godot demo share.
//!
//!     cargo run -p omg-scene --example export_layout

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/demo/layout.json");
    std::fs::create_dir_all(path.parent().unwrap()).expect("assets/demo");
    std::fs::write(&path, omg_scene::walkthrough::layout_json()).expect("write layout.json");
    println!("wrote {}", path.display());
}
