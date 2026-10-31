//! The town's diffraction edges near a point (EDGES_AT=x,y): where the
//! edge set puts them, and how long they are. GEOMETRY=<file>: a host's
//! dumped level instead; the count lying on the ground (both ends within
//! 5 cm of z 0) too.
use omg_audio::{Config, Simulation};
use omg_web::town::Town;

fn main() {
    let at: Vec<f32> = std::env::var("EDGES_AT").unwrap_or_else(|_| "16.25,51.75".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let mut sim = Simulation::new(Config::default());
    let _town = Town::new(&mut sim);
    if let Ok(path) = std::env::var("GEOMETRY") {
        sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY: a dumped geometry"));
    }
    let mesh = sim.mesh();
    // SEG=x,y,z,x,y,z: what a segment crosses
    if let Ok(seg) = std::env::var("SEG") {
        let v: Vec<f32> = seg.split(',').map(|x| x.parse().unwrap()).collect();
        let (a, b) = (omg_core::vec3::Vec3::new(v[0], v[1], v[2]), omg_core::vec3::Vec3::new(v[3], v[4], v[5]));
        let mut hits = Vec::new();
        mesh.segment_hits(a, b, &mut hits);
        for h in &hits {
            let p = a + (b - a) * h.t;
            let m = mesh.materials[h.material as usize];
            let [i0, i1, i2] = mesh.indices[h.tri as usize];
            let q = mesh.positions[i0 as usize];
            println!("hit t {:.3} at ({:.2},{:.2},{:.2}) material t {:?} (tri at ({:.2},{:.2},{:.2}) {} {})", h.t, p.x, p.y, p.z, m.transmission, q.x, q.y, q.z, i1, i2);
        }
        return;
    }
    let es = omg_core::v3::EdgeSet::build(mesh);
    let ground = es.edges.iter().filter(|(a, b)| a.z.abs() < 0.05 && b.z.abs() < 0.05).count();
    println!("{} edges, {ground} lying on the ground", es.edges.len());
    for (i, (a, b)) in es.edges.iter().enumerate() {
        let m = (*a + *b) * 0.5;
        let zmin: f32 = std::env::var("ZMIN").ok().and_then(|z| z.parse().ok()).unwrap_or(-1e9);
        let wide = std::env::var("WIDE").is_ok();
        if (wide || (m.x - at[0]).abs() < 1.2) && (m.y - at[1]).abs() < 1.2 && m.z >= zmin {
            println!("{i:6}: ({:.2},{:.2},{:.2}) – ({:.2},{:.2},{:.2})", a.x, a.y, a.z, b.x, b.y, b.z);
        }
    }
}

#[allow(dead_code)]
fn unused() {}
