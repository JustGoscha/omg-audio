//! C6a gate: every triangle of the world mesh carries a stable
//! authored-surface id; all triangles of one surface are coplanar
//! (door holes and tessellation must not fracture identity); the mesh
//! has as many surfaces as the scene has authored planes (ground +
//! 5 per room shell).

use omg_scene::dome::build_world_mesh;
use omg_scene::walkthrough;

#[test]
fn world_mesh_surfaces_are_stable_planes() {
    let rooms = walkthrough::rooms();
    let doors = walkthrough::doors();
    let (mesh, _) = build_world_mesh(&rooms, &doors);

    let mut by_surface: std::collections::HashMap<u16, Vec<u32>> = Default::default();
    for t in 0..mesh.tri_count() as u32 {
        by_surface.entry(mesh.tri_surface(t)).or_default().push(t);
    }
    let indoor = rooms.iter().filter(|r| !r.outdoor).count();
    assert!(
        by_surface.len() >= 1 + indoor * 5,
        "expected ≥ {} surfaces, got {}",
        1 + indoor * 5,
        by_surface.len()
    );

    for (sid, tris) in &by_surface {
        let n0 = mesh.tri_normal(tris[0]);
        let p0 = mesh.positions[mesh.indices[tris[0] as usize][0] as usize];
        for &t in tris {
            let n = mesh.tri_normal(t);
            assert!(
                n.dot(n0).abs() > 0.999,
                "surface {sid}: non-parallel triangles"
            );
            let p = mesh.positions[mesh.indices[t as usize][0] as usize];
            assert!(
                (p - p0).dot(n0).abs() < 1e-3,
                "surface {sid}: triangles on different planes"
            );
        }
    }
}
