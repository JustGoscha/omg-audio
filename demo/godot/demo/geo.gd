# Geometry batching for the demo level. Everything is built in PLAN
# coordinates (x east, y north, z up, meters — the web demo's and the
# engine's frame) and converted to Godot's (x, z, -y) here.
#
# A Batch is one visual material × one acoustic material: its triangles
# become one MeshInstance3D under an OmgGeometry node carrying the
# acoustic material (so the engine hears exactly what you see), plus an
# optional trimesh collider. The acoustic material defaults to what the
# visual material declares (DemoMats: `omg_material` meta); "" = decor,
# kept out of the sound world — and decor bigger than a couple of square
# meters must say why (`audit()` lists the rest).
# UVs are world meters, projected along each face's normal.
class_name DemoGeo
extends RefCounted

static func P(x: float, y: float, z: float) -> Vector3:
	return Vector3(x, z, -y)

class Batch:
	var st := SurfaceTool.new()
	var faces := PackedVector3Array()
	var vis: Material
	var acoustic: String
	var collide: bool
	var boxes: bool
	var why: String
	var count := 0
	var area := 0.0

	func _init(v: Material, a: String, c: bool, b: bool, w := "") -> void:
		vis = v
		acoustic = a
		collide = c
		boxes = b
		why = w
		st.begin(Mesh.PRIMITIVE_TRIANGLES)

	static func _uv(p: Vector3, n: Vector3) -> Vector2:
		var a := n.abs()
		if a.y >= a.x and a.y >= a.z:
			return Vector2(p.x, p.z)
		if a.x >= a.z:
			return Vector2(p.z, -p.y)
		return Vector2(p.x, -p.y)

	# Godot-space triangle facing `n` (wound clockwise seen from `n`).
	func tri_g(a: Vector3, b: Vector3, c: Vector3, n: Vector3) -> void:
		if (b - a).cross(c - a).dot(n) > 0.0:
			var t := b
			b = c
			c = t
		for p in [a, b, c]:
			st.set_normal(n)
			st.set_uv(_uv(p, n))
			st.add_vertex(p)
			faces.push_back(p)
		count += 1
		area += 0.5 * (b - a).cross(c - a).length()

	# plan-space triangle; its normal comes from the corner order unless
	# `n_plan` is given (then the winding is fixed to face it)
	func tri(a: Vector3, b: Vector3, c: Vector3, n_plan := Vector3.ZERO) -> void:
		var ga := DemoGeo.P(a.x, a.y, a.z)
		var gb := DemoGeo.P(b.x, b.y, b.z)
		var gc := DemoGeo.P(c.x, c.y, c.z)
		var n: Vector3
		if n_plan == Vector3.ZERO:
			n = (gb - ga).cross(gc - ga).normalized()
		else:
			n = DemoGeo.P(n_plan.x, n_plan.y, n_plan.z).normalized()
		tri_g(ga, gb, gc, n)

	func quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3, n_plan := Vector3.ZERO) -> void:
		if n_plan == Vector3.ZERO:
			n_plan = (b - a).cross(c - a).normalized()
		tri(a, b, c, n_plan)
		tri(a, c, d, n_plan)

	# Axis-aligned box face set (plan coords); `skip` lists faces to omit:
	# "x-", "x+", "y-", "y+", "z-", "z+".
	func box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, skip := []) -> void:
		var v := func(x: float, y: float, z: float) -> Vector3: return Vector3(x, y, z)
		if not "x-" in skip: quad(v.call(x0, y0, z0), v.call(x0, y1, z0), v.call(x0, y1, z1), v.call(x0, y0, z1), Vector3(-1, 0, 0))
		if not "x+" in skip: quad(v.call(x1, y0, z0), v.call(x1, y1, z0), v.call(x1, y1, z1), v.call(x1, y0, z1), Vector3(1, 0, 0))
		if not "y-" in skip: quad(v.call(x0, y0, z0), v.call(x1, y0, z0), v.call(x1, y0, z1), v.call(x0, y0, z1), Vector3(0, -1, 0))
		if not "y+" in skip: quad(v.call(x0, y1, z0), v.call(x1, y1, z0), v.call(x1, y1, z1), v.call(x0, y1, z1), Vector3(0, 1, 0))
		if not "z-" in skip: quad(v.call(x0, y0, z0), v.call(x1, y0, z0), v.call(x1, y1, z0), v.call(x0, y1, z0), Vector3(0, 0, -1))
		if not "z+" in skip: quad(v.call(x0, y0, z1), v.call(x1, y0, z1), v.call(x1, y1, z1), v.call(x0, y1, z1), Vector3(0, 0, 1))

	# A box under an arbitrary transform (Godot space): rotated props.
	func box_xf(xf: Transform3D, size: Vector3) -> void:
		var h := size * 0.5
		var c := func(sx: float, sy: float, sz: float) -> Vector3: return xf * Vector3(sx * h.x, sy * h.y, sz * h.z)
		var faces6 := [
			[Vector3(1, 0, 0), [c.call(1, -1, -1), c.call(1, 1, -1), c.call(1, 1, 1), c.call(1, -1, 1)]],
			[Vector3(-1, 0, 0), [c.call(-1, -1, -1), c.call(-1, 1, -1), c.call(-1, 1, 1), c.call(-1, -1, 1)]],
			[Vector3(0, 1, 0), [c.call(-1, 1, -1), c.call(1, 1, -1), c.call(1, 1, 1), c.call(-1, 1, 1)]],
			[Vector3(0, -1, 0), [c.call(-1, -1, -1), c.call(1, -1, -1), c.call(1, -1, 1), c.call(-1, -1, 1)]],
			[Vector3(0, 0, 1), [c.call(-1, -1, 1), c.call(1, -1, 1), c.call(1, 1, 1), c.call(-1, 1, 1)]],
			[Vector3(0, 0, -1), [c.call(-1, -1, -1), c.call(1, -1, -1), c.call(1, 1, -1), c.call(-1, 1, -1)]],
		]
		for f in faces6:
			var n: Vector3 = (xf.basis * f[0]).normalized()
			var q: Array = f[1]
			tri_g(q[0], q[1], q[2], n)
			tri_g(q[0], q[2], q[3], n)

# ---------------------------------------------------------------- the set

var root: Node3D
var batches := {}
# [visual name, declared acoustic, acoustic used, m², why] per batch
var audit_rows := []

func _init(r: Node3D) -> void:
	root = r

# A batch: `acoustic` "auto" = what the visual material declares,
# "" = decor (say `why` if it is big), or an explicit preset name.
func B(vis: Material, acoustic := "auto", collide := true, boxes := false, why := "") -> Batch:
	if acoustic == "auto":
		acoustic = DemoMats.acoustic_of(vis)
	var key := "%d|%s|%s|%s" % [vis.get_instance_id(), acoustic, collide, boxes]
	if not batches.has(key):
		batches[key] = Batch.new(vis, acoustic, collide, boxes, why)
	elif why != "":
		batches[key].why = why
	return batches[key]

# A batch of its own (one mesh): an acoustic BOX must be one piece of
# furniture, not every sofa of the level merged into one bounding box.
func solo(vis: Material, acoustic := "auto", boxes := true) -> Batch:
	if acoustic == "auto":
		acoustic = DemoMats.acoustic_of(vis)
	var b := Batch.new(vis, acoustic, true, boxes)
	batches["solo%d" % batches.size()] = b
	return b

# Texture ↔ sound audit: every visual→acoustic pairing, plus warnings for
# big decor (> 2 m², not part of the sound world) without a reason.
func audit() -> Dictionary:
	var pairs := {}
	var warnings: Array[String] = []
	for r in audit_rows:
		var line := "%s (declares %s) → %s" % [r[0], r[1] if r[1] != "" else "-", r[2] if r[2] != "" else "decor"]
		pairs[line] = pairs.get(line, 0.0) + r[3]
		if r[2] == "" and r[3] > 2.0 and r[4] == "":
			warnings.append("decor %s: %.1f m² with no reason — tag it or say why" % [r[0], r[3]])
	var lines: Array[String] = []
	for k in pairs:
		lines.append("%s  (%.0f m²)" % [k, pairs[k]])
	lines.sort()
	return {"pairs": lines, "warnings": warnings}

# Commit every batch as meshes (+ colliders) under OmgGeometry tags.
func commit() -> void:
	var groups := {}
	for key in batches:
		var b: Batch = batches[key]
		if b.count == 0:
			continue
		var gkey := "%s|%s" % [b.acoustic, b.boxes]
		if not groups.has(gkey):
			var g: Node3D
			if b.acoustic == "":
				g = Node3D.new()
				g.name = "Decor"
				# explicit opt-out: decor keeps its look without its
				# material's acoustics (OmgWorld would read the meta)
				g.set_meta("omg_material", "none")
			else:
				var og := OmgGeometry.new()
				og.material = b.acoustic
				og.as_boxes = b.boxes
				og.name = "Acoustic_" + b.acoustic + ("_boxes" if b.boxes else "")
				g = og
			root.add_child(g)
			groups[gkey] = g
		audit_rows.append([b.vis.resource_name, DemoMats.acoustic_of(b.vis), b.acoustic, b.area, b.why])
		b.st.generate_tangents()
		var mesh := b.st.commit()
		var mi := MeshInstance3D.new()
		mi.mesh = mesh
		mi.material_override = b.vis
		groups[gkey].add_child(mi)
		if b.collide:
			var body := StaticBody3D.new()
			var cs := CollisionShape3D.new()
			var shape := ConcavePolygonShape3D.new()
			shape.backface_collision = true
			shape.set_faces(b.faces)
			cs.shape = shape
			body.add_child(cs)
			mi.add_child(body)
	batches.clear()

# ---------------------------------------------------------------- walls

# A wall in the plane x = `plane` (axis 0, spanning y a0..a1) or
# y = `plane` (axis 1, spanning x), z0..z1, thickness t centred on the
# plane, extended by t/2 past both ends so corners close. `holes`:
# [[center_along, half_width, z_lo, z_hi], …]. `inner` = +1/−1: the room
# lies on the +/− side of the plane (that face gets `bin`), 0 = none.
func wall(axis: int, plane: float, a0: float, a1: float, z0: float, z1: float, t: float,
		holes: Array, bin: Batch, bout: Batch, inner := 0) -> void:
	var e := t * 0.5
	var cuts: Array[float] = [a0 - e, a1 + e]
	for h in holes:
		cuts.append(h[0] - h[1])
		cuts.append(h[0] + h[1])
	cuts.sort()
	for i in range(cuts.size() - 1):
		var s0: float = cuts[i]
		var s1: float = cuts[i + 1]
		if s1 - s0 < 1e-3:
			continue
		var mid := 0.5 * (s0 + s1)
		var spans := [[z0, z1]]
		for h in holes:
			if mid > h[0] - h[1] and mid < h[0] + h[1]:
				spans = []
				if h[2] > z0 + 1e-3:
					spans.append([z0, h[2]])
				if h[3] < z1 - 1e-3:
					spans.append([h[3], z1])
		for sp in spans:
			_wall_piece(axis, plane, s0, s1, sp[0], sp[1], t, bin, bout, inner)

func _wall_piece(axis: int, plane: float, s0: float, s1: float, z0: float, z1: float, t: float,
		bin: Batch, bout: Batch, inner: int) -> void:
	var e := t * 0.5
	var in_face := ""
	if inner != 0:
		in_face = ("x" if axis == 0 else "y") + ("+" if inner > 0 else "-")
	var all := ["x-", "x+", "y-", "y+", "z-", "z+"]
	var skip_out := [in_face] if in_face != "" else []
	var skip_in := all.filter(func(f): return f != in_face)
	if axis == 0:
		bout.box(plane - e, s0, z0, plane + e, s1, z1, skip_out)
		if in_face != "":
			bin.box(plane - e, s0, z0, plane + e, s1, z1, skip_in)
	else:
		bout.box(s0, plane - e, z0, s1, plane + e, z1, skip_out)
		if in_face != "":
			bin.box(s0, plane - e, z0, s1, plane + e, z1, skip_in)

# A horizontal slab x0..x1 × y0..y1, z0..z1 minus rectangular holes
# [[hx0, hy0, hx1, hy1], …]: bottom face → `bunder`, the rest → `btop`.
func slab(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, holes: Array,
		bunder: Batch, btop: Batch) -> void:
	var xs: Array[float] = [x0, x1]
	var ys: Array[float] = [y0, y1]
	for h in holes:
		xs.append_array([clampf(h[0], x0, x1), clampf(h[2], x0, x1)])
		ys.append_array([clampf(h[1], y0, y1), clampf(h[3], y0, y1)])
	xs.sort()
	ys.sort()
	for i in range(xs.size() - 1):
		for j in range(ys.size() - 1):
			var cx := 0.5 * (xs[i] + xs[i + 1])
			var cy := 0.5 * (ys[j] + ys[j + 1])
			if xs[i + 1] - xs[i] < 1e-3 or ys[j + 1] - ys[j] < 1e-3:
				continue
			var cut := false
			for h in holes:
				if cx > h[0] and cx < h[2] and cy > h[1] and cy < h[3]:
					cut = true
			if cut:
				continue
			btop.box(xs[i], ys[j], z0, xs[i + 1], ys[j + 1], z1, ["z-"])
			bunder.box(xs[i], ys[j], z0, xs[i + 1], ys[j + 1], z1, ["x-", "x+", "y-", "y+", "z+"])

# A prism from a plan-space polygon profile in the x-z plane swept along
# y0..y1 (gables, roof slabs): `pts` are (x, z) corners, convex.
func prism_xz(pts: Array, y0: float, y1: float, b: Batch) -> void:
	var c := Vector2.ZERO
	for p in pts:
		c += p
	c /= pts.size()
	for i in pts.size():
		var p: Vector2 = pts[i]
		var q: Vector2 = pts[(i + 1) % pts.size()]
		# end caps (fan from the centroid)
		b.tri(Vector3(c.x, y0, c.y), Vector3(p.x, y0, p.y), Vector3(q.x, y0, q.y), Vector3(0, -1, 0))
		b.tri(Vector3(c.x, y1, c.y), Vector3(p.x, y1, p.y), Vector3(q.x, y1, q.y), Vector3(0, 1, 0))
		# side
		var e := q - p
		var n := Vector2(e.y, -e.x).normalized()
		if n.dot(p - c) < 0.0:
			n = -n
		b.quad(Vector3(p.x, y0, p.y), Vector3(q.x, y0, q.y), Vector3(q.x, y1, q.y), Vector3(p.x, y1, p.y), Vector3(n.x, 0, n.y))

# A semicircular arch (rectangular section w × d) in a vertical plane:
# centre (cx, cy, cz) plan, radius r, spanning along x (axis 0) or y.
func arch(cx: float, cy: float, cz: float, r: float, w: float, d: float, axis: int, b: Batch, segs := 12) -> void:
	for k in segs:
		var a0 := PI * k / segs
		var a1 := PI * (k + 1) / segs
		var pts := []
		for a in [a0, a1]:
			for rr in [r - w * 0.5, r + w * 0.5]:
				pts.append(Vector2(cos(a) * rr, sin(a) * rr))
		# pts: (a0 in, a0 out, a1 in, a1 out)
		var corners := []
		for s in [-0.5 * d, 0.5 * d]:
			for p in pts:
				if axis == 0:
					corners.append(Vector3(cx + p.x, cy + s, cz + p.y))
				else:
					corners.append(Vector3(cx + s, cy + p.x, cz + p.y))
		# corner index: s*4 + (a*2 + rr)
		var f := func(i: int, j: int, k2: int, l: int, n: Vector3) -> void:
			b.quad(corners[i], corners[j], corners[k2], corners[l], n)
		var am := (a0 + a1) * 0.5
		var radial := Vector3(cos(am), 0, sin(am)) if axis == 0 else Vector3(0, cos(am), sin(am))
		var side := Vector3(0, 1, 0) if axis == 0 else Vector3(1, 0, 0)
		f.call(1, 3, 7, 5, radial)      # outer
		f.call(0, 2, 6, 4, -radial)     # inner
		f.call(4, 5, 7, 6, side)        # +side
		f.call(0, 1, 3, 2, -side)       # −side
