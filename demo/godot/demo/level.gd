# The demo town, generated: the same layout as the web demo (rooms, walls,
# doors, windows and landmarks of web/main.js + crates/omg-scene's
# walkthrough), dressed with PBR materials and lights. Every piece of
# architecture is tagged with an acoustic material via OmgGeometry, so
# the sound engine hears the very geometry you see; props that only
# decorate (bells, figures, speaker cabinets, arches) stay untagged.
#
# Plan coordinates throughout (x east, y north, z up), see DemoGeo.P.
class_name DemoLevel
extends RefCounted

# Room plan (walkthrough.rs rooms()): name, min, max, floor z, ceiling
# z, footstep surface. Order matters: the first match wins.
const ROOMS := [
	["Living Room", Vector2(0, 0), Vector2(8, 6), 0.0, 2.7, "wood"],
	["Corridor", Vector2(3.2, 6), Vector2(4.8, 14), 0.0, 2.4, "wood"],
	["Great Hall", Vector2(0, 14), Vector2(14, 24), 0.0, 7.0, "concrete"],
	["Entrance", Vector2(20, 28), Vector2(22, 34), 0.0, 2.6, "concrete"],
	["Club", Vector2(22, 26), Vector2(32, 38), 0.0, 4.5, "concrete"],
	["Old House Upper", Vector2(24, 16), Vector2(31, 23), 2.9, 5.6, "carpet"],
	["Old House", Vector2(24, 16), Vector2(31, 23), -0.5, 2.8, "wood"],
	["Cathedral", Vector2(0, 52), Vector2(16, 74), 0.0, 23.0, "concrete"],
	["Bunker", Vector2(34, 6), Vector2(40, 14), -3.5, -0.8, "concrete"],
	["Bunker Stair", Vector2(31.4, 6), Vector2(34, 8), -3.5, 2.4, "concrete"],
	["Kiosk", Vector2(14.5, 34), Vector2(16.5, 36), 0.0, 2.7, "wood"],
]

# Toggleable doors (E), web DOORS order: pos, axis (0: in an x=const
# wall), half width, height, heavy. 4 (club inner) and 6 (cathedral
# portal) start closed, as in the web.
const DOORS := [
	[Vector2(4, 6), 1, 0.55, 2.0, false],
	[Vector2(4, 14), 1, 0.55, 2.0, false],
	[Vector2(7, 24), 1, 0.55, 2.0, false],
	[Vector2(20, 31), 0, 0.55, 2.0, true],
	[Vector2(22, 31), 0, 0.55, 2.0, true],
	[Vector2(26.5, 23), 1, 0.55, 2.0, false],
	[Vector2(8, 52), 1, 1.2, 3.5, true],
	[Vector2(31.4, 7), 0, 0.7, 2.0, true],
]
const DOOR_OPEN := [true, true, true, true, false, true, false, true]

# Glass panes (walkthrough doors with glass): pos, axis, half, z centre, heavy.
const WINDOWS := [
	[Vector2(3.0, 0.0), 1, 1.3, 1.5, false],
	[Vector2(32.0, 32.0), 0, 1.8, 1.5, true],
	[Vector2(26.0, 38.0), 1, 1.8, 1.5, true],
	[Vector2(29.3, 23.0), 1, 1.4, 1.5, false],
	[Vector2(24.0, 19.5), 0, 1.4, 1.5, false],
	[Vector2(29.3, 23.0), 1, 1.4, 4.5, false],
	[Vector2(24.0, 19.5), 0, 1.4, 4.5, false],
]
const WIN_H := 1.4

# Club PA: four corner speakers (walkthrough CLUB_SPEAKERS).
const CLUB_SPEAKERS := [Vector2(23.5, 27.5), Vector2(30.5, 27.5), Vector2(23.5, 36.5), Vector2(30.5, 36.5)]
# The garden fountain beside the belfry path (walkthrough FOUNTAIN).
const FOUNTAIN := Vector2(14.2, 199.0)

static func room_at(x: float, y: float, z: float) -> Array:
	for r in ROOMS:
		var mn: Vector2 = r[1]
		var mx: Vector2 = r[2]
		if x > mn.x and x < mx.x and y > mn.y and y < mx.y and z >= r[3] and z < r[4]:
			return r
	return ["Outside", Vector2.ZERO, Vector2.ZERO, 0.0, 0.0, "grass"]

static func v(x: float, y: float, z: float) -> Vector3:
	return DemoGeo.P(x, y, z)

var geo: DemoGeo
var root: Node3D
# filled while building, used by the demo
var door_nodes := []        # [{hinge, shape, omg, pos, hw, open, target, axis}]
var curtain_sets := []      # [{string, curtains: [{omg, cloth, roll, top, h}], open, target}]
var markers := {}           # decorative objects the demo animates
var labels := false         # name furniture kinds in the scene (shots)
var lights := []

func _init(r: Node3D) -> void:
	root = r
	geo = DemoGeo.new(r)

func build() -> void:
	_ground()
	_living_corridor_hall()
	_club()
	_old_house()
	_cathedral()
	_bunker()
	_belfry()
	_square_props()
	_furniture()
	_props()
	_windows()
	_doors()
	_curtains()
	_street()
	_canyon()
	_lamps()
	_hall_lights()
	_trees()
	_fountain()
	geo.commit()

# ------------------------------------------------------------------ ground

func _ground() -> void:
	var g := geo.B(DemoMats.grass(), "grass")
	# town tile (fine ray-tracing patches) with the bunker shaft cut out
	var h := [31.4, 6.0, 34.0, 8.0]
	var t := [-60.0, -60.0, 90.0, 400.0]
	for r in [[t[0], t[1], t[2], h[1]], [t[0], h[3], t[2], t[3]], [t[0], h[1], h[0], h[3]], [h[2], h[1], t[2], h[3]]]:
		g.quad(Vector3(r[0], r[1], 0), Vector3(r[2], r[1], 0), Vector3(r[2], r[3], 0), Vector3(r[0], r[3], 0), Vector3(0, 0, 1))
	# far field: the plain runs on to the horizon (decor: kilometres of
	# fine ground triangles would only bloat the tracer's BVH)
	var far := 4000.0
	g = geo.B(DemoMats.grass(), "", true, false, "far ground: kept out of the tracer BVH")
	for r in [[-far, -far, far, t[1]], [-far, t[3], far, far], [-far, t[1], t[0], t[3]], [t[2], t[1], far, t[3]]]:
		g.quad(Vector3(r[0], r[1], 0), Vector3(r[2], r[1], 0), Vector3(r[2], r[3], 0), Vector3(r[0], r[3], 0), Vector3(0, 0, 1))
	# paving: the square, the paths — acoustic paving slabs on the grass,
	# from layout.json's "ground" (the street and kerbs: _street)
	var c := geo.B(DemoMats.cobbles(), "auto", false)
	var path := geo.B(DemoMats.cobbles(3.0, Color(0.75, 0.72, 0.65)), "auto", false)
	for pv in _layout_ground():
		if pv["surface"] != "paving":
			continue
		var mn: Vector2 = pv["min"]
		var mx: Vector2 = pv["max"]
		if pv["name"] == "belfry path":
			path.box(mn.x, mn.y, 0.0, mx.x, mx.y, 0.02, ["z-"])
		else:
			c.box(mn.x, mn.y, 0.0, mx.x, mx.y, 0.025, ["z-"])

# ------------------------------------------------- living room, corridor, hall

func _living_corridor_hall() -> void:
	var brick := geo.B(DemoMats.brick(), "brick")
	var warm := geo.B(DemoMats.plaster(Color(0.95, 0.88, 0.76), "brick"))
	var conc_out := geo.B(DemoMats.blocks(Color(0.78, 0.76, 0.74)), "concrete")
	var conc_in := geo.B(DemoMats.plaster(Color(0.88, 0.88, 0.86), "concrete"))
	var cor_in := geo.B(DemoMats.plaster(Color(0.82, 0.86, 0.9), "concrete"))
	var door := func(x: float) -> Array: return [x, 0.55, 0.0, 2.0]
	var win := func(x: float, hw: float, zc: float) -> Array: return [x, hw, zc - WIN_H * 0.5, zc + WIN_H * 0.5]

	# living room: brick, warm plaster inside (barrier 3.1: parapet)
	var t := 0.24
	geo.wall(1, 0, 0, 8, 0, 3.1, t, [win.call(3.0, 1.3, 1.5)], warm, brick, 1)
	geo.wall(0, 0, 0, 6, 0, 3.1, t, [], warm, brick, 1)
	geo.wall(0, 8, 0, 6, 0, 3.1, t, [], warm, brick, -1)
	geo.wall(1, 6, 0, 8, 0, 3.1, t, [door.call(4.0)], warm, brick, -1)
	# parquet floor (the rug is carpet, see _furniture), tiled ceiling
	geo.B(DemoMats.parquet()).box(0, 0, -0.1, 8, 6, 0.02)
	geo.slab(0, 0, 8, 6, 2.7, 2.9, [], geo.B(DemoMats.plaster(Color(0.96, 0.95, 0.92), "acoustic_tile")),
		geo.B(DemoMats.concrete(Color(0.45, 0.45, 0.47)), "concrete"))

	# corridor
	geo.wall(0, 3.2, 6, 14, 0, 3.1, 0.2, [], cor_in, conc_out, 1)
	geo.wall(0, 4.8, 6, 14, 0, 3.1, 0.2, [], cor_in, conc_out, -1)
	geo.B(DemoMats.floorboards(), "wood").box(3.2, 6, -0.1, 4.8, 14, 0.02)
	geo.slab(3.2, 6, 4.8, 14, 2.4, 2.6, [], geo.B(DemoMats.plaster(Color(0.9, 0.9, 0.9), "concrete")),
		geo.B(DemoMats.concrete(Color(0.45, 0.45, 0.47)), "concrete"))

	# great hall: 7 m concrete hall (seminar room)
	t = 0.3
	geo.wall(1, 14, 0, 14, 0, 7.4, t, [door.call(4.0)], conc_in, conc_out, 1)
	geo.wall(0, 0, 14, 24, 0, 7.4, t, [], conc_in, conc_out, 1)
	geo.wall(0, 14, 14, 24, 0, 7.4, t, [], conc_in, conc_out, -1)
	geo.wall(1, 24, 0, 14, 0, 7.4, t, [door.call(7.0)], conc_in, conc_out, -1)
	geo.B(DemoMats.concrete(Color(0.6, 0.58, 0.55)), "concrete").box(0, 14, -0.1, 14, 24, 0.02)
	geo.slab(0, 14, 14, 24, 7.0, 7.2, [], conc_in, geo.B(DemoMats.concrete(Color(0.45, 0.45, 0.47)), "concrete"))
	_cornice(0, 14, 14, 24, 7.4)
	_cornice(0, 0, 8, 6, 3.1)

# a stone coping along a flat roof's parapet
func _cornice(x0: float, y0: float, x1: float, y1: float, z: float) -> void:
	var b := geo.B(DemoMats.sandstone(), "", false, false, "parapet coping (trim)")
	var e := 0.22
	b.box(x0 - e, y0 - e, z, x1 + e, y0 + e, z + 0.12)
	b.box(x0 - e, y1 - e, z, x1 + e, y1 + e, z + 0.12)
	b.box(x0 - e, y0 + e, z, x0 + e, y1 - e, z + 0.12)
	b.box(x1 - e, y0 + e, z, x1 + e, y1 - e, z + 0.12)

# ------------------------------------------------------------------- club

func _club() -> void:
	var out := geo.B(DemoMats.blocks(Color(0.42, 0.42, 0.46)), "concrete")
	var dark := geo.B(DemoMats.concrete(Color(0.16, 0.15, 0.18)), "concrete")
	var lobby := geo.B(DemoMats.plaster(Color(0.55, 0.32, 0.42), "concrete"))
	var door := func(y: float) -> Array: return [y, 0.55, 0.0, 2.0]
	# entrance lobby
	geo.wall(0, 20, 28, 34, 0, 3.0, 0.2, [door.call(31.0)], lobby, out, 1)
	geo.wall(1, 28, 20, 22, 0, 3.0, 0.2, [], lobby, out, 1)
	geo.wall(1, 34, 20, 22, 0, 3.0, 0.2, [], lobby, out, -1)
	geo.slab(20, 28, 22, 34, 2.6, 2.8, [], lobby, geo.B(DemoMats.concrete(Color(0.4, 0.4, 0.42)), "concrete"))
	geo.B(DemoMats.concrete(Color(0.3, 0.28, 0.3)), "concrete").box(20, 28, -0.1, 22, 34, 0.02)
	# the club hall (west wall shared with the lobby: one wall, as the web)
	var t := 0.35
	geo.wall(0, 22, 26, 38, 0, 4.9, t, [door.call(31.0)], dark, out, 1)
	geo.wall(0, 32, 26, 38, 0, 4.9, t, [[32.0, 1.8, 0.8, 2.2]], dark, out, -1)
	geo.wall(1, 26, 22, 32, 0, 4.9, t, [], dark, out, 1)
	geo.wall(1, 38, 22, 32, 0, 4.9, t, [[26.0, 1.8, 0.8, 2.2]], dark, out, -1)
	geo.B(DemoMats.concrete(Color(0.22, 0.2, 0.22)), "concrete").box(22, 26, -0.1, 32, 38, 0.02)
	geo.slab(22, 26, 32, 38, 4.5, 4.7, [], dark, geo.B(DemoMats.concrete(Color(0.4, 0.4, 0.42)), "concrete"))
	_cornice(22, 26, 32, 38, 4.9)
	# neon sign over the entrance
	var sign := Label3D.new()
	sign.text = "CLUB"
	sign.font_size = 160
	sign.pixel_size = 0.004
	sign.modulate = Color(1.0, 0.35, 0.75)
	sign.outline_modulate = Color(0.5, 0.0, 0.3)
	sign.position = v(19.85, 31, 3.4)
	sign.rotation_degrees = Vector3(0, -90, 0)
	root.add_child(sign)
	var neon := OmniLight3D.new()
	neon.position = v(19.4, 31, 3.3)
	neon.light_color = Color(1.0, 0.3, 0.7)
	neon.light_energy = 2.0
	neon.omni_range = 7.0
	root.add_child(neon)
	# LED strips along the club ceiling + coloured wash lights
	var led := geo.B(DemoMats.glow(Color(0.9, 0.2, 1.0), 4.0), "", false, false, "LED strips (light)")
	led.box(22.3, 26.4, 4.3, 31.7, 26.5, 4.36)
	led.box(22.3, 37.5, 4.3, 31.7, 37.6, 4.36)
	var cols := [Color(1.0, 0.2, 0.6), Color(0.2, 0.4, 1.0), Color(0.1, 0.9, 0.8), Color(0.8, 0.2, 1.0)]
	var i := 0
	for p in [Vector2(25, 29), Vector2(29, 29), Vector2(25, 35), Vector2(29, 35)]:
		var l := OmniLight3D.new()
		l.position = v(p.x, p.y, 4.0)
		l.light_color = cols[i]
		l.light_energy = 3.0
		l.omni_range = 7.0
		root.add_child(l)
		lights.append(l)
		i += 1
	markers["club_lights"] = lights.duplicate()
	# PA cabinets at the four emitters (decor: the source sits inside)
	var cab := geo.B(DemoMats.plain(Color(0.05, 0.05, 0.06), 0.6), "", true, false, "PA cabinets: the club sources sit inside them")
	var cone := geo.B(DemoMats.plain(Color(0.12, 0.12, 0.13), 0.4), "", false, false, "speaker cones")
	for s in CLUB_SPEAKERS:
		cab.box(s.x - 0.35, s.y - 0.28, 1.35, s.x + 0.35, s.y + 0.28, 2.45)
		var face := -1.0 if s.x > 27 else 1.0
		cone.box(s.x + face * 0.36 - 0.02, s.y - 0.2, 1.5, s.x + face * 0.36 + 0.02, s.y + 0.2, 2.3)
		cab.box(s.x - 0.08, s.y - 0.08, 0.0, s.x + 0.08, s.y + 0.08, 1.35)

# -------------------------------------------------------------- old house

func _old_house() -> void:
	var brick := geo.B(DemoMats.brick(), "brick")
	var inside := geo.B(DemoMats.plaster(Color(0.92, 0.86, 0.74), "brick"))
	var t := 0.25
	var w_lo := func(x: float, hw: float) -> Array: return [x, hw, 0.8, 2.2]
	# one wall per side, both storeys (0 → 5.9 incl. parapet)
	geo.wall(1, 16, 24, 31, 0, 5.9, t, [], inside, brick, 1)
	geo.wall(0, 24, 16, 23, 0, 5.9, t, [w_lo.call(19.5, 1.4), [19.5, 1.4, 3.8, 5.2]], inside, brick, 1)
	geo.wall(0, 31, 16, 23, 0, 5.9, t, [], inside, brick, -1)
	geo.wall(1, 23, 24, 31, 0, 5.9, t, [[26.5, 0.55, 0.0, 2.0], w_lo.call(29.3, 1.4), [29.3, 1.4, 3.8, 5.2]], inside, brick, -1)
	# ground floor, the upper slab with its stairwell along the west wall
	geo.B(DemoMats.floorboards(), "wood").box(24, 16, -0.1, 31, 23, 0.02)
	var under := geo.B(DemoMats.plaster(Color(0.95, 0.93, 0.88), "brick"))
	var boards := geo.B(DemoMats.floorboards(), "wood")
	geo.slab(24, 16, 31, 23, 2.8, 3.0, [[24.0, 16.9, 25.7, 21.3]], under, boards)
	# stairs: 13 treads rising north to the landing (decor + a ramp collider)
	var tread := geo.B(DemoMats.planks(Color(0.8, 0.7, 0.6)), "auto", false)
	for k in 13:
		var z := 3.0 * (k + 1) / 13.0
		var y0 := 17.0 + 4.3 * k / 13.0
		tread.box(24.15, y0, z - 0.2, 25.6, y0 + 4.3 / 13.0 + 0.02, z)
	# balustrade along the stairwell: a handrail on posts (it also keeps
	# walkers upstairs from stepping into the hole)
	var rail := geo.B(DemoMats.planks(Color(0.55, 0.4, 0.3)), "", false, false, "handrail")
	rail.box(25.6, 16.9, 3.88, 25.7, 21.3, 3.95)
	var y := 16.95
	while y < 21.3:
		rail.box(25.62, y, 3.0, 25.68, y + 0.05, 3.88)
		y += 0.36
	var guard := StaticBody3D.new()
	var gs := CollisionShape3D.new()
	var gb := BoxShape3D.new()
	gb.size = Vector3(0.1, 1.0, 4.4)
	gs.shape = gb
	guard.add_child(gs)
	guard.position = v(25.65, 19.1, 3.5)
	root.add_child(guard)
	_ramp(Vector3(24.85, 17.0, 0.0), Vector3(24.85, 21.3, 3.0), 1.45)
	geo.slab(24, 16, 31, 23, 5.6, 5.8, [], under, geo.B(DemoMats.concrete(Color(0.35, 0.33, 0.33))))
	_cornice(24, 16, 31, 23, 5.9)
	# chimney
	var ch := geo.B(DemoMats.brick())
	ch.box(29.6, 21.4, 5.8, 30.4, 22.2, 7.0)

# an invisible walkable ramp (stairs, bunker steps) from a to b, width w
func _ramp(a: Vector3, b: Vector3, w: float) -> void:
	var body := StaticBody3D.new()
	var cs := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	var ga := v(a.x, a.y, a.z)
	var gb := v(b.x, b.y, b.z)
	var len := ga.distance_to(gb)
	shape.size = Vector3(w, 0.1, len)
	cs.shape = shape
	body.add_child(cs)
	root.add_child(body)
	body.look_at_from_position((ga + gb) * 0.5 + Vector3(0, -0.05, 0), gb + Vector3(0, -0.05, 0), Vector3.UP)

# --------------------------------------------------------------- cathedral

# The nave section exactly as the engine's walkthrough meshes it: walls
# to the eaves (19 m), a round barrel vault springing at 15 m (8 facets),
# a 50° gable roof up to the 28.5 m ridge.
const CATH := [0.0, 52.0, 16.0, 74.0]
const SPRING := 15.0
const EAVE := 19.0
const RIDGE := 28.5

func _cathedral() -> void:
	var x0: float = CATH[0]
	var y0: float = CATH[1]
	var x1: float = CATH[2]
	var y1: float = CATH[3]
	var mid := 0.5 * (x0 + x1)
	var outside := geo.B(DemoMats.sandstone(), "stone")
	var inside := geo.B(DemoMats.sandstone(3.0, Color(0.78, 0.74, 0.68)))
	var t := 0.5
	geo.wall(1, y0, x0, x1, 0, EAVE, t, [[8.0, 1.2, 0.0, 3.5]], inside, outside, 1)
	geo.wall(1, y1, x0, x1, 0, EAVE, t, [], inside, outside, -1)
	geo.wall(0, x0, y0, y1, 0, EAVE, t, [], inside, outside, 1)
	geo.wall(0, x1, y0, y1, 0, EAVE, t, [], inside, outside, -1)
	geo.B(DemoMats.marble()).box(x0, y0, -0.1, x1, y1, 0.02)
	# gables: eaves → ridge on both end walls
	for y in [y0, y1]:
		geo.prism_xz([Vector2(x0 - 0.25, EAVE), Vector2(x1 + 0.25, EAVE), Vector2(mid, RIDGE)], y - t * 0.5, y + t * 0.5, outside)
	# the barrel vault: 8 flat strips (inscribed half-polygon)
	# one face per strip (a thin shell, seen from both sides): coincident
	# back faces would read to the engine as a zero-thickness wall
	var vault_mat: BaseMaterial3D = DemoMats.plaster(Color(0.86, 0.82, 0.74), "stone").duplicate()
	vault_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	var vault := geo.B(vault_mat, "stone")
	var r := 0.5 * (x1 - x0)
	var prof := []
	for k in 9:
		var th := PI * (1.0 - k / 8.0)
		prof.append(Vector2(mid + r * cos(th), SPRING + r * sin(th)))
	for k in 8:
		var a: Vector2 = prof[k]
		var b: Vector2 = prof[k + 1]
		var n := Vector2(mid - 0.5 * (a.x + b.x), SPRING - 0.5 * (a.y + b.y)).normalized()
		vault.quad(Vector3(a.x, y0, a.y), Vector3(b.x, y0, b.y), Vector3(b.x, y1, b.y), Vector3(a.x, y1, a.y), Vector3(n.x, 0, n.y))
	# roof: two slate pitches with an overhang, 0.25 m thick
	var slope := (RIDGE - EAVE) / (mid - x0)
	var ov := 0.7
	for s in [-1.0, 1.0]:
		var e: float = mid + s * (r + ov)
		var ez := EAVE - ov * slope
		var pts := [Vector2(e, ez), Vector2(mid, RIDGE), Vector2(mid, RIDGE + 0.3), Vector2(e, ez + 0.3)]
		geo.prism_xz(pts, y0 - 0.6, y1 + 0.6, geo.B(DemoMats.slate(), "stone"))
	# ridge cap
	geo.B(DemoMats.metal(Color(0.25, 0.27, 0.28)), "", false, false, "ridge cap (trim)").box(mid - 0.15, y0 - 0.6, RIDGE + 0.2, mid + 0.15, y1 + 0.6, RIDGE + 0.42)
	_cathedral_glass(x0, y0, x1)
	_cathedral_interior()
	# stepped buttresses at every bay line
	var butt := geo.B(DemoMats.sandstone(), "stone")
	var y := 55.0
	while y <= 71.01:
		butt.box(x0 - 1.2, y - 0.45, 0, x0, y + 0.45, 11)
		butt.box(x0 - 0.7, y - 0.35, 11, x0, y + 0.35, 16)
		butt.box(x1, y - 0.45, 0, x1 + 1.2, y + 0.45, 11)
		butt.box(x1, y - 0.35, 11, x1 + 0.7, y + 0.35, 16)
		y += 3.2
	# portal surround and a cross on the apex
	var stone := geo.B(DemoMats.sandstone(), "auto", false)
	geo.arch(mid, y0 - 0.3, 3.5, 1.45, 0.4, 0.4, 0, stone)
	stone.box(mid - 1.65, y0 - 0.5, 0, mid - 1.25, y0 - 0.1, 3.5)
	stone.box(mid + 1.25, y0 - 0.5, 0, mid + 1.65, y0 - 0.1, 3.5)
	stone.box(mid - 0.12, y0 - 0.12, RIDGE, mid + 0.12, y0 + 0.12, RIDGE + 2.4)
	stone.box(mid - 0.65, y0 - 0.12, RIDGE + 1.55, mid + 0.65, y0 + 0.12, RIDGE + 1.8)

func _stained(w: int, h: int, rose: bool) -> ImageTexture:
	var img := Image.create(w, h, false, Image.FORMAT_RGBA8)
	var cols := [Color("2a4fa8"), Color("a8242e"), Color("d8a83a"), Color("2f7a4a"), Color("5a2f8a"), Color("3a7ab8")]
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var lead := Color("16121c")
	if rose:
		var c := Vector2(w, h) * 0.5
		var rad := w * 0.5
		for y in h:
			for x in w:
				var d := Vector2(x, y) - c
				var rr := d.length() / rad
				if rr > 1.0:
					img.set_pixel(x, y, Color(0, 0, 0, 0))
					continue
				var a := fposmod(d.angle(), TAU)
				var ring := 0 if rr < 0.62 else 1
				var k := int(floor((a + ring * PI / 12.0) / (TAU / 12.0)))
				var col: Color = cols[(k + ring * 3) % cols.size()]
				var edge := absf(fposmod((a + ring * PI / 12.0), TAU / 12.0) - TAU / 24.0) > TAU / 24.0 - 0.06
				if rr < 0.18:
					col = Color("e0b84a")
				if edge or absf(rr - 0.18) < 0.02 or absf(rr - 0.62) < 0.02 or rr > 0.97:
					col = lead
				img.set_pixel(x, y, col)
	else:
		# a round-arched lancet of leaded panes
		var pw := w / 4.0
		var ph := h / 14.0
		var pane := []
		for i in 4 * 14:
			pane.append(cols[rng.randi() % cols.size()])
		var arch_r := w * 0.5
		for y in h:
			for x in w:
				var top := y < arch_r
				var inside := true
				if top:
					inside = Vector2(x - w * 0.5, y - arch_r).length() <= arch_r
				if not inside:
					img.set_pixel(x, y, Color(0, 0, 0, 0))
					continue
				var i := mini(int(x / pw), 3)
				var j := mini(int(y / ph), 13)
				var fx := fposmod(x, pw)
				var fy := fposmod(y, ph)
				var col: Color = pane[j * 4 + i]
				if fx < 1.5 or fx > pw - 1.5 or fy < 1.5 or fy > ph - 1.5:
					col = lead
				img.set_pixel(x, y, col)
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)

func _glass_mat(tex: Texture2D, energy: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_texture = tex
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	m.alpha_scissor_threshold = 0.5
	m.emission_enabled = true
	m.emission_texture = tex
	m.emission_energy_multiplier = energy
	m.roughness = 0.2
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m

# stained glass in every bay of both flanks + the rose over the portal
# (visual: the walls are solid stone to the engine, as in the web demo)
func _cathedral_glass(x0: float, y0: float, x1: float) -> void:
	var lancet := _glass_mat(_stained(64, 320, false), 1.6)
	var rose := _glass_mat(_stained(256, 256, true), 1.8)
	var ww := 1.1
	var wh := 6.6
	for x in [x0, x1]:
		var y := 56.6
		while y < 70.5:
			for off in [-0.27, 0.27]:
				var q := MeshInstance3D.new()
				var qm := QuadMesh.new()
				qm.size = Vector2(ww, wh)
				q.mesh = qm
				q.material_override = lancet
				q.position = v(x + off, y, 4.0 + wh * 0.5)
				q.rotation_degrees = Vector3(0, 90, 0)
				root.add_child(q)
			y += 3.2
	for off in [-0.27, 0.27]:
		var q := MeshInstance3D.new()
		var qm := QuadMesh.new()
		qm.size = Vector2(4.8, 4.8)
		q.mesh = qm
		q.material_override = rose
		q.position = v(8.0, y0 + off, 13.0)
		root.add_child(q)
	var ring := geo.B(DemoMats.sandstone(), "auto", false)
	for side in [-0.32, 0.32]:
		var seg := 24
		for k in seg:
			var a0 := TAU * k / seg
			var a1 := TAU * (k + 1) / seg
			var p0 := Vector2(cos(a0), sin(a0)) * 2.5
			var p1 := Vector2(cos(a1), sin(a1)) * 2.5
			var q0 := Vector2(cos(a0), sin(a0)) * 2.85
			var q1 := Vector2(cos(a1), sin(a1)) * 2.85
			var yy: float = y0 + side
			ring.quad(Vector3(8 + p0.x, yy, 13 + p0.y), Vector3(8 + p1.x, yy, 13 + p1.y), Vector3(8 + q1.x, yy, 13 + q1.y), Vector3(8 + q0.x, yy, 13 + q0.y), Vector3(0, signf(side), 0))

func _cathedral_interior() -> void:
	var stone := geo.B(DemoMats.sandstone(1.5, Color(0.8, 0.76, 0.7)), "auto", false)
	var dark := geo.B(DemoMats.sandstone(1.5, Color(0.62, 0.58, 0.54)), "auto", false)
	var ys := []
	var y := 55.0
	while y <= 71.01:
		ys.append(y)
		y += 3.2
	var col_mesh := CylinderMesh.new()
	col_mesh.top_radius = 0.42
	col_mesh.bottom_radius = 0.5
	col_mesh.height = 8.0
	col_mesh.radial_segments = 16
	# columns: acoustic boxes (stone), as the engine's furniture pillars
	var cols := OmgGeometry.new()
	cols.material = "stone"
	cols.as_boxes = true
	cols.name = "CathedralColumns"
	root.add_child(cols)
	var col_mat := DemoMats.pbr("white_sandstone_bricks_03", 1.2, Color(0.82, 0.78, 0.72))
	for x in [4.0, 12.0]:
		for yy in ys:
			var c := MeshInstance3D.new()
			c.mesh = col_mesh
			c.material_override = col_mat
			c.position = v(x, yy, 4.0)
			cols.add_child(c)
			var body := StaticBody3D.new()
			var cs := CollisionShape3D.new()
			var sh := CylinderShape3D.new()
			sh.radius = 0.5
			sh.height = 8.0
			cs.shape = sh
			body.add_child(cs)
			c.add_child(body)
			dark.box(x - 0.65, yy - 0.65, 8.0, x + 0.65, yy + 0.65, 8.5)
			dark.box(x - 0.65, yy - 0.65, 0.0, x + 0.65, yy + 0.65, 0.5)
		for i in range(ys.size() - 1):
			geo.arch(x, 0.5 * (ys[i] + ys[i + 1]), 8.5, 1.6, 0.44, 0.9, 1, stone, 10)
			stone.box(x - 0.45, ys[i] + 0.65, 10.0, x + 0.45, ys[i + 1] - 0.65, 10.15)
			for off in [-0.85, 0.85]:
				geo.arch(x, 0.5 * (ys[i] + ys[i + 1]) + off, 10.15, 0.7, 0.2, 0.5, 1, dark, 8)
	for yy in ys:
		geo.arch(8.0, yy, 8.5, 4.0, 0.5, 0.6, 0, dark, 14)
		# ribs hugging the vault (its 8 facets, a hand's width below)
		for k in 8:
			var r := 7.75
			var a0 := PI * (1.0 - k / 8.0)
			var a1 := PI * (1.0 - (k + 1) / 8.0)
			var p0 := Vector2(8 + r * cos(a0), SPRING + r * sin(a0))
			var p1 := Vector2(8 + r * cos(a1), SPRING + r * sin(a1))
			geo.prism_xz([p0, p1, p1 + (Vector2(8, SPRING) - p1).normalized() * 0.3, p0 + (Vector2(8, SPRING) - p0).normalized() * 0.3], yy - 0.18, yy + 0.18, stone)
	# apse panels (five facets on a shallow half-ellipse behind the
	# altar, which stands at local y 19 — keep them clear of it), altar,
	# candles
	var apse := func(phi: float) -> Vector2: return Vector2(8 + 4.2 * cos(phi), 73.7 - 1.75 * sin(phi))
	for k in 5:
		var p0: Vector2 = apse.call(PI * (0.0625 + 0.175 * k))
		var p1: Vector2 = apse.call(PI * (0.0625 + 0.175 * (k + 1)))
		var m := (p0 + p1) * 0.5
		var xf := Transform3D(Basis(Vector3.UP, (p1 - p0).angle()), v(m.x, m.y, 6))
		stone.box_xf(xf, Vector3(p0.distance_to(p1) + 0.12, 12, 0.3))
	# the altar itself comes from layout.json when present (furniture);
	# its cloth and candles sit on whichever altar is there
	var ay := 70.5
	var shared_altar := _layout_item("altar")
	if shared_altar.is_empty():
		var altar := OmgGeometry.new()
		altar.material = "stone"
		altar.as_boxes = true
		altar.name = "Altar"
		root.add_child(altar)
		var am := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(2.4, 1.1, 1.2)
		am.mesh = bm
		am.material_override = DemoMats.pbr("marble_tiles", 1.0, Color(0.9, 0.88, 0.85))
		am.position = v(8, ay, 0.55)
		altar.add_child(am)
	else:
		ay = 0.5 * (shared_altar["min"].y + shared_altar["max"].y)
	var cloth := geo.B(DemoMats.fabric(Color(0.48, 0.1, 0.14)), "auto", false)
	cloth.box(6.75, ay - 0.65, 1.1, 9.25, ay + 0.65, 1.16)
	cloth.box(6.75, ay - 0.65, 0.4, 9.25, ay - 0.62, 1.16)
	var wax := geo.B(DemoMats.plain(Color(0.95, 0.92, 0.85), 0.6), "", false)
	var flame := geo.B(DemoMats.glow(Color(1.0, 0.7, 0.3), 6.0), "", false)
	var candles := []
	for cx in [6.9, 7.4, 8.6, 9.1]:
		wax.box(cx - 0.04, ay - 0.05, 1.16, cx + 0.04, ay + 0.05, 1.5)
		flame.box(cx - 0.015, ay - 0.015, 1.5, cx + 0.015, ay + 0.015, 1.56)
	for p in [Vector3(8, ay - 0.1, 2.0), Vector3(8, 60, 5.0)]:
		var l := OmniLight3D.new()
		l.position = v(p.x, p.y, p.z)
		l.light_color = Color(1.0, 0.72, 0.42)
		l.light_energy = 4.0 if p.y > 65 else 3.0
		l.omni_range = 16.0
		l.shadow_enabled = p.y > 65
		root.add_child(l)
		candles.append(l)
	var cool := OmniLight3D.new()
	cool.position = v(8, 64, 18)
	cool.light_color = Color(0.55, 0.65, 0.9)
	cool.light_energy = 1.5
	cool.omni_range = 18.0
	root.add_child(cool)
	markers["candles"] = candles
	# pews: big wooden absorbers and diffusers in a church (from
	# layout.json when present; this is the fallback)
	if not _layout_item("pew").is_empty():
		return
	var pew := geo.B(DemoMats.planks(Color(0.55, 0.4, 0.3), "furniture_wood"))
	var py2 := 54.5
	while py2 < 66.0:
		for side in [[5.4, 7.2], [8.8, 10.6]]:
			pew.box(side[0], py2, 0.42, side[1], py2 + 0.45, 0.48)
			pew.box(side[0], py2 + 0.4, 0.48, side[1], py2 + 0.47, 0.95)
			pew.box(side[0], py2, 0.0, side[0] + 0.06, py2 + 0.45, 0.42)
			pew.box(side[1] - 0.06, py2, 0.0, side[1], py2 + 0.45, 0.42)
		py2 += 1.3

# ------------------------------------------------------------------ bunker

func _bunker() -> void:
	var conc := geo.B(DemoMats.concrete(Color(0.62, 0.62, 0.58)), "concrete")
	var cin := geo.B(DemoMats.concrete(Color(0.5, 0.52, 0.47)), "concrete")
	# the buried box (34..40 × 6..14, floor −3, ceiling −0.8); its west
	# wall south of y = 8 belongs to the stair shaft (the hatch)
	geo.wall(1, 6, 34, 40, -3, -0.6, 0.4, [], cin, conc, 1)
	geo.wall(1, 14, 34, 40, -3, -0.6, 0.4, [], cin, conc, -1)
	geo.wall(0, 40, 6, 14, -3, -0.6, 0.4, [], cin, conc, -1)
	geo.wall(0, 34, 8, 14, -3, -0.6, 0.4, [], cin, conc, 1)
	geo.B(DemoMats.concrete(Color(0.4, 0.4, 0.37)), "concrete").box(34, 6, -3.2, 40, 14, -3.0)
	geo.slab(34, 6, 40, 14, -0.8, -0.6, [], cin, conc)
	# the blockhouse over the stair shaft: walls from the shaft floor up
	var bh := geo.B(DemoMats.concrete(Color(0.55, 0.56, 0.52)), "concrete")
	geo.wall(1, 6, 31.4, 34, -3, 2.6, 0.3, [], cin, bh, 1)
	geo.wall(1, 8, 31.4, 34, -3, 2.6, 0.3, [], cin, bh, -1)
	geo.wall(0, 31.4, 6, 8, -3, 2.6, 0.3, [[7.0, 0.7, 0.0, 2.0]], cin, bh, 1)
	geo.wall(0, 34, 6, 8, -3, 2.6, 0.3, [[7.0, 0.8, -3.0, -0.6]], cin, bh, -1)
	geo.slab(31.4, 6, 34, 8, 2.4, 2.6, [], cin, bh)
	# landing inside the blast door, the shaft floor, steps (+ ramp)
	conc.box(31.4, 6, -0.2, 31.6, 8, 0.0)
	conc.box(31.6, 6, -3.2, 34, 8, -3.0)
	var step := geo.B(DemoMats.concrete(Color(0.45, 0.45, 0.42)), "auto", false)
	for i in 12:
		var x := 31.6 + i * 0.2
		step.box(x, 6.15, -3.0, x + 0.2, 7.85, -0.25 * (i + 1))
	_ramp(Vector3(31.6, 7.0, 0.0), Vector3(34.05, 7.0, -3.0), 1.6)
	# sign
	var post := geo.B(DemoMats.metal(Color(0.3, 0.3, 0.28)), "", true)
	post.box(31.0, 4.6, 0.0, 31.08, 4.68, 1.9)
	geo.B(DemoMats.plain(Color(0.16, 0.18, 0.15), 0.7), "", false).box(30.95, 4.0, 1.75, 30.99, 5.3, 2.25)
	var sign := Label3D.new()
	sign.text = "BUNKER"
	sign.font_size = 64
	sign.pixel_size = 0.004
	sign.modulate = Color(0.85, 0.75, 0.4)
	sign.position = v(30.93, 4.65, 2.0)
	sign.rotation_degrees = Vector3(0, -90, 0)
	root.add_child(sign)
	# a caged lamp over the door, a dim tube light below
	var l := OmniLight3D.new()
	l.position = v(31.0, 7, 2.3)
	l.light_color = Color(1.0, 0.85, 0.6)
	l.light_energy = 1.2
	l.omni_range = 6.0
	root.add_child(l)
	geo.B(DemoMats.glow(Color(1.0, 0.85, 0.6), 3.0), "", false).box(31.15, 6.85, 2.2, 31.25, 7.15, 2.35)
	var tube := OmniLight3D.new()
	tube.position = v(37, 10, -1.2)
	tube.light_color = Color(0.75, 0.9, 0.8)
	tube.light_energy = 1.4
	tube.omni_range = 7.0
	root.add_child(tube)
	geo.B(DemoMats.glow(Color(0.8, 1.0, 0.9), 3.0), "", false).box(35.5, 9.95, -0.85, 38.5, 10.05, -0.8)

# ------------------------------------------------------------------ belfry

func _belfry() -> void:
	var brick := geo.B(DemoMats.brick(), "brick")
	geo.wall(1, 318, 7, 13, 0, 24.5, 0.4, [], brick, brick, 0)
	geo.wall(1, 324, 7, 13, 0, 24.5, 0.4, [], brick, brick, 0)
	geo.wall(0, 7, 318, 324, 0, 24.5, 0.4, [], brick, brick, 0)
	geo.wall(0, 13, 318, 324, 0, 24.5, 0.4, [], brick, brick, 0)
	geo.slab(7, 318, 13, 324, 24.0, 24.2, [], brick, brick)
	# the open bell stage: four corner piers, a cornice, the spire
	var c := Vector2(10, 321)
	for o in [Vector2(-2.4, -2.4), Vector2(2.4, -2.4), Vector2(-2.4, 2.4), Vector2(2.4, 2.4)]:
		brick.box(c.x + o.x - 0.6, c.y + o.y - 0.6, 24.2, c.x + o.x + 0.6, c.y + o.y + 0.6, 27.0)
	var corn := geo.B(DemoMats.sandstone(), "stone")
	corn.box(c.x - 3.4, c.y - 3.4, 27.0, c.x + 3.4, c.y + 3.4, 27.5)
	var sp := geo.B(DemoMats.slate(), "stone")
	var apex := Vector3(c.x, c.y, 35.0)
	var base := [Vector3(c.x - 3.2, c.y - 3.2, 27.5), Vector3(c.x + 3.2, c.y - 3.2, 27.5), Vector3(c.x + 3.2, c.y + 3.2, 27.5), Vector3(c.x - 3.2, c.y + 3.2, 27.5)]
	for i in 4:
		var a: Vector3 = base[i]
		var b: Vector3 = base[(i + 1) % 4]
		var mid := (a + b) * 0.5
		var n := Vector3(mid.x - c.x, mid.y - c.y, 1.0).normalized()
		sp.tri(a, b, apex, n)
	# the bells (decor: the source hangs among them)
	var bronze := DemoMats.plain(Color(0.55, 0.4, 0.2), 0.35, 0.9)
	for b in [[Vector2(-0.6, 0.0), 0.9, 25.0], [Vector2(0.9, 0.5), 0.55, 25.4]]:
		var m := MeshInstance3D.new()
		var cm := CylinderMesh.new()
		cm.top_radius = b[1] * 0.45
		cm.bottom_radius = b[1]
		cm.height = b[1] * 1.3
		m.mesh = cm
		m.material_override = bronze
		var o: Vector2 = b[0]
		m.position = v(c.x + o.x, c.y + o.y, b[2] + b[1] * 0.65)
		root.add_child(m)
	geo.B(DemoMats.planks(Color(0.4, 0.3, 0.2)), "", false, false, "bell frame: the bells hang in it").box(c.x - 2, c.y - 0.15, 26.4, c.x + 2, c.y + 0.15, 26.6)
	var lamp := OmniLight3D.new()
	lamp.position = v(c.x, c.y, 25.8)
	lamp.light_color = Color(1.0, 0.75, 0.45)
	lamp.light_energy = 4.0
	lamp.omni_range = 12.0
	root.add_child(lamp)
	# floodlit at night: a lamp on the ground before each face, aimed up
	# the brick to the bell stage
	var flood := geo.B(DemoMats.metal(Color(0.2, 0.21, 0.22)), "", false, false, "floodlight housings")
	for f in [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]:
		var at: Vector2 = c + f * 8.0
		flood.box(at.x - 0.25, at.y - 0.25, 0.0, at.x + 0.25, at.y + 0.25, 0.35)
		var s := SpotLight3D.new()
		var from := v(at.x, at.y, 0.4)
		var to := v(c.x + f.x * 3.0, c.y + f.y * 3.0, 11.0)
		s.transform = Transform3D(Basis.looking_at(to - from, Vector3.UP), from)
		s.light_color = Color(1.0, 0.86, 0.66)
		s.light_energy = 12.0
		s.spot_range = 36.0
		s.spot_angle = 34.0
		s.spot_attenuation = 0.4
		s.light_volumetric_fog_energy = 3.0 # the beams show in the night air
		root.add_child(s)

# --------------------------------------------- colonnade, kiosk, the square

func _square_props() -> void:
	# colonnade: a solid stone screen (the engine's occluder) with columns
	var st := geo.B(DemoMats.sandstone())
	st.box(16.0, 15.0, 0.0, 16.5, 21.0, 2.9)
	var cap := geo.B(DemoMats.sandstone(), "auto", false)
	cap.box(15.8, 14.8, 2.5, 16.7, 21.2, 2.9)
	var col_mesh := CylinderMesh.new()
	col_mesh.top_radius = 0.17
	col_mesh.bottom_radius = 0.2
	col_mesh.height = 2.5
	var cm := DemoMats.sandstone()
	var cols := Node3D.new()
	cols.set_meta("omg_material", "none") # the screen block is the occluder
	root.add_child(cols)
	for i in 6:
		for x in [15.85, 16.65]:
			var m := MeshInstance3D.new()
			m.mesh = col_mesh
			m.material_override = cm
			m.position = v(x, 15.5 + i, 1.25)
			cols.add_child(m)
	# kiosk: a timber booth with a lit hatch and an overhanging roof
	var wood := geo.B(DemoMats.planks(Color(0.75, 0.6, 0.45)))
	wood.box(14.5, 34.0, 0.0, 16.5, 36.0, 3.0)
	geo.B(DemoMats.planks(Color(0.45, 0.35, 0.28)), "auto", false).box(14.2, 33.7, 3.0, 16.8, 36.3, 3.15)
	geo.B(DemoMats.glow(Color(1.0, 0.82, 0.55), 1.5), "", false, false, "kiosk hatch light").box(14.46, 34.4, 1.0, 14.48, 35.6, 1.9)
	var kl := OmniLight3D.new()
	kl.position = v(14.0, 35.0, 1.7)
	kl.light_color = Color(1.0, 0.8, 0.55)
	kl.light_energy = 1.2
	kl.omni_range = 5.0
	root.add_child(kl)
	# benches on the square
	var bench := geo.B(DemoMats.planks(Color(0.6, 0.45, 0.32), "furniture_wood"))
	for p in [Vector2(10, 30), Vector2(10, 36), Vector2(18, 41)]:
		bench.box(p.x - 0.9, p.y - 0.22, 0.42, p.x + 0.9, p.y + 0.22, 0.48)
		bench.box(p.x - 0.9, p.y + 0.18, 0.48, p.x + 0.9, p.y + 0.24, 0.9)

# --------------------------------------------------------------- furniture

# walkthrough.rs furniture(), world coordinates, with acoustic kinds.
const FURNITURE := [
	# living room
	[2.0, 4.4, 0.0, 4.2, 5.3, 0.75, "soft", "sofa"],
	[2.5, 3.3, 0.0, 3.7, 3.9, 0.45, "furniture_wood", "table"],
	[4.9, 4.3, 0.0, 5.7, 5.1, 0.75, "soft", "chair"],
	[0.3, 0.2, 0.0, 2.3, 0.7, 0.9, "cabinet", "cabinet"],
	[5.4, 0.4, 0.0, 5.8, 3.4, 2.2, "books", "books"],
	# great hall: tables, chairs, lectern, whiteboard
	[1.8, 15.6, 0.0, 3.6, 16.3, 0.74, "furniture_wood", "table"],
	[1.8, 18.4, 0.0, 3.6, 19.1, 0.74, "furniture_wood", "table"],
	[1.8, 21.2, 0.0, 3.6, 21.9, 0.74, "furniture_wood", "table"],
	[4.6, 15.6, 0.0, 6.4, 16.3, 0.74, "furniture_wood", "table"],
	[4.6, 18.4, 0.0, 6.4, 19.1, 0.74, "furniture_wood", "table"],
	[4.6, 21.2, 0.0, 6.4, 21.9, 0.74, "furniture_wood", "table"],
	[1.2, 15.7, 0.0, 1.65, 16.15, 0.45, "furniture_wood", "chair_wood"],
	[1.2, 18.5, 0.0, 1.65, 18.95, 0.45, "furniture_wood", "chair_wood"],
	[1.2, 21.3, 0.0, 1.65, 21.75, 0.45, "furniture_wood", "chair_wood"],
	[4.0, 15.7, 0.0, 4.45, 16.15, 0.45, "furniture_wood", "chair_wood"],
	[4.0, 18.5, 0.0, 4.45, 18.95, 0.45, "furniture_wood", "chair_wood"],
	[4.0, 21.3, 0.0, 4.45, 21.75, 0.45, "furniture_wood", "chair_wood"],
	[10.2, 19.9, 0.0, 10.8, 20.3, 1.2, "cabinet", "lectern"],
	[12.0, 16.5, 0.0, 12.3, 19.5, 2.2, "panel", "whiteboard"],
	# club bar
	[23.0, 29.0, 0.0, 24.0, 33.5, 1.1, "cabinet", "bar"],
	# old house ground floor
	[25.0, 18.0, 0.0, 27.2, 18.9, 0.75, "soft", "sofa"],
	[25.6, 19.4, 0.0, 26.6, 20.0, 0.4, "furniture_wood", "table"],
	[24.3, 20.4, 0.4, 24.55, 21.9, 1.4, "panel", "tv"],
	[28.4, 16.3, 0.0, 30.7, 16.95, 0.95, "cabinet", "kitchen"],
	[30.0, 17.3, 0.0, 30.7, 18.0, 1.9, "cabinet", "fridge"],
	[26.2, 20.8, 0.0, 26.8, 21.4, 0.25, "soft", "pillow"],
	[27.4, 21.2, 0.0, 28.0, 21.8, 0.25, "soft", "pillow"],
	# old house upper floor
	# (the walkthrough puts the bed at x 24.8–26.6, over the stairwell
	# this level actually cuts into the slab: moved east to clear it)
	[26.0, 16.8, 3.0, 27.8, 18.8, 3.6, "soft", "bed"],
	[24.25, 19.6, 3.8, 24.45, 21.6, 5.0, "panel", "tv"],
	[29.8, 16.4, 3.0, 30.6, 18.0, 4.0, "cabinet", "cabinet"],
	[28.6, 20.6, 3.0, 29.2, 21.2, 3.25, "soft", "pillow"],
]

const LAYOUT := "demo/layout.json"

# Furniture: from the shared demo layout (assets/demo/layout.json, the
# same data the web loads) when present, else the FURNITURE table. Each
# piece: a low-poly mesh per kind, collision, and its acoustic material
# on the piece (explicit `omg_material` meta: the engine traces the
# piece's real faces).
func _furniture() -> void:
	var items := _layout_cached()
	if items.is_empty():
		for f in FURNITURE:
			items.append({"kind": f[7], "acoustic": f[6], "min": Vector3(f[0], f[1], f[2]), "max": Vector3(f[3], f[4], f[5]), "yaw": 0.0})
	for it in items:
		# the cathedral's round columns are built (and traced) by
		# _cathedral_interior at the same spots as the layout's pillars
		if it["kind"] == "pillar":
			continue
		_piece(it)

var _layout_cache = null

# The paved outdoor patches of layout.json: [{name, surface, min, max}]
# (min/max as Vector2); the town's own set when the file is missing.
func _layout_ground() -> Array:
	var path := DemoAssets.file(LAYOUT)
	var data = JSON.parse_string(FileAccess.get_file_as_string(path)) if FileAccess.file_exists(path) else null
	var raw: Array = data.get("ground", []) if data is Dictionary else []
	if raw.is_empty():
		for r in [["square", 5.5, 24.2, 21.5, 44.0], ["lane", 21.5, 23.2, 34.5, 26.0], ["lane", 14.0, 23.2, 21.5, 24.2],
				["path", 6.0, 44.0, 10.0, 51.8], ["path", -15.0, 26.0, 5.5, 29.0], ["path", 24.0, 6.0, 31.4, 8.0],
				["path", 34.5, 14.0, 37.0, 23.2], ["belfry path", 8.6, 74.4, 11.4, 316.0]]:
			raw.append({"name": r[0], "surface": "paving", "min": [r[1], r[2]], "max": [r[3], r[4]]})
	var out := []
	for g in raw:
		out.append({"name": str(g["name"]), "surface": str(g["surface"]),
			"min": Vector2(g["min"][0], g["min"][1]), "max": Vector2(g["max"][0], g["max"][1])})
	return out

# The paved patch under (x, y) — "paving", "asphalt", "concrete" — or ""
# (the slabs are thin and not collidable: the floor ray finds the grass).
func ground_at(x: float, y: float) -> String:
	if _ground_cache == null:
		_ground_cache = _layout_ground()
	for pv in _ground_cache:
		var mn: Vector2 = pv["min"]
		var mx: Vector2 = pv["max"]
		if x >= mn.x and x <= mx.x and y >= mn.y and y <= mx.y:
			return pv["surface"]
	return ""

var _ground_cache = null

# A rug from the layout under (x, y) at floor height z (plan coordinates):
# the footsteps' carpet (rugs are decor, the floor ray passes them).
func rug_at(x: float, y: float, z: float) -> bool:
	for it in _layout_cached():
		if it["kind"] != "rug":
			continue
		var mn: Vector3 = it["min"]
		var mx: Vector3 = it["max"]
		# into the rug's own frame (min/max are its unturned extents)
		var c := Vector2((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5)
		var h := Vector2(mx.x - mn.x, mx.y - mn.y) * 0.5
		var d := Vector2(x, y) - c
		var yaw: float = it["yaw"]
		var local := Vector2(d.x * cos(yaw) + d.y * sin(yaw), -d.x * sin(yaw) + d.y * cos(yaw))
		if absf(local.x) <= h.x and absf(local.y) <= h.y and absf(mn.z - z) < 0.5:
			return true
	return false

func _layout_cached() -> Array:
	if _layout_cache == null:
		_layout_cache = _layout_items()
	return _layout_cache

# The first layout piece of `kind` ({} when none / no layout file).
func _layout_item(kind: String) -> Dictionary:
	for it in _layout_cached():
		if it["kind"] == kind:
			return it
	return {}

# Tolerant reader: {"rooms": [{"furniture": [...]}, ...]} or
# {"furniture": [...]} or a bare list; items carry "kind", "material" /
# "acoustic", and either "min"+"max" / "aabb" [x0,y0,z0,x1,y1,z1], or
# "position" (centre of the footprint, z = floor) + "size" [x,y,z] +
# "yaw" (degrees, z-up world). layout.json gives "pos" + "size" (own
# frame) + "yaw"; its "min"/"max" are the TURNED world box, used only
# when no own-frame size is given.
func _layout_items() -> Array:
	var path := DemoAssets.file(LAYOUT)
	if not FileAccess.file_exists(path):
		return []
	var data = JSON.parse_string(FileAccess.get_file_as_string(path))
	var raw := []
	if data is Array:
		raw = data
	elif data is Dictionary:
		if data.has("furniture"):
			raw.append_array(data["furniture"])
		for r in data.get("rooms", []):
			if r is Dictionary:
				raw.append_array(r.get("furniture", []))
	var out := []
	for f in raw:
		if not f is Dictionary:
			continue
		# "material" names the acoustic preset; "acoustic": false marks a
		# visual-only piece (rugs, floor lamps). yaw is in degrees in the
		# file (CCW from +y, z-up world) — the same turn as rotation.y.
		var mat := str(f.get("material", "furniture_wood"))
		if f.get("acoustic", true) is bool and not f.get("acoustic", true):
			mat = "none"
		var it := {"kind": str(f.get("kind", "box")), "acoustic": mat, "yaw": deg_to_rad(float(f.get("yaw", 0.0)))}
		if f.has("pos") and f.has("size"):
			# the piece's OWN frame: centre of the footprint, z = bottom,
			# size = width × depth × height before the yaw turn
			var c := _vec(f["pos"])
			var sz := _vec(f["size"])
			it["min"] = Vector3(c.x - sz.x * 0.5, c.y - sz.y * 0.5, c.z)
			it["max"] = Vector3(c.x + sz.x * 0.5, c.y + sz.y * 0.5, c.z + sz.z)
		elif f.has("min") and f.has("max"):
			it["min"] = _vec(f["min"])
			it["max"] = _vec(f["max"])
		elif f.has("aabb"):
			var b: Array = f["aabb"]
			it["min"] = Vector3(b[0], b[1], b[2])
			it["max"] = Vector3(b[3], b[4], b[5])
		elif f.has("position") and f.has("size"):
			var c := _vec(f["position"])
			var sz := _vec(f["size"])
			it["min"] = Vector3(c.x - sz.x * 0.5, c.y - sz.y * 0.5, c.z)
			it["max"] = Vector3(c.x + sz.x * 0.5, c.y + sz.y * 0.5, c.z + sz.z)
		else:
			continue
		out.append(it)
	if not out.is_empty():
		print("demo: %d furniture pieces from %s" % [out.size(), DemoAssets.root + LAYOUT])
	return out

static func _vec(a) -> Vector3:
	if a is Array and a.size() >= 3:
		return Vector3(a[0], a[1], a[2])
	return Vector3.ZERO

var _body: StaticBody3D

# a box part in the piece's local plan frame (x width, y depth, z up)
func _part(parent: Node3D, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, m: Material, solid := true) -> void:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(x1 - x0, z1 - z0, y1 - y0)
	mi.mesh = bm
	mi.material_override = m
	mi.position = Vector3((x0 + x1) * 0.5, (z0 + z1) * 0.5, -(y0 + y1) * 0.5)
	parent.add_child(mi)
	if solid:
		var cs := CollisionShape3D.new()
		var sh := BoxShape3D.new()
		sh.size = bm.size
		cs.shape = sh
		cs.position = mi.position
		_body.add_child(cs)

# One piece: a root at the footprint's centre (turned by yaw), a
# low-poly mesh per kind; the acoustic material rides on the root.
func _piece(it: Dictionary) -> void:
	var mn: Vector3 = it["min"]
	var mx: Vector3 = it["max"]
	var size := mx - mn
	var kind: String = it["kind"]
	var node := Node3D.new()
	node.name = kind.capitalize().replace(" ", "")
	node.set_meta("omg_material", it["acoustic"])
	node.position = v((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, mn.z)
	node.rotation.y = float(it["yaw"])
	root.add_child(node)
	var decor := Node3D.new()
	decor.set_meta("omg_material", "none") # trims, screens, legs
	node.add_child(decor)
	if labels:
		var tag := Label3D.new()
		tag.text = "%s (%s)" % [kind, it["acoustic"]]
		tag.font_size = 40
		tag.pixel_size = 0.0007
		tag.fixed_size = true
		tag.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		tag.no_depth_test = true
		tag.modulate = Color(0.7, 1.0, 0.8)
		tag.position = Vector3(0, size.z + 0.25, 0)
		node.add_child(tag)
	_body = StaticBody3D.new()
	node.add_child(_body)
	var hx := size.x * 0.5
	var hy := size.y * 0.5
	var h := size.z
	var wood := DemoMats.planks(Color(0.75, 0.55, 0.4))
	var dark := DemoMats.planks(Color(0.45, 0.32, 0.24))
	# own frame: +y is the piece's FRONT (seat side, screen, book spines)
	match kind:
		"sofa", "chair", "armchair":
			var fab := DemoMats.fabric(Color(0.42, 0.18, 0.16))
			_part(node, -hx, -hy, 0.0, hx, hy, 0.42, fab)
			_part(node, -hx, -hy, 0.42, hx, -hy + 0.22, h, fab)
			_part(node, -hx, -hy, 0.42, -hx + 0.18, hy, 0.62, fab)
			_part(node, hx - 0.18, -hy, 0.42, hx, hy, 0.62, fab)
		"bed":
			_part(node, -hx, -hy, 0.0, hx, hy, h * 0.6, dark)
			_part(node, -hx + 0.03, -hy + 0.03, h * 0.6, hx - 0.03, hy - 0.03, h, DemoMats.fabric(Color(0.55, 0.6, 0.7)))
			_part(decor, -hx + 0.2, hy - 0.5, h, hx - 0.2, hy - 0.1, h + 0.12, DemoMats.fabric(Color(0.92, 0.92, 0.9)), false)
		"table", "desk", "coffee_table":
			_part(node, -hx, -hy, h - 0.05, hx, hy, h, wood)
			for c in [[-hx, -hy], [hx - 0.06, -hy], [-hx, hy - 0.06], [hx - 0.06, hy - 0.06]]:
				_part(decor, c[0], c[1], 0.0, c[0] + 0.06, c[1] + 0.06, h - 0.05, wood, false)
		"chair_wood", "stool":
			_part(node, -hx, -hy, h - 0.04, hx, hy, h, wood)
			_part(decor, -hx, -hy, 0.0, -hx + 0.04, hy, h - 0.04, wood, false)
			_part(decor, hx - 0.04, -hy, 0.0, hx, hy, h - 0.04, wood, false)
			_part(decor, -hx, hy - 0.04, h, hx, hy, h + 0.45, wood, false)
		"books", "bookshelf":
			_part(node, -hx, -hy, 0.0, hx, hy, h, dark)
			var spines := [Color(0.5, 0.15, 0.12), Color(0.15, 0.3, 0.5), Color(0.6, 0.5, 0.2), Color(0.2, 0.4, 0.25), Color(0.4, 0.2, 0.4)]
			var rng := RandomNumberGenerator.new()
			rng.seed = 11
			# spines along the long side, on the front (+y) face
			var long_x := size.x >= size.y
			for shelf in int(h / 0.42):
				var z := 0.1 + shelf * 0.42
				var a := (-hx if long_x else -hy) + 0.05
				var lim := (hx if long_x else hy) - 0.08
				while a < lim:
					var w := rng.randf_range(0.03, 0.07)
					var bh := rng.randf_range(0.22, 0.32)
					var m := DemoMats.plain(spines[rng.randi() % spines.size()], 0.8)
					if long_x:
						_part(decor, a, hy - 0.01, z, a + w, hy + 0.02, z + bh, m, false)
					else:
						_part(decor, hx - 0.01, a, z, hx + 0.02, a + w, z + bh, m, false)
					a += w + 0.004
		"tv":
			_part(node, -hx, -hy, 0.0, hx, hy, h, DemoMats.plain(Color(0.03, 0.03, 0.04), 0.3))
			var glow := DemoMats.glow(Color(0.25, 0.4, 0.7), 0.6)
			_part(decor, -hx + 0.05, hy + 0.001, 0.05, hx - 0.05, hy + 0.007, h - 0.05, glow, false)
		"whiteboard":
			var paper := DemoMats.plain(Color(0.95, 0.95, 0.93), 0.25)
			if size.y > 0.2:
				# a flip chart on an easel: board at the front, a back leg
				var steel := DemoMats.metal(Color(0.3, 0.3, 0.32))
				var b0 := h - 1.0
				_part(node, -hx, hy - 0.06, b0, hx, hy - 0.03, h, DemoMats.plain(Color(0.55, 0.55, 0.58), 0.6))
				_part(decor, -hx + 0.03, hy - 0.03, b0 + 0.03, hx - 0.03, hy - 0.005, h - 0.06, paper, false)
				_part(decor, -hx, hy - 0.07, h - 0.06, hx, hy, h, steel, false)
				_part(decor, -hx, hy - 0.1, b0 - 0.02, hx, hy - 0.03, b0, steel, false)
				for lx in [-hx, hx - 0.04]:
					_part(node, lx, hy - 0.08, 0.0, lx + 0.04, hy - 0.04, b0, steel)
				_part(node, -0.02, -hy, 0.0, 0.02, -hy + 0.04, b0 + 0.05, steel)
				_part(decor, -0.02, -hy, b0 + 0.02, 0.02, hy - 0.06, b0 + 0.05, steel, false)
			else:
				_part(node, -hx, -hy, 0.0, hx, hy, h, paper)
		"fridge":
			_part(node, -hx, -hy, 0.0, hx, hy, h, DemoMats.plain(Color(0.85, 0.87, 0.88), 0.3, 0.3))
		"pillow":
			_part(node, -hx, -hy, 0.0, hx, hy, h, DemoMats.fabric(Color(0.6, 0.4, 0.5)))
		"lectern":
			_part(node, -hx, -hy, 0.0, hx, hy, h, dark)
			_part(decor, -hx - 0.05, -hy - 0.05, h, hx + 0.05, hy + 0.05, h + 0.05, dark, false)
		"rug":
			# the room floors lie 2 cm up: float the rug just above them
			_part(decor, -hx, -hy, 0.025, hx, hy, 0.035, DemoMats.carpet(Color(0.42, 0.2, 0.17)), false)
			_part(decor, -hx + 0.12, -hy + 0.12, 0.035, hx - 0.12, hy - 0.12, 0.038, DemoMats.carpet(Color(0.55, 0.3, 0.24)), false)
		"table_lamp":
			var brass := DemoMats.metal(Color(0.7, 0.58, 0.32))
			_part(node, -hx * 0.6, -hy * 0.6, 0.0, hx * 0.6, hy * 0.6, 0.03, brass)
			_part(decor, -0.012, -0.012, 0.03, 0.012, 0.012, h * 0.55, brass, false)
			_part(decor, -hx, -hy, h * 0.5, hx, hy, h, DemoMats.glow(Color(1.0, 0.86, 0.6), 1.5), false)
		"candle_stand":
			var brass := DemoMats.metal(Color(0.62, 0.5, 0.26))
			_part(node, -hx, -hy, 0.0, hx, hy, 0.05, brass)
			_part(node, -0.025, -0.025, 0.05, 0.025, 0.025, h - 0.3, brass)
			_part(decor, -hx * 0.6, -hy * 0.6, h - 0.3, hx * 0.6, hy * 0.6, h - 0.27, brass, false)
			_part(decor, -0.035, -0.035, h - 0.27, 0.035, 0.035, h - 0.05, DemoMats.plain(Color(0.95, 0.92, 0.85), 0.6), false)
			_part(decor, -0.015, -0.015, h - 0.05, 0.015, 0.015, h, DemoMats.glow(Color(1.0, 0.7, 0.3), 6.0), false)
		"font":
			var st := DemoMats.sandstone(1.0, Color(0.72, 0.7, 0.68))
			_part(node, -hx, -hy, 0.0, hx, hy, 0.1, st)
			_part(node, -hx * 0.45, -hy * 0.45, 0.1, hx * 0.45, hy * 0.45, h - 0.35, st)
			_part(node, -hx, -hy, h - 0.35, hx, hy, h - 0.3, st)
			_part(node, -hx, -hy, h - 0.3, hx, -hy + 0.06, h, st)
			_part(node, -hx, hy - 0.06, h - 0.3, hx, hy, h, st)
			_part(node, -hx, -hy + 0.06, h - 0.3, -hx + 0.06, hy - 0.06, h, st)
			_part(node, hx - 0.06, -hy + 0.06, h - 0.3, hx, hy - 0.06, h, st)
			_part(decor, -hx + 0.06, -hy + 0.06, h - 0.3, hx - 0.06, hy - 0.06, h - 0.08, DemoMats.plain(Color(0.3, 0.42, 0.55), 0.1), false)
		"barrel":
			var drum := DemoMats.plain(Color(0.24, 0.35, 0.23), 0.5, 0.4)
			# an octagonal drum from two crossed boxes, with hoops
			var r := hx - 0.012
			_part(node, -r, -r * 0.6, 0.0, r, r * 0.6, h, drum)
			_part(node, -r * 0.6, -r, 0.0, r * 0.6, r, h, drum)
			var hoop := DemoMats.plain(Color(0.16, 0.22, 0.15), 0.5, 0.4)
			for hz in [0.12, h * 0.5, h - 0.16]:
				_part(decor, -hx, -r * 0.6, hz, hx, r * 0.6, hz + 0.03, hoop, false)
				_part(decor, -r * 0.6, -hx, hz, r * 0.6, hx, hz + 0.03, hoop, false)
		_: # cabinet, sideboard, kitchen, bar, dresser, crate, …
			_part(node, -hx, -hy, 0.0, hx, hy, h, dark)

func _props() -> void:
	# a piano-black hi-fi for the music (the rugs come from the layout)
	var hifi := geo.B(DemoMats.plain(Color(0.04, 0.04, 0.05), 0.2, 0.1), "", true, false, "hi-fi: the music source sits in it")
	hifi.box(1.75, 2.75, 0.0, 2.25, 3.25, 1.05)
	geo.B(DemoMats.plain(Color(0.15, 0.15, 0.16), 0.5), "", false).box(1.9, 3.26, 0.75, 2.1, 3.27, 0.95)
	# radio on a crate in the bunker
	geo.B(DemoMats.planks(Color(0.6, 0.5, 0.35))).box(37.1, 12.1, -3.0, 37.9, 12.9, -2.2)
	geo.B(DemoMats.plain(Color(0.45, 0.3, 0.18), 0.5), "", false, false, "radio set: the radio source sits in it").box(37.25, 12.3, -2.2, 37.75, 12.65, -1.88)
	geo.B(DemoMats.glow(Color(1.0, 0.7, 0.3), 2.0), "", false).box(37.3, 12.29, -2.1, 37.5, 12.3, -2.0)
	# figures: the narrator behind her lectern, the flautist in the nave
	_figure(Vector2(10.5, 20.6), 0.0, Color(0.25, 0.35, 0.3))
	_figure(Vector2(8.0, 66.4), 0.0, Color(0.3, 0.28, 0.4))
	# lamps inside: warm pools in every room
	for l in [[Vector3(4, 3, 2.4), Color(1.0, 0.78, 0.5), 2.2, 7.0, true], [Vector3(4, 10, 2.2), Color(1.0, 0.85, 0.65), 1.0, 5.0, false],
			[Vector3(21, 31, 2.3), Color(1.0, 0.5, 0.7), 1.2, 5.0, false], [Vector3(27, 19.5, 2.5), Color(1.0, 0.75, 0.5), 2.0, 7.0, true],
			[Vector3(27, 19.5, 5.2), Color(1.0, 0.8, 0.6), 1.5, 6.0, false]]:
		var o := OmniLight3D.new()
		var p: Vector3 = l[0]
		o.position = v(p.x, p.y, p.z)
		o.light_color = l[1]
		o.light_energy = l[2]
		o.omni_range = l[3]
		o.shadow_enabled = l[4]
		root.add_child(o)
		geo.B(DemoMats.glow(l[1], 3.0), "", false, false, "lamp shades (light)").box(p.x - 0.15, p.y - 0.15, p.z + 0.1, p.x + 0.15, p.y + 0.15, p.z + 0.18)

# The Great Hall (the seminar room): wall sconces throwing cones of
# light up the walls onto the ceiling, a small lamp on every table (from
# the layout) and the presenter's desk, and a faint fill from above.
func _hall_lights() -> void:
	var warm := Color(1.0, 0.82, 0.6)
	var brass := geo.B(DemoMats.metal(Color(0.62, 0.5, 0.3)), "", false, false, "sconce brackets / lamp bases")
	var bowl := geo.B(DemoMats.glow(warm, 2.2), "", false, false, "sconce bowls / lamp shades (light)")
	# [x, y, inward normal]: two walls of four, the west wall two; clear of
	# the doors (4, 14), (7, 24) and the whiteboards on the east wall
	var sconces := []
	for x in [1.8, 5.2, 9.2, 12.4]:
		sconces.append([Vector2(x, 14.0), Vector2(0, 1)])
		sconces.append([Vector2(x, 24.0), Vector2(0, -1)])
	for y in [17.0, 21.0]:
		sconces.append([Vector2(0.0, y), Vector2(1, 0)])
	for sc in sconces:
		var w: Vector2 = sc[0]
		var n: Vector2 = sc[1]
		var at: Vector2 = w + n * 0.24 # the wall's inner face + the bowl
		var t := Vector2(-n.y, n.x) * 0.16
		brass.box(minf(at.x - t.x, w.x + n.x * 0.1), minf(at.y - t.y, w.y + n.y * 0.1), 2.45,
			maxf(at.x + t.x, w.x + n.x * 0.1), maxf(at.y + t.y, w.y + n.y * 0.1), 2.5)
		bowl.box(at.x - absf(t.x) - 0.04, at.y - absf(t.y) - 0.04, 2.5, at.x + absf(t.x) + 0.04, at.y + absf(t.y) + 0.04, 2.62)
		var up := SpotLight3D.new()
		up.position = v(at.x, at.y, 2.62)
		up.rotation_degrees = Vector3(90, 0, 0) # the cone points up
		up.light_color = warm
		up.light_energy = 4.0
		up.spot_range = 6.0
		up.spot_angle = 38.0
		up.spot_attenuation = 0.6
		up.light_volumetric_fog_energy = 2.5
		root.add_child(up)
		var glow := OmniLight3D.new() # the bowl's own soft spill
		glow.position = v(at.x, at.y, 2.4)
		glow.light_color = warm
		glow.light_energy = 0.5
		glow.omni_range = 2.5
		glow.set_meta("no_shadow", true)
		glow.set_meta("no_halo", true) # the sconce's uplight has its dot
		root.add_child(glow)
	# table lamps: every table and desk standing in the hall
	for it in _layout_cached():
		var mn: Vector3 = it["min"]
		var mx: Vector3 = it["max"]
		var c := Vector2((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5)
		if not (it["kind"] in ["table", "desk"]) or c.x < 0.0 or c.x > 14.0 or c.y < 14.0 or c.y > 24.0:
			continue
		var top := mx.z
		brass.box(c.x - 0.06, c.y - 0.06, top, c.x + 0.06, c.y + 0.06, top + 0.02)
		brass.box(c.x - 0.012, c.y - 0.012, top + 0.02, c.x + 0.012, c.y + 0.012, top + 0.3)
		bowl.box(c.x - 0.1, c.y - 0.1, top + 0.26, c.x + 0.1, c.y + 0.1, top + 0.4)
		var tl := OmniLight3D.new()
		tl.position = v(c.x, c.y, top + 0.33)
		tl.light_color = warm
		tl.light_energy = 0.7
		tl.omni_range = 2.6
		tl.set_meta("no_shadow", true)
		root.add_child(tl)
	# a faint fill from the ceiling (no fixture: the sconces carry the look)
	var fill := OmniLight3D.new()
	fill.position = v(7.0, 19.0, 6.0)
	fill.light_color = Color(1.0, 0.92, 0.8)
	fill.light_energy = 1.2
	fill.omni_range = 11.0
	fill.set_meta("no_shadow", true)
	fill.set_meta("no_halo", true)
	root.add_child(fill)

func _figure(at: Vector2, _yaw: float, c: Color) -> void:
	var body := geo.B(DemoMats.fabric(c))
	body.box(at.x - 0.2, at.y - 0.13, 0.0, at.x + 0.2, at.y + 0.13, 1.45)
	var head := MeshInstance3D.new()
	var s := SphereMesh.new()
	s.radius = 0.12
	s.height = 0.26
	head.mesh = s
	head.material_override = DemoMats.plain(Color(0.75, 0.6, 0.5), 0.6)
	head.position = v(at.x, at.y, 1.6)
	root.add_child(head)

# -------------------------------------------------------- windows & doors

func _windows() -> void:
	var frame := geo.B(DemoMats.planks(Color(0.9, 0.88, 0.84)), "", false, false, "window frames: the pane is an acoustic OmgDoor")
	var pane := geo.B(DemoMats.glass(), "", true, false, "glass panes: acoustic as OmgDoor (glass)")
	for w in WINDOWS:
		var p: Vector2 = w[0]
		var hw: float = w[2]
		var z0: float = w[3] - WIN_H * 0.5
		var z1: float = w[3] + WIN_H * 0.5
		var omg := OmgDoor.new()
		omg.material = "heavy_glass" if w[4] else "glass"
		if w[1] == 0:
			pane.box(p.x - 0.02, p.y - hw, z0, p.x + 0.02, p.y + hw, z1)
			frame.box(p.x - 0.06, p.y - hw, z0, p.x + 0.06, p.y + hw, z0 + 0.06)
			frame.box(p.x - 0.06, p.y - hw, z1 - 0.06, p.x + 0.06, p.y + hw, z1)
			frame.box(p.x - 0.06, p.y - 0.03, z0, p.x + 0.06, p.y + 0.03, z1)
			omg.rotation_degrees = Vector3(0, 90, 0)
		else:
			pane.box(p.x - hw, p.y - 0.02, z0, p.x + hw, p.y + 0.02, z1)
			frame.box(p.x - hw, p.y - 0.06, z0, p.x + hw, p.y + 0.06, z0 + 0.06)
			frame.box(p.x - hw, p.y - 0.06, z1 - 0.06, p.x + hw, p.y + 0.06, z1)
			frame.box(p.x - 0.03, p.y - 0.06, z0, p.x + 0.03, p.y + 0.06, z1)
		omg.size = Vector3(2 * hw, WIN_H, 0.08)
		omg.openness = 0.0
		omg.position = v(p.x, p.y, w[3])
		root.add_child(omg)

func _doors() -> void:
	var wood_m := DemoMats.planks(Color(0.62, 0.42, 0.28))
	var steel_m := DemoMats.metal(Color(0.42, 0.45, 0.48))
	for i in DOORS.size():
		var d: Array = DOORS[i]
		var p: Vector2 = d[0]
		var hw: float = d[2]
		var h: float = d[3]
		var heavy: bool = d[4]
		# hinge at one jamb; the leaf swings ~112°. The leaf is only the
		# look: its sound is the OmgDoor below (a swinging mesh would be
		# frozen into the static geometry)
		var hinge := Node3D.new()
		hinge.set_meta("omg_material", "none")
		var leaf := MeshInstance3D.new()
		var bm := BoxMesh.new()
		var thick := 0.12 if heavy else 0.06
		bm.size = Vector3(2 * hw, h, thick)
		leaf.mesh = bm
		leaf.material_override = steel_m if (heavy and i == 7) else wood_m
		if d[1] == 0:
			hinge.position = v(p.x, p.y - hw, h * 0.5)
			hinge.rotation_degrees = Vector3(0, 90, 0)
		else:
			hinge.position = v(p.x - hw, p.y, h * 0.5)
		leaf.position = Vector3(hw, 0, 0)
		hinge.add_child(leaf)
		var body := AnimatableBody3D.new()
		var cs := CollisionShape3D.new()
		var sh := BoxShape3D.new()
		sh.size = bm.size
		cs.shape = sh
		body.add_child(cs)
		leaf.add_child(body)
		# a knob
		var knob := MeshInstance3D.new()
		var km := SphereMesh.new()
		km.radius = 0.035
		km.height = 0.07
		knob.mesh = km
		knob.material_override = DemoMats.metal(Color(0.75, 0.62, 0.3))
		knob.position = Vector3(hw * 0.8, -0.05 * h, thick * 0.7)
		leaf.add_child(knob)
		root.add_child(hinge)
		# the acoustic leaf: covered fraction priced continuously
		var omg := OmgDoor.new()
		omg.material = "heavy_door" if heavy else "door"
		omg.size = Vector3(2 * hw, h, 0.08)
		omg.position = v(p.x, p.y, h * 0.5)
		if d[1] == 0:
			omg.rotation_degrees = Vector3(0, 90, 0)
		var open: float = 1.0 if DOOR_OPEN[i] else 0.0
		omg.openness = open
		root.add_child(omg)
		door_nodes.append({"hinge": hinge, "shape": cs, "omg": omg, "pos": p, "hw": hw, "open": open, "target": open, "axis": d[1]})
		_pose_door(door_nodes[-1])

func update_doors(dt: float) -> void:
	for dn in door_nodes:
		dn["open"] += (dn["target"] - dn["open"]) * minf(1.0, dt * 7.0)
		# there: stop (easing in for ever, the leaf would keep moving the
		# acoustic world by fractions of a millimetre)
		if absf(dn["target"] - dn["open"]) < 0.001:
			dn["open"] = dn["target"]
		_pose_door(dn)

func _pose_door(dn: Dictionary) -> void:
	var o: float = dn["open"]
	var hinge: Node3D = dn["hinge"]
	var base := 90.0 if dn["axis"] == 0 else 0.0
	hinge.rotation_degrees = Vector3(0, base + rad_to_deg(o * 1.95), 0)
	(dn["omg"] as OmgDoor).openness = o
	# passable once it has swung well out of the frame
	(dn["shape"] as CollisionShape3D).disabled = o > 0.45

# ---------------------------------------------------------------- curtains

# Drapes that a pull cord (E) lowers or rolls up: in the seminar room
# along its south, west and north walls (the doors and the front wall
# left free), in the cathedral between the lancets of both flanks. Each
# hangs 10 cm in front of its wall. To the engine a curtain is an OmgDoor
# turned on its side (its covered share grows from the top down) of
# "drape": cloth that reflects nothing — sound passes through it, off the
# wall behind and back, losing most of its mids and highs on the way.
const CURTAIN_GAP := 0.10   # off the wall's inner face
const CURTAIN_SPEED := 0.3  # of its drop per second (a 10 m drape: ~3 s)

func _curtains() -> void:
	# seminar room (walls 0.3 m: inner faces at x 0.15, y 14.15 / 23.85)
	var hall := []
	var hz := [0.25, 6.6]   # bottom, top (the roll)
	for seg in [[1, 14.15 + CURTAIN_GAP, 0.4, 3.3], [1, 14.15 + CURTAIN_GAP, 4.7, 13.6],
			[1, 23.85 - CURTAIN_GAP, 0.4, 6.3], [1, 23.85 - CURTAIN_GAP, 7.7, 13.6],
			[0, 0.15 + CURTAIN_GAP, 14.4, 23.6]]:
		hall.append(seg + hz)
	_curtain_set(Vector2(2.6, 14.45), hall)
	# cathedral (walls 0.5 m: inner faces at x 0.25 / 15.75): the wall
	# between the lancets (1.1 m wide, centred every 3.2 m from y 56.6)
	var nave := []
	var cz := [0.3, 11.0]
	var edges := [52.4]
	var y := 56.6
	while y < 70.5:
		edges.append(y - 0.65)
		edges.append(y + 0.65)
		y += 3.2
	edges.append(73.6)
	for x in [0.25 + CURTAIN_GAP, 15.75 - CURTAIN_GAP]:
		for k in range(0, edges.size(), 2):
			nave.append([0, x, edges[k], edges[k + 1]] + cz)
	_curtain_set(Vector2(5.6, 52.5), nave)

# One pull cord and its drapes: [axis (0: a wall x = c), c, from, to,
# bottom z, top z] each.
func _curtain_set(cord: Vector2, spans: Array) -> void:
	var holder := Node3D.new()
	holder.set_meta("omg_material", "none") # the look only: the sound is the OmgDoors
	root.add_child(holder)
	var cloth_m := StandardMaterial3D.new()
	cloth_m.albedo_color = Color(0.42, 0.07, 0.09)
	cloth_m.roughness = 1.0
	cloth_m.cull_mode = BaseMaterial3D.CULL_DISABLED # pleats seen from both sides
	var roll_m := StandardMaterial3D.new()
	roll_m.albedo_color = Color(0.36, 0.06, 0.08)
	roll_m.roughness = 1.0
	var curtains := []
	for sp in spans:
		var axis: int = sp[0]
		var c: float = sp[1]
		var a0: float = sp[2]
		var a1: float = sp[3]
		var z0: float = sp[4]
		var z1: float = sp[5]
		var w := a1 - a0
		var h := z1 - z0
		var mid := 0.5 * (a0 + a1)
		var at := Vector2(c, mid) if axis == 0 else Vector2(mid, c)
		# the sound: a box of drape, covering from the top down
		var omg := OmgDoor.new()
		omg.material = "drape"
		omg.size = Vector3(h, w, 0.02)
		var along := Vector3(0, 0, -1) if axis == 0 else Vector3(1, 0, 0) # plan y / plan x
		var down := Vector3(0, -1, 0)
		omg.transform = Transform3D(Basis(down, along, down.cross(along)), v(at.x, at.y, 0.5 * (z0 + z1)))
		omg.openness = 1.0
		root.add_child(omg)
		# the look: pleated cloth (scaled from the top) and its roll
		var cloth := MeshInstance3D.new()
		cloth.mesh = _pleats(w, h, randf() * TAU)
		cloth.material_override = cloth_m
		# the mesh's width runs along local x, its folds along local z:
		# turned for a wall x = c (width along plan y)
		if axis == 0:
			cloth.rotation_degrees = Vector3(0, 90, 0)
		holder.add_child(cloth)
		var roll := MeshInstance3D.new()
		var cm := CylinderMesh.new()
		cm.top_radius = 0.09
		cm.bottom_radius = 0.09
		cm.height = w
		roll.mesh = cm
		roll.material_override = roll_m
		roll.position = v(at.x, at.y, z1)
		roll.rotation_degrees = Vector3(90, 0, 0) if axis == 0 else Vector3(0, 0, 90)
		holder.add_child(roll)
		curtains.append({"omg": omg, "cloth": cloth, "roll": roll, "at": at, "top": z1, "h": h})
	# the cord: gold, a red tassel at hand height
	var cord_m := DemoMats.plain(Color(0.85, 0.68, 0.25), 0.4, 0.6)
	var line := MeshInstance3D.new()
	var lm := CylinderMesh.new()
	lm.top_radius = 0.008
	lm.bottom_radius = 0.008
	lm.height = 1.3
	line.mesh = lm
	line.material_override = cord_m
	line.position = v(cord.x, cord.y, 1.15 + 0.65)
	holder.add_child(line)
	var tassel := MeshInstance3D.new()
	var tm := CylinderMesh.new()
	tm.top_radius = 0.02
	tm.bottom_radius = 0.045
	tm.height = 0.14
	tassel.mesh = tm
	tassel.material_override = DemoMats.plain(Color(0.6, 0.08, 0.1), 0.9)
	tassel.position = v(cord.x, cord.y, 1.08)
	holder.add_child(tassel)
	curtain_sets.append({"cord": cord, "curtains": curtains, "open": 1.0, "target": 1.0})
	_pose_curtains(curtain_sets[-1])

# A drape's cloth: `w` wide (local x), `h` tall (local y, centred), its
# pleats a wave across it (local z) — 4.5 cm deep every 28 cm, inside the
# 10 cm the drape hangs off its wall.
func _pleats(w: float, h: float, phase: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var amp := 0.045
	var lam := 0.28
	var n := maxi(2, int(ceil(w / 0.035)))
	var k := TAU / lam
	for i in n:
		var u0 := -0.5 * w + w * i / n
		var u1 := -0.5 * w + w * (i + 1) / n
		var z0 := amp * sin(k * u0 + phase)
		var z1 := amp * sin(k * u1 + phase)
		var n0 := Vector3(-amp * k * cos(k * u0 + phase), 0, 1).normalized()
		var n1 := Vector3(-amp * k * cos(k * u1 + phase), 0, 1).normalized()
		var a := Vector3(u0, -0.5 * h, z0)
		var b := Vector3(u1, -0.5 * h, z1)
		var c := Vector3(u1, 0.5 * h, z1)
		var d := Vector3(u0, 0.5 * h, z0)
		for vtx in [[a, n0], [b, n1], [c, n1], [a, n0], [c, n1], [d, n0]]:
			st.set_normal(vtx[1])
			st.add_vertex(vtx[0])
	return st.commit()

func update_curtains(dt: float) -> void:
	for cs in curtain_sets:
		if cs["open"] == cs["target"]:
			continue
		cs["open"] = move_toward(cs["open"], cs["target"], dt * CURTAIN_SPEED)
		_pose_curtains(cs)

# The drop shown and priced: `open` 1 rolled up, 0 down to the floor.
func _pose_curtains(cs: Dictionary) -> void:
	var o: float = cs["open"]
	for c in cs["curtains"]:
		(c["omg"] as OmgDoor).openness = o
		var down := (1.0 - o) * float(c["h"])
		var cloth: MeshInstance3D = c["cloth"]
		cloth.visible = down > 0.01
		cloth.scale = Vector3(1, maxf(1.0 - o, 0.001), 1)
		var at: Vector2 = c["at"]
		cloth.position = v(at.x, at.y, float(c["top"]) - 0.5 * down)

# The cord you look at (or stand at), within `reach`: pulled, its drapes
# go down — or back up. True when there was one.
func pull_cord_ahead(at: Vector2, dir: Vector2, reach := 2.0) -> bool:
	var best := -1
	var best_d := INF
	for i in curtain_sets.size():
		var to: Vector2 = curtain_sets[i]["cord"] - at
		var d := to.length()
		if d > reach or (d > 0.6 and dir.dot(to / d) < 0.8):
			continue
		if d < best_d:
			best_d = d
			best = i
	if best < 0:
		return false
	var cs: Dictionary = curtain_sets[best]
	cs["target"] = 0.0 if cs["target"] > 0.5 else 1.0
	return true

# The door you look at: the doorway (its opening in the wall plane) your
# horizontal view ray `dir` from `at` (plan coordinates) passes through
# within `reach` metres — or the one you stand in. Not just the nearest
# door around you (a door behind your back stays as it is).
func toggle_door_ahead(at: Vector2, dir: Vector2, reach := 2.0) -> int:
	var best := -1
	var best_t := INF
	for i in door_nodes.size():
		var dn: Dictionary = door_nodes[i]
		var p: Vector2 = dn["pos"]
		var hw: float = dn["hw"]
		# axis 0: the wall is the plane x = p.x (the opening runs along y)
		var n := Vector2(1, 0) if dn["axis"] == 0 else Vector2(0, 1)
		var u := Vector2(0, 1) if dn["axis"] == 0 else Vector2(1, 0)
		var dist := (p - at).dot(n)
		var t := 0.0
		if absf(dist) > 0.35 or absf((at - p).dot(u)) > hw + 0.1:
			var dn_dir := dir.dot(n)
			if absf(dn_dir) < 1e-3:
				continue
			t = dist / dn_dir
			if t < 0.0 or t > reach or absf((at + dir * t - p).dot(u)) > hw + 0.15:
				continue
		if t < best_t:
			best_t = t
			best = i
	if best >= 0:
		door_nodes[best]["target"] = 0.0 if door_nodes[best]["target"] > 0.5 else 1.0
	return best

# ------------------------------------------------------------------ street

# The echo canyon south of the town (layout.json "canyon"): two sheer
# rock cliffs, faces 40 m apart, on a rock floor reached by a track from
# the street. A clap (K) between them comes back again and again.
func _canyon() -> void:
	var path := DemoAssets.file(LAYOUT)
	var data = JSON.parse_string(FileAccess.get_file_as_string(path)) if FileAccess.file_exists(path) else null
	var canyon: Dictionary = data.get("canyon", {}) if data is Dictionary else {}
	var cliffs: Array = canyon.get("cliffs", [{"min": [-12, -120], "max": [-6, -70]}, {"min": [34, -120], "max": [40, -70]}])
	var h := float(canyon.get("height", 24.0))
	var rock := geo.B(DemoMats.sandstone(6.0, Color(0.55, 0.52, 0.48)), "rock")
	for c in cliffs:
		rock.box(float(c["min"][0]), float(c["min"][1]), 0.0, float(c["max"][0]), float(c["max"][1]), h)
	var floor_mat := geo.B(DemoMats.sandstone(4.0, Color(0.45, 0.43, 0.4)), "rock", false)
	for pv in _layout_ground():
		if pv["surface"] == "stone":
			var mn: Vector2 = pv["min"]
			var mx: Vector2 = pv["max"]
			floor_mat.box(mn.x, mn.y, 0.0, mx.x, mx.y, 0.02, ["z-"])

func _street() -> void:
	# acoustic asphalt and curbs along the town (the ground tile's
	# extent); beyond it the road runs on as decor, like the far plain
	for part in [[-60.0, 400.0, "auto", ""], [-1950.0, -60.0, "", "far road"], [400.0, 1950.0, "", "far road"]]:
		var road := geo.B(DemoMats.asphalt(), part[2], false, false, part[3])
		road.box(-20.2, part[0], 0.0, -15.8, part[1], 0.03, ["z-"])
		var curb := geo.B(DemoMats.concrete(Color(0.7, 0.7, 0.68)), part[2], false, false, part[3])
		curb.box(-21.0, part[0], 0.0, -20.2, part[1], 0.12, ["z-"])
		curb.box(-15.8, part[0], 0.0, -15.0, part[1], 0.12, ["z-"])
	var dash := geo.B(DemoMats.plain(Color(0.85, 0.85, 0.75), 0.6), "", false, false, "road paint")
	var y := -300.0
	while y < 400.0:
		dash.box(-18.09, y, 0.03, -17.91, y + 2.2, 0.034, ["z-"])
		y += 6.0

func _lamp(x: float, y: float, h := 4.2, shadow := false) -> void:
	var pole := geo.B(DemoMats.metal(Color(0.12, 0.13, 0.14)), "", true, false, "lamp posts: 12 cm poles")
	pole.box(x - 0.06, y - 0.06, 0.0, x + 0.06, y + 0.06, h)
	pole.box(x - 0.22, y - 0.22, h, x + 0.22, y + 0.22, h + 0.08)
	geo.B(DemoMats.glow(Color(1.0, 0.78, 0.48), 2.5), "", false, false, "lamp heads (light)").box(x - 0.15, y - 0.15, h - 0.3, x + 0.15, y + 0.15, h)
	var l := OmniLight3D.new()
	l.position = v(x, y, h - 0.4)
	l.light_color = Color(1.0, 0.74, 0.45)
	l.light_energy = 2.4
	l.omni_range = 13.0
	l.omni_attenuation = 1.2
	l.shadow_enabled = shadow
	root.add_child(l)

func _lamps() -> void:
	for p in [Vector2(8.5, 26.5), Vector2(19.5, 26.0), Vector2(12.0, 38.5), Vector2(20.5, 42.5), Vector2(33.5, 24.8),
			Vector2(6.0, 33.0), Vector2(10.4, 44.0), Vector2(5.0, 50.2), Vector2(11.5, 50.2), Vector2(30.0, 9.0),
			Vector2(23.5, 24.6)]:
		_lamp(p.x, p.y, 4.2, p == Vector2(8.5, 26.5) or p == Vector2(19.5, 26.0))
	var y := -60.0
	while y <= 160.0:
		_lamp(-14.6, y, 6.0)
		y += 28.0
	y = 90.0
	while y < 316.0:
		_lamp(12.2, y, 3.5)
		y += 45.0

# ------------------------------------------------------------------- trees

# Decor only (the engine's plain is open grass): trunks and layered
# crowns on the outskirts, a row along the road and the cathedral path.
# A small lit fountain beside the belfry path, halfway to the tower: a
# square stone basin (acoustic stone, solid), a pedestal with a small
# upper bowl, water glowing faintly, a cool light in the basin and a jet
# of droplets. The water's sound is the "fountain" source in sounds.gd.
func _fountain() -> void:
	var c := FOUNTAIN
	var stone := geo.B(DemoMats.sandstone(1.6, Color(0.8, 0.78, 0.72)), "stone")
	var h := 1.3 # outer half-width
	var t := 0.22 # rim thickness
	stone.box(c.x - h, c.y - h, 0.0, c.x + h, c.y - h + t, 0.5)
	stone.box(c.x - h, c.y + h - t, 0.0, c.x + h, c.y + h, 0.5)
	stone.box(c.x - h, c.y - h + t, 0.0, c.x - h + t, c.y + h - t, 0.5)
	stone.box(c.x + h - t, c.y - h + t, 0.0, c.x + h, c.y + h - t, 0.5)
	stone.box(c.x - h + t, c.y - h + t, 0.0, c.x + h - t, c.y + h - t, 0.1) # the floor of the basin
	stone.box(c.x - 0.17, c.y - 0.17, 0.1, c.x + 0.17, c.y + 0.17, 0.9) # pedestal
	stone.box(c.x - 0.45, c.y - 0.45, 0.9, c.x + 0.45, c.y + 0.45, 1.02) # upper bowl
	var water_c := Color(0.3, 0.62, 0.8)
	geo.B(DemoMats.glow(water_c, 0.5), "", false, false, "fountain water (light)").box(
		c.x - h + t, c.y - h + t, 0.36, c.x + h - t, c.y + h - t, 0.4)
	geo.B(DemoMats.glow(water_c, 0.7), "", false, false, "fountain bowl water (light)").box(
		c.x - 0.4, c.y - 0.4, 1.02, c.x + 0.4, c.y + 0.4, 1.04)
	# the light in the water, cool, washing the basin and the jet
	var l := OmniLight3D.new()
	l.position = v(c.x, c.y, 0.6)
	l.light_color = Color(0.55, 0.85, 1.0)
	l.light_energy = 1.8
	l.omni_range = 6.0
	l.set_meta("no_shadow", true)
	root.add_child(l)
	# the jet: droplets thrown up from the bowl, falling back into the basin
	var jet := GPUParticles3D.new()
	jet.amount = 320
	jet.lifetime = 0.95
	jet.visibility_aabb = AABB(Vector3(-1.5, -1.2, -1.5), Vector3(3, 2.6, 3))
	var pm := ParticleProcessMaterial.new()
	pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE
	pm.emission_sphere_radius = 0.04
	pm.direction = Vector3(0, 1, 0)
	pm.spread = 13.0
	pm.initial_velocity_min = 2.6
	pm.initial_velocity_max = 3.3
	pm.gravity = Vector3(0, -9.8, 0)
	jet.process_material = pm
	var drop := QuadMesh.new()
	drop.size = Vector2(0.035, 0.07)
	var dm := StandardMaterial3D.new()
	dm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	dm.albedo_color = Color(0.7, 0.92, 1.0, 0.8)
	dm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	dm.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
	dm.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
	drop.material = dm
	jet.draw_pass_1 = drop
	jet.position = v(c.x, c.y, 1.06)
	root.add_child(jet)

func _trees() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	var bark := DemoMats.plain(Color(0.22, 0.16, 0.11), 0.95)
	var leaf_cols := [Color(0.12, 0.2, 0.1), Color(0.1, 0.17, 0.09), Color(0.15, 0.22, 0.11)]
	var spots := []
	for y in range(-40, 140, 14):
		spots.append(Vector2(-23.0 + rng.randf_range(-1, 1), y + rng.randf_range(-3, 3)))
	for y in range(80, 300, 22):
		spots.append(Vector2(16.0 + rng.randf_range(-1, 3), y + rng.randf_range(-4, 4)))
		spots.append(Vector2(4.0 + rng.randf_range(-3, 1), y + 11 + rng.randf_range(-4, 4)))
	for p in [Vector2(-6, -8), Vector2(12, -10), Vector2(44, 2), Vector2(46, 20), Vector2(40, 34), Vector2(42, 48),
			Vector2(24, 50), Vector2(28, 60), Vector2(-8, 44), Vector2(-10, 60), Vector2(22, 72), Vector2(-6, 80),
			Vector2(-4, 8), Vector2(18, 6), Vector2(36, -6), Vector2(50, 70)]:
		spots.append(p)
	for p in spots:
		var h := rng.randf_range(5.0, 9.0)
		var t := MeshInstance3D.new()
		var tm := CylinderMesh.new()
		tm.top_radius = 0.12
		tm.bottom_radius = 0.22
		tm.height = h * 0.5
		tm.radial_segments = 8
		t.mesh = tm
		t.material_override = bark
		t.position = v(p.x, p.y, h * 0.25)
		root.add_child(t)
		var leaves := DemoMats.plain(leaf_cols[rng.randi() % 3], 0.9)
		for k in 3:
			var c := MeshInstance3D.new()
			var cm := SphereMesh.new()
			var r := h * (0.32 - 0.07 * k)
			cm.radius = r
			cm.height = r * 1.6
			cm.radial_segments = 10
			cm.rings = 6
			c.mesh = cm
			c.material_override = leaves
			c.position = v(p.x + rng.randf_range(-0.4, 0.4), p.y + rng.randf_range(-0.4, 0.4), h * (0.5 + 0.17 * k))
			root.add_child(c)
