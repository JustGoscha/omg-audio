# The traced sound paths, drawn (the web demo's "debug rays"), over
# everything so a path around a corner stays readable.
#
# DIRECTIONS (the default view): where the sound comes from. Every path
# the engine traced — exactly solved (direct, edge bends, mirror chains),
# the walk's samples, and what reaches the ears through an opening by
# light tracing (ears → the bounce → the opening → the source) — is
# gathered by the direction it arrives from at the ears (16 bearings × 5
# heights), whatever its source. A direction's energy is the sum of its
# paths, held a moment as it falls (no flicker); its line is its
# strongest path. Only the strongest few show — one per ~50 paths traced,
# 3 to 10, none 30 dB under the loudest — wider and redder the louder:
# pale blue (−30 dB) through yellow to red (the loudest), 2 to 12 px.
#
# ALL: every path as a thin line, coloured by source (samples faint) —
# the engine's raw picture.
#   var pv := DemoPathsView.new(world); add_child(pv); pv.set_mode(DemoPathsView.DIRECTIONS)
class_name DemoPathsView
extends MeshInstance3D

enum { OFF, DIRECTIONS, ALL }

const REFRESH_S := 0.1
const FADE_S := 0.35        # fade in / out
const HOLD_S := 0.6         # a direction's energy falls with this time constant
const RANGE_DB := 30.0      # directions this far under the loudest still show
const PER_PATHS := 50       # one direction shown per this many paths traced
const SHOW_MIN := 3
const SHOW_MAX := 10
const W_MIN_PX := 2.0
const W_MAX_PX := 12.0
const NEAR_M := 0.6         # the first metres from the ears are not drawn
const AZ_CELLS := 16
const EL_EDGES := [-45.0, -15.0, 15.0, 45.0]
# ALL
const ALL_RANGE_DB := 40.0
const SAMPLED_ALPHA := 0.35

var world: OmgWorld
var lines := ImmediateMesh.new()
var mode := OFF
var since := 0.0
var bundles := {}           # direction cell → {e, points, alpha, target, t}
var shown := {}             # ALL: key → {points, colour, alpha, target}
var top_e := 0.0
var samples := 0

func _init(w: OmgWorld) -> void:
	world = w
	mesh = lines
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	set_meta("omg_material", "none") # never acoustic geometry
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.no_depth_test = true
	material_override = m
	visible = false

func set_mode(m: int) -> void:
	mode = m
	world.debug_paths = m != OFF
	visible = m != OFF
	since = REFRESH_S
	bundles.clear()
	shown.clear()
	top_e = 0.0
	lines.clear_surfaces()

func set_enabled(on: bool) -> void:
	set_mode(DIRECTIONS if on else OFF)

func is_enabled() -> bool:
	return visible

func _process(dt: float) -> void:
	if not visible:
		return
	since += dt
	if since >= REFRESH_S:
		var paths := world.paths()
		if mode == DIRECTIONS:
			_gather(paths, since)
		else:
			_take(paths)
		since = 0.0
	var k := minf(1.0, dt / FADE_S)
	var table := bundles if mode == DIRECTIONS else shown
	for key in table.keys():
		var s: Dictionary = table[key]
		s["alpha"] = lerpf(s["alpha"], s["target"], k)
		if s["target"] == 0.0 and s["alpha"] < 0.01:
			table.erase(key)
	if mode == DIRECTIONS:
		_draw_ribbons()
	else:
		_draw_lines()

# --------------------------------------------------------------- directions

# The direction cell a path arrives from (its first leg from the ears).
static func _cell(pts: PackedVector3Array) -> int:
	if pts.size() < 2:
		return 0
	var d := (pts[1] - pts[0]).normalized()
	var az := fposmod(rad_to_deg(atan2(-d.z, d.x)) + 360.0 / AZ_CELLS * 0.5, 360.0)
	var el := rad_to_deg(asin(clampf(d.y, -1.0, 1.0)))
	var e := 0
	while e < EL_EDGES.size() and el > EL_EDGES[e]:
		e += 1
	return e * AZ_CELLS + int(az / (360.0 / AZ_CELLS)) % AZ_CELLS

func _gather(paths: Array, dt: float) -> void:
	var now := {}  # key → [energy sum, strongest energy, its points]
	for p in paths:
		var pts: PackedVector3Array = p["points"]
		var key := _cell(pts)
		var e: float = p["energy"]
		var acc: Array = now.get(key, [0.0, -1.0, pts])
		acc[0] += e
		if e > acc[1]:
			acc[1] = e
			acc[2] = pts
		now[key] = acc
	# what each direction carries: the sum now, or what it carried a moment
	# ago fading (`HOLD_S`) — whichever is more
	var fall := exp(-dt / HOLD_S)
	for key in now:
		var acc: Array = now[key]
		var b: Dictionary = bundles.get(key, {"e": 0.0, "alpha": 0.0, "target": 0.0, "points": acc[2]})
		b["e"] = maxf(acc[0], b["e"] * fall)
		b["points"] = acc[2]
		bundles[key] = b
	for key in bundles:
		if not now.has(key):
			bundles[key]["e"] *= fall
	var order := bundles.keys()
	order.sort_custom(func(a, c): return bundles[a]["e"] > bundles[c]["e"])
	var top: float = bundles[order[0]]["e"] if not order.is_empty() else 0.0
	top_e = top if top > top_e else lerpf(top_e, top, 0.2)
	var n := clampi(int(ceil(paths.size() / float(PER_PATHS))), SHOW_MIN, SHOW_MAX)
	for i in order.size():
		var b: Dictionary = bundles[order[i]]
		var rel := 10.0 * log(maxf(b["e"], 1e-30) / maxf(top_e, 1e-30)) / log(10.0)
		var t := clampf(1.0 + rel / RANGE_DB, 0.0, 1.0)
		b["t"] = t
		b["target"] = (0.45 + 0.5 * t) if (i < n and t > 0.0) else 0.0

# pale blue → yellow → red
static func _heat(t: float) -> Color:
	var cold := Color(0.55, 0.72, 1.0)
	var warm := Color(1.0, 0.85, 0.3)
	var hot := Color(1.0, 0.12, 0.08)
	return cold.lerp(warm, t * 2.0) if t < 0.5 else warm.lerp(hot, t * 2.0 - 1.0)

func _draw_ribbons() -> void:
	lines.clear_surfaces()
	if bundles.is_empty():
		return
	var cam := get_viewport().get_camera_3d()
	if cam == null:
		return
	var eye := cam.global_position
	# world metres per pixel at 1 m from the camera
	var px := 2.0 * tan(deg_to_rad(cam.fov) * 0.5) / maxf(get_viewport().get_visible_rect().size.y, 1.0)
	var drawn := false
	for b in bundles.values():
		if b["alpha"] < 0.01 or not b.has("t"):
			continue
		var t: float = b["t"]
		var c := _heat(t)
		c.a = b["alpha"]
		var w := lerpf(W_MIN_PX, W_MAX_PX, t) * 0.5 * px
		var pts: PackedVector3Array = b["points"]
		if pts.size() < 2:
			continue
		if not drawn:
			lines.surface_begin(Mesh.PRIMITIVE_TRIANGLES)
			drawn = true
		for i in pts.size() - 1:
			var a := pts[i]
			var z := pts[i + 1]
			if i == 0:
				# from a little way out: at the ears a ribbon fills the view
				var seg := (z - a).length()
				if seg <= NEAR_M:
					continue
				a = a + (z - a) * (NEAR_M / seg)
			var mid := (a + z) * 0.5
			var side := (z - a).cross(mid - eye).normalized()
			var sa := side * (w * (a - eye).length())
			var sz := side * (w * (z - eye).length())
			for v in [a - sa, a + sa, z + sz, a - sa, z + sz, z - sz]:
				lines.surface_set_color(c)
				lines.surface_add_vertex(v)
	if drawn:
		lines.surface_end()

# ---------------------------------------------------------------------- all

func _take(paths: Array) -> void:
	var top := 1e-30
	for p in paths:
		top = maxf(top, p["energy"])
	top_e = top if top > top_e else lerpf(top_e, top, 0.1)
	for s in shown.values():
		s["target"] = 0.0
	for p in paths:
		var kind: int = p["kind"]
		var rel := 10.0 * log(maxf(float(p["energy"]), 1e-30) / top_e) / log(10.0)
		var t := clampf(1.0 + rel / ALL_RANGE_DB, 0.0, 1.0)
		if t <= 0.0:
			continue
		var target := 0.15 + 0.85 * t
		var key: String
		if kind >= 3:
			# samples (and light-traced paths) are new every tick: each
			# fades out on its own
			samples += 1
			key = "s%d" % samples
			target *= SAMPLED_ALPHA
		else:
			var src = p["source"]
			key = "%d:%d:%d" % [src.get_instance_id() if src != null else 0, kind, p["key"]]
		if shown.has(key):
			var s: Dictionary = shown[key]
			s["points"] = p["points"]
			s["target"] = target
		else:
			shown[key] = {"points": p["points"], "colour": colour(p["source"]), "alpha": 0.0,
				"target": target if kind < 3 else 0.0}
			if kind >= 3:
				shown[key]["alpha"] = target # a sample shows, then fades

func _draw_lines() -> void:
	lines.clear_surfaces()
	if shown.is_empty():
		return
	lines.surface_begin(Mesh.PRIMITIVE_LINES)
	for s in shown.values():
		var c: Color = s["colour"]
		c.a = s["alpha"]
		var pts: PackedVector3Array = s["points"]
		for i in pts.size() - 1:
			lines.surface_set_color(c)
			lines.surface_add_vertex(pts[i])
			lines.surface_set_color(c)
			lines.surface_add_vertex(pts[i + 1])
	lines.surface_end()

# A steady hue per source node.
static func colour(src) -> Color:
	if src == null:
		return Color(1, 1, 1)
	var h := float(hash(src.get_instance_id()) % 1000) / 1000.0
	return Color.from_hsv(h, 0.7, 1.0)
