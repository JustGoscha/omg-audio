# The traced sound paths, drawn (the web demo's "debug rays"): listener's
# ears, every reflection or edge bend, the source — lines coloured by
# source, brighter the more energy they carry. Drawn over everything, so
# a path around a corner stays readable.
#
# Mode EXACT shows only the exactly solved paths (direct, edge bends,
# mirror chains): the same path tick after tick, moving smoothly with the
# listener — a steady picture. ALL adds the walk's strongest random
# samples of each tick, faint. Paths fade in and out instead of popping.
#   var pv := DemoPathsView.new(world); add_child(pv); pv.set_mode(DemoPathsView.EXACT)
class_name DemoPathsView
extends MeshInstance3D

enum { OFF, EXACT, ALL }

const REFRESH_S := 0.1
const FADE_S := 0.35        # fade in / out
const RANGE_DB := 40.0      # paths this far under the strongest still show
const SAMPLED_ALPHA := 0.35
const MIRRORS_PER_SOURCE := 12  # the strongest chains (direct and bends always)

var world: OmgWorld
var lines := ImmediateMesh.new()
var mode := OFF
var since := 0.0
var shown := {}             # key → {points, colour, alpha, target}
var top_e := 0.0            # the strongest path's energy, smoothed
var samples := 0            # unique keys for sampled paths

func _init(w: OmgWorld) -> void:
	world = w
	mesh = lines
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	set_meta("omg_material", "none") # never acoustic geometry
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.no_depth_test = true
	material_override = m
	visible = false

func set_mode(m: int) -> void:
	mode = m
	world.debug_paths = m != OFF
	visible = m != OFF
	since = REFRESH_S
	if m == OFF:
		shown.clear()
		top_e = 0.0
		lines.clear_surfaces()

func set_enabled(on: bool) -> void:
	set_mode(EXACT if on else OFF)

func is_enabled() -> bool:
	return visible

func _process(dt: float) -> void:
	if not visible:
		return
	since += dt
	if since >= REFRESH_S:
		since = 0.0
		_take(world.paths())
	# ease every path toward its target; drop the faded
	var k := minf(1.0, dt / FADE_S)
	for key in shown.keys():
		var s: Dictionary = shown[key]
		s["alpha"] = lerpf(s["alpha"], s["target"], k)
		if s["target"] == 0.0 and s["alpha"] < 0.01:
			shown.erase(key)
	_draw()

func _take(paths: Array) -> void:
	var top := 1e-30
	for p in paths:
		if mode == ALL or p["kind"] < 3:
			top = maxf(top, p["energy"])
	# the reference rises at once and falls slowly (no flicker in scale)
	top_e = top if top > top_e else lerpf(top_e, top, 0.1)
	for s in shown.values():
		s["target"] = 0.0
	var mirrors := {}       # source → chains taken (paths come strongest first)
	for p in paths:
		var kind: int = p["kind"]
		if kind == 3 and mode != ALL:
			continue
		if kind == 2:
			var n: int = mirrors.get(p["source"], 0)
			if n >= MIRRORS_PER_SOURCE:
				continue
			mirrors[p["source"]] = n + 1
		var rel := 10.0 * log(maxf(float(p["energy"]), 1e-30) / top_e) / log(10.0)
		var t := clampf(1.0 + rel / RANGE_DB, 0.0, 1.0)
		if t <= 0.0:
			continue
		var target := 0.15 + 0.85 * t
		var key: String
		if kind == 3:
			# samples are new every tick: each fades out on its own
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
			shown[key] = {"points": p["points"], "colour": _colour(p["source"]), "alpha": 0.0,
				"target": target if kind < 3 else 0.0}
			if kind == 3:
				shown[key]["alpha"] = target # a sample shows, then fades

func _draw() -> void:
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
static func _colour(src) -> Color:
	if src == null:
		return Color(1, 1, 1)
	var h := float(hash(src.get_instance_id()) % 1000) / 1000.0
	return Color.from_hsv(h, 0.7, 1.0)
