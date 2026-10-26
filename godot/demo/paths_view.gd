# The traced sound paths, drawn (the web demo's "debug rays"): the
# engine's strongest paths of the last tick — listener's ears, every
# reflection or edge bend, the source — as lines coloured by source and
# brighter the more energy they carry, refreshed ten times a second.
# Drawn over everything, so a path around a corner stays readable.
#   var pv := DemoPathsView.new(world); add_child(pv); pv.set_enabled(true)
class_name DemoPathsView
extends MeshInstance3D

const REFRESH_S := 0.1

var world: OmgWorld
var lines := ImmediateMesh.new()
var since := 0.0

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

func set_enabled(on: bool) -> void:
	world.debug_paths = on
	visible = on
	since = REFRESH_S
	if not on:
		lines.clear_surfaces()

func is_enabled() -> bool:
	return visible

func _process(dt: float) -> void:
	if not visible:
		return
	since += dt
	if since < REFRESH_S:
		return
	since = 0.0
	_redraw(world.paths())

func _redraw(paths: Array) -> void:
	lines.clear_surfaces()
	if paths.is_empty():
		return
	var top := 1e-12
	for p in paths:
		top = maxf(top, p["energy"])
	lines.surface_begin(Mesh.PRIMITIVE_LINES)
	for p in paths:
		var c := _colour(p["source"])
		c.a = 0.2 + 0.8 * sqrt(float(p["energy"]) / top)
		var pts: PackedVector3Array = p["points"]
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
