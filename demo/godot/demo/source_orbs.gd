# A sphere at every sounding source, pulsing with what it brings to the
# ears now — its share of the output (`OmgWorld.meters`, its peak since
# the last refresh): −70 … −10 dBFS sizes it 6 … 46 cm across, swelling
# at once and easing back over a quarter second; its colour the source's
# own (as the paths view's every-path lines), seen through walls. The
# player's own sources (feet, hands, lips) show none.
class_name DemoSourceOrbs
extends Node3D

const FLOOR_DB := -70.0
const SPAN_DB := 60.0
const R_MIN := 0.06
const R_SPAN := 0.4
const RELEASE_S := 0.25
const OWN := 1.0 # m: sources this close to the camera are the player's

var world: OmgWorld
var on := false
var orbs := {} # source instance id → {node: MeshInstance3D, mat, level}
var sphere := SphereMesh.new()

func _init(w: OmgWorld) -> void:
	world = w
	set_meta("omg_material", "none") # never acoustic geometry
	sphere.radius = 1.0
	sphere.height = 2.0
	sphere.radial_segments = 16
	sphere.rings = 8
	visible = false

func set_enabled(v: bool) -> void:
	on = v
	visible = v
	if not v:
		for o in orbs.values():
			o["node"].queue_free()
		orbs.clear()

func _process(dt: float) -> void:
	if not on:
		return
	var cam := get_viewport().get_camera_3d()
	var eye := cam.global_position if cam != null else Vector3.ZERO
	var m: Dictionary = world.meters()
	var seen := {}
	for s in get_tree().get_nodes_in_group("omg_source"):
		var src := s as OmgSource
		var id := src.get_instance_id()
		var sounding := src.active and not src.muted and src.is_visible_in_tree() and src.global_position.distance_to(eye) > OWN
		if not sounding:
			continue
		seen[id] = true
		var o: Dictionary = orbs.get(id, {})
		if o.is_empty():
			var mi := MeshInstance3D.new()
			mi.mesh = sphere
			mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			var mat := StandardMaterial3D.new()
			mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			mat.no_depth_test = true
			mat.albedo_color = DemoPathsView.colour(src)
			mi.material_override = mat
			add_child(mi)
			o = {"node": mi, "mat": mat, "level": 0.0}
			orbs[id] = o
		var lv: Vector2 = m.get(src.source_id(), Vector2.ZERO)
		var t := clampf((linear_to_db(maxf(lv.x, 1e-7)) - FLOOR_DB) / SPAN_DB, 0.0, 1.0)
		var level: float = o["level"]
		level = t if t > level else lerpf(level, t, minf(1.0, dt / RELEASE_S))
		o["level"] = level
		var mi: MeshInstance3D = o["node"]
		mi.global_position = src.global_position
		mi.scale = Vector3.ONE * (R_MIN + R_SPAN * level) * 0.5
		var mat: StandardMaterial3D = o["mat"]
		mat.albedo_color.a = 0.2 + 0.6 * level
	for id in orbs.keys():
		if not seen.has(id):
			orbs[id]["node"].queue_free()
			orbs.erase(id)
