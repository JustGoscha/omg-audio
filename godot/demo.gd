# First-person walk through the omg-audio scene, spatialized by the
# OmgEngine GDExtension. Click to capture the mouse, WASD to move,
# Esc to release. The engine runs its own 20 Hz simulation thread; this
# script only feeds listener pose in and pulls rendered stereo out into
# an AudioStreamGenerator.
extends Node3D

const FS := 48000.0
# Room plan (mirrors crates/omg-scene/src/walkthrough.rs): name, min, max,
# height, outdoor. Coordinates are plan meters; Godot position = (x, h, -y).
const ROOMS := [
	["Living Room", Vector2(0, 0), Vector2(8, 6), 2.7, false],
	["Corridor", Vector2(3.2, 6), Vector2(4.8, 14), 2.4, false],
	["Great Hall", Vector2(0, 14), Vector2(14, 24), 7.0, false],
	["Entrance", Vector2(20, 28), Vector2(22, 34), 2.6, false],
	["Club", Vector2(22, 26), Vector2(32, 38), 4.5, false],
	["Old House", Vector2(24, 16), Vector2(31, 23), 5.9, false],
	["Colonnade", Vector2(16, 15), Vector2(16.5, 21), 2.5, false],
	["Kiosk", Vector2(14.5, 34), Vector2(16.5, 36), 2.7, false],
	["Outside", Vector2(-8, -8), Vector2(42, 46), 0.0, true],
]
const DOORS := [Vector2(4, 6), Vector2(4, 14), Vector2(7, 24), Vector2(20, 31), Vector2(22, 31), Vector2(26.5, 23)]

var engine: OmgEngine
var player: AudioStreamPlayer
var cam: Camera3D
var pos := Vector2(4, 2) # plan position, start in the living room
var heading := 0.0

func _ready() -> void:
	_build_world()
	cam = Camera3D.new()
	cam.position = Vector3(pos.x, 1.6, -pos.y)
	add_child(cam)
	cam.make_current()

	engine = OmgEngine.new()
	engine.setup(FS)
	engine.set_source_samples(0, _synth_music())
	engine.set_source_samples(2, _synth_club())
	engine.set_listener(pos.x, pos.y, 0.0)

	var gen := AudioStreamGenerator.new()
	gen.mix_rate = FS
	gen.buffer_length = 0.15
	player = AudioStreamPlayer.new()
	player.stream = gen
	add_child(player)
	player.play()

func _process(_dt: float) -> void:
	_move(_dt)
	engine.set_listener(pos.x, pos.y, 0.0)
	engine.set_head_yaw(heading)
	var pb: AudioStreamGeneratorPlayback = player.get_stream_playback()
	while pb.get_frames_available() > 512:
		pb.push_buffer(engine.render(mini(pb.get_frames_available(), 2048)))

func _move(dt: float) -> void:
	var f := Vector2(-sin(heading), cos(heading)) # plan-space forward
	var r := Vector2(f.y, -f.x)
	var v := Vector2.ZERO
	if Input.is_key_pressed(KEY_W): v += f
	if Input.is_key_pressed(KEY_S): v -= f
	if Input.is_key_pressed(KEY_D): v += r
	if Input.is_key_pressed(KEY_A): v -= r
	var speed := 8.0 if Input.is_key_pressed(KEY_SHIFT) else 4.0
	pos += v.normalized() * speed * dt if v.length() > 0 else Vector2.ZERO
	pos = pos.clamp(Vector2(-7, -7), Vector2(41, 45))
	cam.position = Vector3(pos.x, 1.6, -pos.y)
	cam.rotation = Vector3(0, heading, 0)

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif e is InputEventKey and e.keycode == KEY_ESCAPE:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif e is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		heading -= e.relative.x * 0.0025

func _build_world() -> void:
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	add_child(sun)
	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.05, 0.06, 0.1)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.4, 0.4, 0.5)
	env.environment = e
	add_child(env)

	var ground := MeshInstance3D.new()
	var gm := PlaneMesh.new()
	gm.size = Vector2(60, 62)
	ground.mesh = gm
	ground.position = Vector3(17, 0, -19)
	var gmat := StandardMaterial3D.new()
	gmat.albedo_color = Color(0.12, 0.16, 0.12)
	ground.material_override = gmat
	add_child(ground)

	for room in ROOMS:
		if room[4]:
			continue
		var size: Vector2 = room[2] - room[1]
		var center: Vector2 = (room[1] + room[2]) * 0.5
		var box := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(size.x, room[3], size.y)
		box.mesh = bm
		box.position = Vector3(center.x, room[3] * 0.5, -center.y)
		var m := StandardMaterial3D.new()
		m.albedo_color = Color(0.5, 0.55, 0.65, 0.3)
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		box.material_override = m
		add_child(box)

	for d in DOORS:
		var marker := MeshInstance3D.new()
		var mm := BoxMesh.new()
		mm.size = Vector3(0.4, 2.1, 0.4)
		marker.mesh = mm
		marker.position = Vector3(d.x, 1.05, -d.y)
		var dm := StandardMaterial3D.new()
		dm.albedo_color = Color(0.2, 0.9, 0.4)
		marker.material_override = dm
		add_child(marker)

# --- tiny CC0-by-construction source loops -------------------------------

func _synth_music() -> PackedFloat32Array:
	# plucked arpeggio, 2 s loop (living-room source)
	var n := int(FS * 2.0)
	var buf := PackedFloat32Array()
	buf.resize(n)
	var notes := [220.0, 277.18, 329.63, 440.0]
	for i in 8:
		var f: float = notes[i % notes.size()] * (2.0 if i >= 4 else 1.0)
		var s0 := int(i * FS * 0.25)
		for k in int(FS * 0.24):
			var t := k / FS
			var env := exp(-t * 6.0)
			buf[s0 + k] += 0.35 * env * sin(TAU * f * t) * (1.0 + 0.3 * sin(TAU * 2.0 * f * t))
	return buf

func _synth_club() -> PackedFloat32Array:
	# four-on-the-floor kick + sub bass at 124 BPM, 4-beat loop
	var spb := 60.0 / 124.0
	var n := int(FS * spb * 4.0)
	var buf := PackedFloat32Array()
	buf.resize(n)
	for beat in 4:
		var s0 := int(beat * spb * FS)
		for k in int(FS * 0.30):
			var t := k / FS
			var f := 48.0 + 110.0 * exp(-t * 35.0)
			if s0 + k < n:
				buf[s0 + k] += 0.9 * exp(-t * 9.0) * sin(TAU * f * t)
	for k in n:
		var t := k / FS
		var g := 0.22 if fmod(t, spb) > spb * 0.5 else 0.0
		buf[k] += g * sin(TAU * 55.0 * t)
	return buf
