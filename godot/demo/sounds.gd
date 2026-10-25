# The demo's sound design — the web demo's, event for event:
# per-room loops, footsteps (distance-driven strides, ±1.5 dB / ±3 %),
# owls on their timers, the carillon (B, and by itself every 30 s–2 min),
# whispers at the ear (V only), passing cars with per-spawn motor loops,
# thrown balls (woosh + bounces), rain (R) and the cricket bed.
# Everything here is demo logic on top of the engine's generic nodes.
class_name DemoSounds
extends Node

const A := "res://../assets/"

# The web mixer (web/main.js MIXER) sets each source's real SPL:
# level = 90 dB + 20·log10(authored gain) + (fader default − anchor).
static func spl(authored_gain: float, anchor: float, fader: float) -> float:
	return 90.0 + 20.0 * log(authored_gain) / log(10.0) + fader - anchor

# name, plan position, z, loop file, level (dB SPL @ 1 m)
var STATIC := [
	["music", Vector2(2, 3), 1.5, "aria48.ogg", spl(1.0, 90, 72)],
	["voice", Vector2(10.5, 20.5), 1.5, "alice48.ogg", spl(0.5, 84, 65)],
	["flute", Vector2(8, 66), 1.5, "flute48.ogg", spl(1.0, 88, 76)],
	["radio", Vector2(37.5, 12.5), -1.5, "radio48.ogg", spl(0.5, 80, 64)],
	["owl·cathedral", Vector2(4, 53), 24.25, "", spl(1.6, 74, 66)],
	["owl·house", Vector2(27.5, 19.5), 6.3, "", spl(1.4, 74, 66)],
	["bells", Vector2(10, 321), 25.5, "", spl(30.0, 100, 104)],
]
var CLUB_LEVEL := spl(5.0, 104, 104) - 6.0206 # power split over 4 sample-locked speakers
var BALL_LEVEL := spl(0.9, 89, 89)
var CAR_LEVEL := spl(2.2, 92, 86)
var FEET_LEVEL := spl(1.0, 78, 62)

const SURFACES := ["wood", "carpet", "concrete", "grass"]
const RAIN_LEVELS := [0.0, 0.3, 0.65, 1.0]
const RAIN_NAMES := ["off", "drizzle", "rain", "downpour"]
const CAR_LANES := [-18.9, -17.1]

var world: OmgWorld
var root: Node3D
var level: DemoLevel
var player: DemoPlayer
var sources := {}       # name → OmgSource
var balls: Array[OmgSource] = []
var cars_src: Array[OmgSource] = []
var feet: OmgSource
var clip := {}          # name → clip id
var steps := []         # [surface][variant] → clip id
var t0 := 0.0
var wild := {}
var stride_acc := 0.0
var stride_next := 1.5
var last_step := -1.0
var last_variant := -1
var rain_level := 0
var projs := []
var cars := []
var next_car := 0.0
var rain_fx: GPUParticles3D

func _now() -> float:
	return Time.get_ticks_msec() / 1000.0

func _src(name: String, p: Vector2, z: float, file: String, level_db: float, active := true) -> OmgSource:
	var s := OmgSource.new()
	s.name = name.replace("·", "_")
	s.file = (A + file) if file != "" else ""
	s.level_db = level_db
	s.active = active
	s.position = DemoGeo.P(p.x, p.y, z)
	root.add_child(s)
	return s

func setup(w: OmgWorld, r: Node3D, l: DemoLevel, pl: DemoPlayer) -> void:
	world = w
	root = r
	level = l
	player = pl
	for d in STATIC:
		sources[d[0]] = _src(d[0], d[1], d[2], d[3], d[4])
	var i := 0
	for s in DemoLevel.CLUB_SPEAKERS:
		sources["club%d" % i] = _src("club%d" % i, s, 1.9, "club48.ogg", CLUB_LEVEL)
		i += 1
	for k in 3:
		balls.append(_src("ball%d" % k, Vector2.ZERO, 1.5, "", BALL_LEVEL, false))
	for k in 2:
		cars_src.append(_src("car%d" % k, Vector2(-18, 0), 0.7, "", CAR_LEVEL, false))
	feet = _src("feet", Vector2(3, 3), 0.2, "", FEET_LEVEL)

	# fx bank, shaped on import as the web shapes it on decode
	clip["woosh"] = world.add_clip(_woosh(world.sample_rate()))
	clip["thump"] = world.load_clip(A + "fx_thump.ogg", {"peak": 0.4})
	for surf in SURFACES:
		var row := []
		for v in 5:
			row.append(world.load_clip(A + "steps/%s_%d.ogg" % [surf, v], {
				"peak": 0.32, "highpass_hz": 160.0, "trim_start": 0.01, "trim_end": 0.004,
				"fade_in": 0.001, "fade_out": 0.01}))
		steps.append(row)
	for k in 3:
		clip["owl%d" % k] = world.load_clip(A + "birds/owl_%d.ogg" % k, {"peak": 0.5})
	clip["bell_big"] = world.load_clip(A + "birds/bell_big.ogg", {"peak": 0.6, "max_seconds": 16.0, "fade_out": 2.0})
	clip["bell_small"] = world.load_clip(A + "birds/bell_small.ogg", {"peak": 0.5, "max_seconds": 10.0, "fade_out": 2.0})
	for k in 2:
		clip["whisper%d" % k] = world.load_clip(A + "whisper/whisper_%d.ogg" % k,
			{"peak": 0.5, "trim_start": 0.02, "max_seconds": 7.0, "fade_out": 1.5})
	# warm the motor decodes (each spawn picks one, varispeed per car)
	for m in 4:
		_motor_clip(m, 1.0)
	t0 = _now()
	wild = {"owl1": 8.0 + randf() * 12.0, "owl2": 16.0 + randf() * 16.0, "bells": 10.0 + randf() * 15.0}
	next_car = 3.0
	_owl_figures()
	_rain_particles()

# Ball flight woosh, synthesised (band-passed noise, a swell that dies as
# the ball slows) — web/main.js makeWoosh.
func _woosh(sr: float) -> PackedFloat32Array:
	var n := int(sr * 5.0)
	var out := PackedFloat32Array()
	out.resize(n)
	var lp := 0.0
	var lp2 := 0.0
	var wob := randf() * 6.28
	var peak := 1e-6
	for i in n:
		var t := i / sr
		var f := 260.0 + 260.0 * exp(-t * 0.8) * (1.0 + 0.18 * sin(wob + t * 4.7))
		var k := 1.0 - exp(-TAU * f / sr)
		lp += k * (randf() * 2.0 - 1.0 - lp)
		lp2 += k * (lp - lp2)
		var env := minf(1.0, t * 7.0) * exp(-t * 0.55)
		out[i] = (lp - lp2) * env
		peak = maxf(peak, absf(out[i]))
	for i in n:
		out[i] *= 0.12 / peak
	return out

func _motor_clip(m: int, rate: float) -> int:
	return world.load_clip(A + "motor%d48.ogg" % m, {"rate": snappedf(rate, 0.05), "normalize": true})

# ---------------------------------------------------------------- events

func ring_bells() -> void:
	wild["bells"] = _now() - t0 + 30.0 + randf() * 90.0
	sources["bells"].play_clip(clip["bell_big"] if randf() < 0.75 else clip["bell_small"], 1.0, 1.0)

func whisper() -> void:
	world.whisper(clip["whisper%d" % (randi() % 2)], randf() < 0.5, 0.05 + randf() * 0.07, 0.55 + randf() * 0.25)

func cycle_rain() -> String:
	rain_level = (rain_level + 1) % RAIN_LEVELS.size()
	world.rain_intensity = RAIN_LEVELS[rain_level]
	return RAIN_NAMES[rain_level]

func _step(force := false) -> void:
	if not force and (_now() - last_step < 0.32):
		return
	last_step = _now()
	stride_acc = 0.0
	stride_next = 1.35 + randf() * 0.3
	var p := player.plan()
	var surf: String = DemoLevel.room_at(p.x, p.y, p.z + 0.5)[5]
	var v := randi() % 5
	if v == last_variant:
		v = (v + 1) % 5
	last_variant = v
	var gain := pow(10.0, (randf() * 3.0 - 1.5) / 20.0)
	var rate := 0.97 + randf() * 0.06
	feet.play_clip(steps[SURFACES.find(surf)][v], gain, rate)

func throw_ball() -> void:
	if projs.size() >= 3:
		return
	var used := projs.map(func(p): return p["slot"])
	var slot := 0
	while slot in used:
		slot += 1
	var dir := player.look_dir()
	var at := player.cam.global_position + Vector3(dir.x, 0, dir.z).normalized() * 0.4
	var mesh := MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.12
	sm.height = 0.24
	mesh.mesh = sm
	mesh.material_override = DemoMats.glow(Color(1.0, 0.95, 0.65), 3.0)
	var light := OmniLight3D.new()
	light.light_color = Color(1.0, 0.8, 0.5)
	light.light_energy = 1.5
	light.omni_range = 6.0
	mesh.add_child(light)
	root.add_child(mesh)
	mesh.global_position = at
	var src: OmgSource = balls[slot]
	src.global_position = at
	src.active = true
	src.play_clip(clip["woosh"], 1.0, 1.0)
	projs.append({"slot": slot, "mesh": mesh, "light": light, "vel": dir * 13.0 + Vector3(0, 2.5, 0), "landed": -1.0, "fade": -1.0})

func _update_balls(dt: float) -> void:
	var space := player.get_world_3d().direct_space_state
	for p in projs.duplicate():
		var mesh: MeshInstance3D = p["mesh"]
		var src: OmgSource = balls[p["slot"]]
		var now := _now()
		if p["fade"] > 0.0:
			var k: float = (now - p["fade"]) / 1.2
			(p["light"] as OmniLight3D).light_energy = maxf(0.0, 1.5 * (1.0 - k))
			mesh.scale = Vector3.ONE * maxf(0.01, 1.0 - k)
			if k > 1.2:
				mesh.queue_free()
				src.active = false
				projs.erase(p)
			continue
		var vel: Vector3 = p["vel"]
		vel.y -= 9.8 * dt
		var from := mesh.global_position
		var to := from + vel * dt
		var q := PhysicsRayQueryParameters3D.create(from, to + vel.normalized() * 0.12)
		q.exclude = [player.get_rid()]
		var hit := space.intersect_ray(q)
		if not hit.is_empty():
			var n: Vector3 = hit["normal"]
			var speed := vel.length()
			vel = vel.bounce(n) * (0.55 if n.y > 0.7 else 0.6)
			if n.y > 0.7:
				vel.x *= 0.8
				vel.z *= 0.8
				if p["landed"] < 0.0:
					p["landed"] = now
			if speed > 1.5:
				src.play_clip(clip["thump"], 1.0, 1.0)
			to = hit["position"] + n * 0.13
		mesh.global_position = to
		src.global_position = to
		p["vel"] = vel
		if p["landed"] > 0.0 and now - p["landed"] > 3.0:
			p["fade"] = now
			src.stop_clip(clip["woosh"])

func _spawn_car() -> void:
	var free := -1
	for s in 2:
		if not cars.any(func(c): return c["slot"] == s):
			free = s
			break
	if free < 0:
		return
	var north := free == 1
	var body := Node3D.new()
	var paint := DemoMats.plain(Color(0.25, 0.32, 0.45) if north else Color(0.45, 0.2, 0.15), 0.25, 0.6)
	var b := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(1.75, 0.6, 4.2)
	b.mesh = bm
	b.material_override = paint
	b.position.y = 0.6
	body.add_child(b)
	var cabin := MeshInstance3D.new()
	var cm := BoxMesh.new()
	cm.size = Vector3(1.55, 0.5, 2.1)
	cabin.mesh = cm
	cabin.material_override = DemoMats.glass(Color(0.1, 0.12, 0.15, 0.85))
	cabin.position = Vector3(0, 1.15, 0.2)
	body.add_child(cabin)
	for wx in [-0.8, 0.8]:
		for wz in [-1.35, 1.35]:
			var wheel := MeshInstance3D.new()
			var wm := CylinderMesh.new()
			wm.top_radius = 0.33
			wm.bottom_radius = 0.33
			wm.height = 0.22
			wheel.mesh = wm
			wheel.material_override = DemoMats.plain(Color(0.05, 0.05, 0.05), 0.9)
			wheel.rotation_degrees = Vector3(0, 0, 90)
			wheel.position = Vector3(wx, 0.33, wz)
			body.add_child(wheel)
	# front is −z (north-bound) — the body is turned for south-bound
	for hx in [-0.6, 0.6]:
		var hl := MeshInstance3D.new()
		var hm := BoxMesh.new()
		hm.size = Vector3(0.3, 0.12, 0.05)
		hl.mesh = hm
		hl.material_override = DemoMats.glow(Color(1.0, 0.95, 0.8), 6.0)
		hl.position = Vector3(hx, 0.7, -2.11)
		body.add_child(hl)
		var tl := MeshInstance3D.new()
		tl.mesh = hm
		tl.material_override = DemoMats.glow(Color(1.0, 0.1, 0.05), 4.0)
		tl.position = Vector3(hx, 0.72, 2.11)
		body.add_child(tl)
	var beam := SpotLight3D.new()
	beam.position = Vector3(0, 0.7, -2.3)
	beam.light_color = Color(1.0, 0.95, 0.85)
	beam.light_energy = 6.0
	beam.spot_range = 40.0
	beam.spot_angle = 28.0
	beam.rotation_degrees = Vector3(-4, 0, 0)
	body.add_child(beam)
	if not north:
		body.rotation_degrees = Vector3(0, 180, 0)
	root.add_child(body)
	var y := (-1.0 if north else 1.0) * (900.0 + randf() * 900.0)
	var motor := _motor_clip(randi() % 4, 0.85 + randf() * 0.4)
	cars.append({"slot": free, "mesh": body, "x": CAR_LANES[1 if north else 0], "y": y, "north": north,
		"vy": (1.0 if north else -1.0) * (12.0 + randf() * 12.0), "base": 0.6 + randf(), "motor": motor, "started": false})

func _update_cars(dt: float) -> void:
	var now := _now() - t0
	if now > next_car:
		_spawn_car()
		if randf() < 0.35:
			_spawn_car()
		next_car = now + 20.0 + randf() * 25.0
	var me := player.plan()
	for c in cars.duplicate():
		var src: OmgSource = cars_src[c["slot"]]
		if not c["started"] and world.clip_state(c["motor"]) == 1:
			c["started"] = src.set_loop_clip(c["motor"])
		c["y"] += c["vy"] * dt
		var pos := DemoGeo.P(c["x"], c["y"], 0.0)
		(c["mesh"] as Node3D).global_position = pos
		src.global_position = pos + Vector3(0, 0.7, 0)
		# perceptual envelope (web): full level within 140 m, gone by 320 m
		var d := Vector2(c["x"] - me.x, c["y"] - me.y).length()
		var vol: float = c["base"] * (1.0 if d < 140.0 else maxf(0.0, 1.0 - (d - 140.0) / 180.0))
		src.active = c["started"] and vol > 0.01
		src.level_db = CAR_LEVEL + 20.0 * log(maxf(vol, 1e-4)) / log(10.0)
		var gone: bool = c["y"] > 560.0 if c["north"] else c["y"] < -500.0
		if gone:
			(c["mesh"] as Node3D).queue_free()
			src.active = false
			src.stop()
			cars.erase(c)

func _update_wildlife() -> void:
	var s := _now() - t0
	if s > wild["owl1"]:
		wild["owl1"] = s + 20.0 + randf() * 40.0
		sources["owl·cathedral"].play_clip(clip["owl%d" % (randi() % 3)], 1.0, 1.0)
	if s > wild["owl2"]:
		wild["owl2"] = s + 25.0 + randf() * 45.0
		sources["owl·house"].play_clip(clip["owl%d" % (randi() % 3)], 1.0, 1.0)
	if s > wild["bells"]:
		ring_bells()

func _owl_figures() -> void:
	for n in ["owl·cathedral", "owl·house"]:
		var s: OmgSource = sources[n]
		var owl := Node3D.new()
		var body := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = 0.16
		sm.height = 0.42
		body.mesh = sm
		var feathers := DemoMats.plain(Color(0.38, 0.3, 0.22), 0.9)
		body.material_override = feathers
		owl.add_child(body)
		var head := MeshInstance3D.new()
		var hm := SphereMesh.new()
		hm.radius = 0.12
		hm.height = 0.22
		head.mesh = hm
		head.material_override = feathers
		head.position = Vector3(0, 0.27, 0)
		owl.add_child(head)
		for ex in [-0.05, 0.05]:
			var eye := MeshInstance3D.new()
			var em := SphereMesh.new()
			em.radius = 0.025
			em.height = 0.05
			eye.mesh = em
			eye.material_override = DemoMats.glow(Color(1.0, 0.75, 0.2), 1.5)
			eye.position = Vector3(ex, 0.29, -0.1)
			owl.add_child(eye)
		root.add_child(owl)
		owl.global_position = s.global_position + Vector3(0, -0.28, 0)

func _rain_particles() -> void:
	rain_fx = GPUParticles3D.new()
	rain_fx.amount = 6000
	rain_fx.lifetime = 1.4
	rain_fx.visibility_aabb = AABB(Vector3(-20, -25, -20), Vector3(40, 40, 40))
	var pm := ParticleProcessMaterial.new()
	pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	pm.emission_box_extents = Vector3(18, 0.5, 18)
	pm.direction = Vector3(0.05, -1, 0)
	pm.spread = 2.0
	pm.initial_velocity_min = 11.0
	pm.initial_velocity_max = 14.0
	pm.gravity = Vector3(0, -4, 0)
	rain_fx.process_material = pm
	var drop := QuadMesh.new()
	drop.size = Vector2(0.012, 0.45)
	var dm := StandardMaterial3D.new()
	dm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	dm.albedo_color = Color(0.7, 0.75, 0.85, 0.35)
	dm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	dm.billboard_mode = BaseMaterial3D.BILLBOARD_FIXED_Y
	drop.material = dm
	rain_fx.draw_pass_1 = drop
	rain_fx.emitting = false
	root.add_child(rain_fx)

func _process(dt: float) -> void:
	if world == null:
		return
	# the feet: always on, at ground level under the body
	var g := player.global_position
	feet.global_position = g + Vector3(0, 0.2, 0)
	if player.landed:
		_step(true)
	elif player.stepped > 0.0:
		stride_acc += player.stepped
		if stride_acc >= stride_next:
			_step()
	_update_wildlife()
	_update_balls(dt)
	_update_cars(dt)
	# rain follows the camera; density follows the weather
	var target: float = RAIN_LEVELS[rain_level]
	rain_fx.emitting = target > 0.0
	rain_fx.amount_ratio = maxf(0.05, target)
	rain_fx.global_position = player.cam.global_position + Vector3(0, 14, 0)
