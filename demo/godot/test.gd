# Headless smoke test for the GDExtension and the demo built on it:
#   Godot --headless --path godot -s test.gd
# (a fresh checkout needs `Godot --headless --path godot --import` first).
# Exits 0 on success, 1 on failure. (Not `assert`: a failed assert in a
# headless -s script halts _initialize and Godot then idles forever.)
extends SceneTree

var started := false
var player: DemoPlayer

# run on the first frame: during _initialize the root is not in the tree
# yet (no global transforms, no groups)
func _process(_dt: float) -> bool:
	if not started:
		started = true
		_go()
	return false

func _go() -> void:
	var err := _run()
	if err.is_empty():
		err = await _walks()
	if err.is_empty():
		print("GODOT SMOKE TEST PASSED")
		quit(0)
	else:
		printerr("GODOT SMOKE TEST FAILED: ", err)
		quit(1)

static func _peak(out: PackedVector2Array) -> float:
	var p := 0.0
	for v in out:
		p = maxf(p, maxf(absf(v.x), absf(v.y)))
	return p

static func _rms(out: PackedVector2Array) -> float:
	var s := 0.0
	for v in out:
		s += v.x * v.x + v.y * v.y
	return sqrt(s / maxf(1.0, 2.0 * out.size()))

# Let the 20 Hz simulation thread trace the current input a few times.
func _settle(world: OmgWorld, ms: int) -> void:
	var t := Time.get_ticks_msec()
	while Time.get_ticks_msec() - t < ms:
		world.update_now()
		OS.delay_msec(25)

func _run() -> String:
	# ---- 1. the demo level, its geometry collected by the engine
	var level_root := Node3D.new()
	level_root.name = "Level"
	root.add_child(level_root)
	var level := DemoLevel.new(level_root)
	level.build()
	# textures and sound must agree: every visual→acoustic pairing, and
	# no big decor left out of the sound world without a reason
	var audit: Dictionary = level.geo.audit()
	for line in audit["pairs"]:
		print("  ", line)
	if not audit["warnings"].is_empty():
		return "material audit: %s" % [audit["warnings"]]
	player = DemoPlayer.new()
	root.add_child(player)
	player.global_position = DemoGeo.P(3.0, 3.0, 0.05)
	player.set_heading(PI / 2.0)

	var world := OmgWorld.new()
	world.realtime = false # render() by hand
	world.use_gpu = not OS.has_environment("OMG_CPU") # OMG_CPU=1: the CPU walk
	if OS.has_environment("OMG_QUALITY"):
		world.quality = int(OS.get_environment("OMG_QUALITY"))
	world.geometry_root = NodePath("../Level")
	world.ambience_file = DemoSounds.A + "night-nature48.ogg"
	world.ambience_seconds = 15.0
	world.rain_bank_file = DemoSounds.A + "drops48.ogg"
	root.add_child(world)
	var g := world.rebuild_geometry()
	print("geometry: ", g)
	if g["triangles"] < 1000: return "too little acoustic geometry: %s" % [g]
	if g["boxes"] < 12: return "cathedral columns missing as acoustic boxes: %s" % [g]

	# ---- 2. one fx-only source: a one-shot must reach the ears
	var probe := OmgSource.new()
	probe.position = DemoGeo.P(4.0, 3.0, 1.5)
	level_root.add_child(probe)
	var owl := world.load_clip(DemoSounds.A + "birds/owl_0.ogg", {"peak": 0.5})
	if not world.wait_for_clips(60.0): return "clips did not decode"
	if world.clip_state(owl) != 1: return "owl clip failed to decode"
	world.ambience_db = -200.0
	_settle(world, 300)
	var quiet := _peak(world.render(4800))
	if not probe.play_clip(owl, 1.0, 1.0): return "play_clip refused a decoded clip"
	world.update_now()
	var shot := world.render(24000)
	print("fx: silence peak %.5f → one-shot peak %.4f" % [quiet, _peak(shot)])
	if _peak(shot) < 0.01 or _peak(shot) < 10.0 * quiet: return "an fx one-shot produced no output"
	probe.queue_free()

	# ---- 3. the whole demo: every source of the web demo, loops playing
	var sounds := DemoSounds.new()
	root.add_child(sounds)
	sounds.setup(world, level_root, level, player)
	world.ambience_db = -16.0
	if not world.wait_for_clips(120.0): return "demo clips did not decode"
	# activate everything that can sound: balls and cars too
	for b in sounds.balls:
		b.active = true
	for c in sounds.cars_src:
		c.active = true
		c.set_loop_clip(sounds._motor_clip(0, 1.0))
	OS.delay_msec(50)
	_settle(world, 600)
	var out := world.render(9600)
	var st := world.stats()
	print("stats: ", st)
	var n_src := sounds.sources.size() + sounds.balls.size() + sounds.cars_src.size() + 1
	if st["sources"] < n_src: return "expected %d sources, the engine has %d" % [n_src, st["sources"]]
	if n_src < 17: return "the demo should carry the web's 14 sources (club as 4 speakers): %d" % n_src
	print("living room: rms %.4f peak %.4f rt60 %.2f s" % [_rms(out), _peak(out), world.stats()["rt60"]])
	if out.size() != 9600: return "wrong frame count"
	if _peak(out) <= 0.005: return "engine is silent in the living room"
	if _peak(out) >= 1.0: return "engine is clipping"

	# turning the head must change the ear signals (binaural sanity)
	player.rotation.y += PI / 2.0
	world.update_now()
	var out2 := world.render(4800)
	var diff := 0.0
	for v in out2:
		diff = maxf(diff, absf(v.x - v.y))
	if diff <= 1e-4: return "no interaural difference after a head turn"

	# the debug view: the traced paths, from the ears to their sources
	world.debug_paths = true
	_settle(world, 300)
	var paths := world.paths()
	world.debug_paths = false
	world.update_now()
	print("debug paths: %d" % paths.size())
	if paths.is_empty(): return "debug_paths on but paths() is empty"
	var p0: Dictionary = paths[0]
	var pts: PackedVector3Array = p0["points"]
	if p0["source"] == null or pts.size() < 2: return "a debug path without its source node or points: %s" % [p0]
	if pts[0].distance_to(player.cam.global_position) > 0.3:
		return "debug paths start at %s, the ears are at %s" % [pts[0], player.cam.global_position]

	# real-time budget with every demo source live: the output thread must
	# render faster than real time with headroom (the game thread only copies)
	var t0 := Time.get_ticks_usec()
	world.render(96000)
	var secs := (Time.get_ticks_usec() - t0) / 1e6
	var rtf := 2.0 / secs
	st = world.stats()
	print("render speed (%d sources, %d rendering, %d paths): %.2fx realtime · sim tick %.1f ms (%s) · budget %d/%d/%d" % [
		st["sources"], st["rendering"], st["paths"], rtf, st["tick_ms"], "gpu" if st["gpu"] else "cpu", st["taps"], st["points"], st["voices"]])
	if rtf < 1.5: return "render too slow: %.2fx realtime" % rtf
	if st["tick_ms"] > 50.0: return "simulation tick over its 50 ms budget: %.1f ms" % st["tick_ms"]
	# the same at fixed budgets: the floor, and the web's full density
	for b in [0, 1, 4]:
		world.set_render_budget(b)
		world.update_now()
		world.render(4800)
		t0 = Time.get_ticks_usec()
		world.render(96000)
		var sb := world.stats()
		print("  budget rung %d (%d paths / %d hrtf / %d voices): %.2fx realtime" % [b, sb["taps"], sb["points"], sb["voices"], 2.0 / ((Time.get_ticks_usec() - t0) / 1e6)])
	world.set_render_budget(-1)

	# ---- 4. walls: the club heard from outside its closed door is muffled
	var club_in := _listen(world, player, Vector3(31, 30, 0.05))
	var club_out := _listen(world, player, Vector3(17, 31, 0.05))
	print("club inside %.4f · outside the closed inner door %.4f" % [club_in, club_out])
	if club_out >= club_in * 0.7: return "walls do not attenuate the club"

	# ---- 5. many sources: 48 more loops around the square (64+ slots)
	player.global_position = DemoGeo.P(12.0, 30.0, 0.05)
	var music := world.load_clip(DemoSounds.A + "aria48.ogg", {"normalize": true})
	for k in 48:
		var s := OmgSource.new()
		s.level_db = 70.0
		s.position = DemoGeo.P(6.0 + (k % 8) * 2.0, 26.0 + (k / 8) * 2.5, 1.5)
		level_root.add_child(s)
		s.set_loop_clip(music)
	_settle(world, 1500)
	world.render(4800)
	t0 = Time.get_ticks_usec()
	world.render(96000)
	rtf = 2.0 / ((Time.get_ticks_usec() - t0) / 1e6)
	st = world.stats()
	print("stress (%d sources, %d rendering, %d paths): %.2fx realtime · sim tick %.1f ms (%s) · budget %d/%d/%d" % [
		st["sources"], st["rendering"], st["paths"], rtf, st["tick_ms"], "gpu" if st["gpu"] else "cpu", st["taps"], st["points"], st["voices"]])
	if rtf < 1.2: return "render with %d sources too slow: %.2fx" % [st["sources"], rtf]
	if st["tick_ms"] > 50.0: return "simulation tick with %d sources over budget: %.1f ms" % [st["sources"], st["tick_ms"]]
	return ""

func _listen(world: OmgWorld, player: DemoPlayer, plan: Vector3) -> float:
	player.global_position = DemoGeo.P(plan.x, plan.y, plan.z)
	_settle(world, 700)
	world.render(24000)
	return _rms(world.render(24000))

# ---- 6. the walker: doors, stairs, the bunker shaft, walls, flying
func _key(key: int, on: bool) -> void:
	var e := InputEventKey.new()
	e.keycode = key
	e.physical_keycode = key
	e.pressed = on
	Input.parse_input_event(e)

func _teleport(x: float, y: float) -> void:
	player.global_position = DemoGeo.P(x, y, 0.05)
	player.velocity = Vector3.ZERO
	await physics_frame
	await physics_frame

# walk W through plan waypoints; true when the last one is reached
func _walk(route: Array, max_s: float) -> bool:
	_key(KEY_W, true)
	var t := 0.0
	var i := 0
	while i < route.size() and t < max_s:
		var p := player.plan()
		var d: Vector2 = route[i] - Vector2(p.x, p.y)
		if d.length() < 0.3:
			i += 1
			continue
		player.set_heading(atan2(d.y, d.x))
		if OS.has_environment("OMG_WALK_DEBUG") and int(t * 60) % 20 == 0:
			print("walk %.2f s → wp %d: at %s on_floor %s vel %s" % [t, i, p, player.is_on_floor(), player.velocity])
		await physics_frame
		t += 1.0 / Engine.physics_ticks_per_second
	_key(KEY_W, false)
	return i >= route.size()

func _walks() -> String:
	await _teleport(3.0, 3.0)
	# via the living room's open hub (layout.json spot) to the north door
	if not await _walk([Vector2(3.2, 4.7), Vector2(4.0, 5.5), Vector2(4, 6.6), Vector2(4, 13), Vector2(4, 15.5)], 10.0):
		return "could not walk from the living room through the corridor into the hall: stuck at %s" % player.plan()
	await _teleport(26.5, 24.5)
	# in at the north door, down the free strip west of the sofa (layout
	# hub) to the stair foot, up the stairs, onto the upper floor
	if not await _walk([Vector2(26.5, 22.6), Vector2(26.6, 19.5), Vector2(25.9, 17.0),
			Vector2(24.85, 16.6), Vector2(24.85, 21.7), Vector2(27.5, 22.0)], 20.0) \
			or player.plan().z < 2.8:
		return "could not climb the Old House stairs: at %s" % player.plan()
	await _teleport(29.5, 7.0)
	if not await _walk([Vector2(31.0, 7), Vector2(31.7, 7), Vector2(34.5, 7), Vector2(37, 10)], 12.0) or player.plan().z > -2.5:
		return "could not go down into the bunker: at %s" % player.plan()
	await _teleport(2.0, 2.0)
	player.set_heading(PI)
	await _walk([Vector2(-5, 2)], 1.5)
	if player.plan().x < 0.0:
		return "walked through the living room's west wall: at %s" % player.plan()
	player.toggle_fly()
	_key(KEY_SPACE, true)
	for i in 60:
		await physics_frame
	_key(KEY_SPACE, false)
	player.toggle_fly()
	if player.plan().z < 2.5:
		return "fly mode did not climb: at %s" % player.plan()
	print("walks: hall, Old House upstairs, bunker, walls, flying ok")
	return ""
