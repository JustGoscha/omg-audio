# Recorded scenarios: what the engine renders in the game, as you'd hear
# it — the real host (threaded engine, output thread, Godot's audio
# thread, the window, the GPU if on), the demo's own sounds, played and
# recorded in real time (OmgWorld.start_recording: exactly the frames
# Godot plays, and every audio callback's timing).
#
#   Godot --path godot -- --record=<dir> --scenario=fly|car|door|carfar|walk|bunker --engine=v3|v4|v5|v6|v6.1|v6.2 [--tone] [--quality=0..3|-1] [--gpu]
#
# Writes <dir>/<scenario>_<engine>[_tone].wav (+ .wav.csv: the audio
# callbacks) and <dir>/<scenario>_<engine>[_tone].frames.csv (per game
# frame: time, the ears' position, the engine's stats, the scenario's
# phase), then quits.
#
#   fly   flying (8 m/s, as F-fly) from the field north through the
#         cathedral's south wall to the flute, then hovering
#   car   standing 4 m from the west road while a car passes at 100 km/h
#         (from 150 m south to 150 m north)
#   door  in the club's vestibule, 1 m outside its closed inner door: it
#         opens at 2 s and shuts again at 6 s while the club plays
#   carfar  standing in the field west of the town while a car passes
#         30 m off at 100 km/h (assess cartone's pass: x = -130 from
#         y = -50, the ears at (-100, 100))
#   walk  the car stands at (-130, 100) and the ears walk past it 30 m
#         off at 6 m/s (from (-100, 65) north: assess cartone WALK=6)
#   curtains  standing in the seminar room (--at=x,y,z: elsewhere, e.g.
#         the cathedral 8,63,0.4), the curtains rolled up for 6 s, then
#         pulled down (~3 s); the RT60 the engine fits, every second
#   bunker  down in the bunker (floor 3 m below grade), walking back and
#         forth along it at 1.5 m/s for 8 s (--view=1|2: the paths view,
#         exact or all, a screenshot a second)
#   tour  the whole level, one predictable walk (the final render of an
#         engine change): from the piano through the corridor into the
#         seminar (its door shut behind, curtains down, a clap), out to the
#         square, into the club (its inner door opened), out and into the
#         cathedral (its portal opened, curtains down, a clap), down into
#         the bunker to the radio and back up, south to the echo canyon (a
#         clap at three places), to the west road for two cars, and across
#         it into the radar dome (onto its stage, a clap). The RNG
#         seeded and every random event of the demo (owls, bells, cars,
#         the clap's and the steps' variation) off its timers, played at
#         fixed points of the walk instead: the same tour every take (the
#         engine traces in real time: its paths, not the tour, vary a
#         little). --video=fps: frames of the window too (<name>_frames/),
#         for tools/tour_video.sh. --from=n / --until=n: start at step n
#         (where the walk had got to, the doors and curtains as they were
#         left) / stop after step n — a stretch of it; --alone=name: only
#         the sources whose name starts so play, their own recordings (the
#         steps, beds and one-shots off)
#   --paths the scenario's source's exactly solved paths (direct, edge
#           bends, mirror chains) per frame to <name>.paths.csv
#   --portals the engine's portals on (an enclosed source heard through
#           openings from its side; experimental)
#   --tone  the scenario's source plays a steady 500 Hz instead of its
#           recording, alone (the other sources and the beds off):
#           whatever comes out above ~3 kHz, the engine made
class_name DemoRecord
extends Node

var demo: Node
var world: OmgWorld
var player: DemoPlayer
var out_dir := ""
var scenario := "fly"
var engine := "v6"
var tone := false
var quality := -2 # unset: the demo's (auto)
var log_paths := false # --paths: the scenario source's exact paths per frame
var shots := -1 # --view=0|1|2: the paths view off/exact/all, a screenshot a second
var at := Vector3(35.0, 10.0, -2.6) # --at=x,y,z: where the bunker scenario walks (feet)
var video_fps := 0.0 # --video=fps: the tour's frames of the window
var until_step := -1 # --until=n: the tour stops after step n
var portals := false # --portals: the engine's portals on (Config::portals)
var from_step := 0 # --from=n: the tour starts at step n
var alone := "" # --alone=name: the tour with those sources only
var paths_file: FileAccess
var frames: FileAccess
var t0 := 0
var phase := ""

func _init(d: Node, w: OmgWorld, p: DemoPlayer) -> void:
	demo = d
	world = w
	player = p
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--record="):
			out_dir = a.trim_prefix("--record=")
		elif a.begins_with("--scenario="):
			scenario = a.trim_prefix("--scenario=")
		elif a.begins_with("--engine="):
			engine = a.trim_prefix("--engine=")
		elif a == "--portals":
			portals = true
		elif a == "--tone":
			tone = true
		elif a.begins_with("--quality="):
			quality = int(a.trim_prefix("--quality="))
		elif a == "--paths":
			log_paths = true
		elif a.begins_with("--view="):
			shots = int(a.trim_prefix("--view="))
		elif a.begins_with("--video="):
			video_fps = float(a.trim_prefix("--video="))
		elif a.begins_with("--until="):
			until_step = int(a.trim_prefix("--until="))
		elif a.begins_with("--from="):
			from_step = int(a.trim_prefix("--from="))
		elif a.begins_with("--alone="):
			alone = a.trim_prefix("--alone=")
		elif a.begins_with("--at="):
			var v := a.trim_prefix("--at=").split(",")
			at = Vector3(float(v[0]), float(v[1]), float(v[2]))

func run() -> void:
	if demo.loading:
		await demo.loaded
	world.engine_v6 = engine.begins_with("v6")
	world.engine_v61 = engine == "v6.1" or engine == "v6.2"
	world.engine_v62 = engine == "v6.2"
	world.engine_portals = portals
	world.engine_v4 = engine == "v4" or engine == "v5"
	world.engine_v5 = engine == "v5"
	if quality >= -1:
		world.quality = quality
	DirAccess.make_dir_recursive_absolute(out_dir)
	var name := "%s_%s%s%s%s" % [scenario, engine, "_tone" if tone else "", ("_" + alone) if alone != "" else "", "_portals" if portals else ""]
	player.set_physics_process(false) # the script moves the ears
	player.flying = true
	var base := out_dir.path_join(name)
	match scenario:
		"fly":
			await _fly(base)
		"car":
			await _car(base)
		"door":
			await _door(base)
		"carfar":
			await _doppler(base, false)
		"walk":
			await _doppler(base, true)
		"bunker":
			await _bunker(base)
		"curtains":
			await _curtains(base)
		"tour":
			await _tour(base)
		_:
			push_error("unknown scenario " + scenario)
	get_tree().quit()

# A steady 500 Hz loop (whole cycles: the loop has no seam).
func _tone_clip() -> int:
	var sr := world.sample_rate()
	var period := int(round(sr / 500.0))
	var n := period * 200
	var s := PackedFloat32Array()
	s.resize(n)
	for i in n:
		s[i] = 0.5 * sin(TAU * float(i % period) / float(period))
	return world.add_clip(s)

# Tone mode: `keep` plays the tone, every other source and the beds off.
func _alone(keep: String) -> int:
	var id := _tone_clip()
	for k in demo.sounds.sources:
		var src: OmgSource = demo.sounds.sources[k]
		if k == keep:
			src.set_loop_clip(id)
		else:
			src.stop()
	# the demo's own one-shots too: owls and bells on their timers, cars
	# (the car scenario sends its own)
	var snd = demo.sounds
	snd.wild = {"owl1": 1e9, "owl2": 1e9, "bells": 1e9}
	snd.next_car = 1e9
	world.ambience_db = -200.0
	world.rain_db = -200.0
	# the ear off: its tinnitus (5.4 kHz) after loud sound is the ear
	# model's, not the engine's — the tone runs measure the engine
	world.set_ear(false)
	return id

func _place(x: float, y: float, z: float) -> void:
	player.global_position = DemoGeo.P(x, y, z)

func _start(base: String) -> bool:
	if log_paths:
		world.debug_paths = true
		paths_file = FileAccess.open(base + ".paths.csv", FileAccess.WRITE)
		paths_file.store_line("t_ms,kind,key,energy_db,points...")
	frames = FileAccess.open(base + ".frames.csv", FileAccess.WRITE)
	frames.store_line("t_ms,x,y,tick_ms,load,taps,rendering,paths,late,ticks,level,direct,edges,edges2,early,late_rays,chains,fold,portals,phase")
	if not world.start_recording(base + ".wav"):
		push_error("recording failed: " + base)
		return false
	t0 = Time.get_ticks_usec()
	return true

# The exact paths of `src` now: time, kind (0 direct, 1 bend, 2 chain),
# key, mid-band energy (dB), and each path's points after the ears.
func _paths_log(t_ms: float, src: Node) -> void:
	if paths_file == null:
		return
	for p in world.paths():
		if p["source"] != src or int(p["kind"]) > 2:
			continue
		var pts: PackedVector3Array = p["points"]
		var tail := ""
		for i in range(1, pts.size() - 1):
			var q := Vector3(pts[i].x, -pts[i].z, pts[i].y) # plan: x, y north, z up
			tail += ",%.2f,%.2f,%.2f" % [q.x, q.y, q.z]
		paths_file.store_line("%.1f,%d,%d,%.1f%s" % [t_ms, int(p["kind"]), int(p["key"]), 10.0 * log(maxf(float(p["energy"]), 1e-30)) / log(10.0), tail])

func _stop(base: String) -> void:
	var lost := world.stop_recording()
	frames.close()
	if paths_file:
		paths_file.close()
	print("recorded %s.wav (%d frames lost)" % [base, lost])

func _now() -> float:
	return (Time.get_ticks_usec() - t0) / 1e6

func _frame_log(t_ms: float, x: float, y: float) -> void:
	var s: Dictionary = world.stats()
	var parts: PackedFloat32Array = s.get("tick_parts", PackedFloat32Array())
	var p := ""
	for v in parts:
		p += ",%.2f" % v
	frames.store_line("%.1f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d%s,%s" % [t_ms, x, y, s.get("tick_ms", 0.0), s.get("load", 0.0),
		s.get("taps", 0), s.get("rendering", 0), s.get("paths", 0), s.get("late", 0), s.get("ticks", 0), world.effective_quality(), p, phase])

func _fly(base: String) -> void:
	if tone:
		_alone("flute")
	var x := 4.0
	var y0 := 40.0
	var y1 := 62.0
	var speed := 8.0
	_place(x, y0, 0.4)
	player.set_heading(PI / 2.0)
	# settle where the flight starts (traces, kernels, the ear)
	await get_tree().create_timer(4.0).timeout
	if not _start(base):
		return
	var t := 0.0
	# 1 s still, the flight, 3 s hovering by the flute
	var fly_s := (y1 - y0) / speed
	while t < 1.0 + fly_s + 3.0:
		# the ears move on physics steps, as the player does
		await get_tree().physics_frame
		t = _now()
		var y := clampf(y0 + (t - 1.0) * speed, y0, y1)
		_place(x, y, 0.4)
		phase = "outside" if y < 51.0 else ("crossing" if y < 53.5 else "inside")
		_frame_log(t * 1000.0, x, y)
	_stop(base)

func _car(base: String) -> void:
	var lane: float = DemoSounds.CAR_LANES[1]
	var x := lane + 4.0
	_place(x, 0.0, 0.4)
	player.set_heading(PI) # facing the road (west)
	var snd = demo.sounds
	snd.next_car = 1e9 # no cars at random
	var tone_id := _alone("") if tone else -1
	await get_tree().create_timer(3.0).timeout
	snd._spawn_car()
	var c: Dictionary = snd.cars.back()
	var v := 27.8
	var north: bool = c["north"]
	c["x"] = lane if north else DemoSounds.CAR_LANES[0]
	c["y"] = -150.0 if north else 150.0
	c["vy"] = v if north else -v
	if tone:
		c["motor"] = tone_id
	# its first paths and the motor's start
	await get_tree().create_timer(1.0).timeout
	if not _start(base):
		return
	var t := 0.0
	while t < 300.0 / v:
		await get_tree().process_frame
		t = _now()
		var along: float = c["y"] if north else -float(c["y"])
		phase = "approach" if along < -15.0 else ("pass" if along < 15.0 else "away")
		_frame_log(t * 1000.0, x, 0.0)
		_paths_log(t * 1000.0, snd.cars_src[c["slot"]])
	_stop(base)

func _door(base: String) -> void:
	if tone:
		_alone("club0")
	var door: Dictionary = demo.level.door_nodes[4] # the club's inner door
	door["target"] = 0.0
	_place(21.0, 31.0, 0.4)
	player.set_heading(0.0) # facing the club (east)
	await get_tree().create_timer(4.0).timeout
	if not _start(base):
		return
	var t := 0.0
	while t < 9.0:
		await get_tree().process_frame
		t = _now()
		if t >= 2.0 and t < 6.0:
			door["target"] = 1.0
		elif t >= 6.0:
			door["target"] = 0.0
		phase = "closed" if t < 2.0 else ("opening" if t < 3.0 else ("open" if t < 6.0 else ("shutting" if t < 7.0 else "shut")))
		_frame_log(t * 1000.0, 21.0, 31.0)
	_stop(base)

# Doppler in the field (see carfar, walk): a car 30 m off at 100 km/h past
# the still ears, or standing while the ears walk past at 6 m/s.
func _doppler(base: String, walk: bool) -> void:
	var lis := Vector2(-100.0, 65.0 if walk else 100.0)
	_place(lis.x, lis.y, 0.4)
	player.set_heading(PI) # facing the road (west)
	var snd = demo.sounds
	snd.next_car = 1e9 # no cars at random
	var tone_id := _alone("") if tone else -1
	await get_tree().create_timer(3.0).timeout
	snd._spawn_car()
	var c: Dictionary = snd.cars.back()
	var v := 27.78
	var walk_v := 6.0
	c["x"] = -130.0
	c["y"] = 100.0 if walk else -50.0
	c["vy"] = 0.0 if walk else v
	if tone:
		c["motor"] = tone_id
	# its first paths and the motor's start
	await get_tree().create_timer(1.0).timeout
	if not _start(base):
		return
	var t := 0.0
	while t < 300.0 / v + 1.0:
		# the ears move on physics steps, as the player does (the car too)
		await get_tree().physics_frame
		t = _now()
		var y := lis.y + (walk_v * t if walk else 0.0)
		_place(lis.x, y, 0.4)
		phase = "walk" if walk else "pass"
		_frame_log(t * 1000.0, lis.x, y)
		_paths_log(t * 1000.0, snd.cars_src[c["slot"]])
	_stop(base)

# Down in the bunker: walking back and forth along it (x 35..39, y 10).
func _bunker(base: String) -> void:
	_place(at.x, at.y, at.z)
	player.set_heading(0.0)
	await get_tree().create_timer(3.0).timeout
	if not _start(base):
		return
	if shots >= 0:
		demo.paths_view.set_mode(shots)
	var t := 0.0
	var shot := 0
	while t < 8.0:
		await get_tree().physics_frame
		t = _now()
		if shots >= 0 and t >= shot:
			demo.get_viewport().get_texture().get_image().save_png("%s_%d.png" % [base, shot])
			print("t %.0f s: %.0f fps, %d draw calls" % [t, Engine.get_frames_per_second(), Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)])
			shot += 1
		var d := fmod(t * 1.5, 8.0)
		var x := at.x + (d if d < 4.0 else 8.0 - d)
		_place(x, at.y, at.z)
		phase = "bunker"
		_frame_log(t * 1000.0, x, at.y)
	_stop(base)

# The curtains of the room the ears stand in: up 6 s, then pulled down.
func _curtains(base: String) -> void:
	var spot := at if at != Vector3(35.0, 10.0, -2.6) else Vector3(7.0, 19.0, 0.4)
	_place(spot.x, spot.y, spot.z)
	player.set_heading(PI) # facing west: the back wall and both flanks
	await get_tree().create_timer(3.0).timeout
	if not _start(base):
		return
	var t := 0.0
	var sec := 0
	var pulled := false
	while t < 16.0:
		await get_tree().physics_frame
		t = _now()
		if not pulled and t >= 6.0:
			pulled = true
			var best := 0
			for i in demo.level.curtain_sets.size():
				if (demo.level.curtain_sets[i]["cord"] - Vector2(spot.x, spot.y)).length() < (demo.level.curtain_sets[best]["cord"] - Vector2(spot.x, spot.y)).length():
					best = i
			demo.level.curtain_sets[best]["target"] = 0.0
		if t >= sec:
			var st: Dictionary = world.stats()
			print("curtains t %d s: %s rt60 %.2f s, rung %d" % [sec, "down" if pulled else "up", st.get("rt60", 0.0), world.effective_quality()])
			if shots >= 0 and (sec == 5 or sec == 12):
				demo.get_viewport().get_texture().get_image().save_png("%s_%d.png" % [base, sec])
			sec += 1
		phase = "down" if pulled else "up"
		_frame_log(t * 1000.0, spot.x, spot.y)
	_stop(base)


# The whole level, one predictable walk: legs walked at a steady pace,
# actions at fixed points of it (see the header).
const TOUR_WALK := 2.5 # m/s
const TOUR_STEP := 0.75 # m between footsteps
func _tour(base: String) -> void:
	seed(20261030)
	var snd = demo.sounds
	var lv = demo.level
	snd.wild = {"owl1": 1e9, "owl2": 1e9, "bells": 1e9}
	snd.next_car = 1e9
	# [kind, args…, phase]: "go" x, y, z (feet) — walk there (straight:
	# the legs keep clear of solids — through the kiosk the ears heard
	# nothing for half a second, rightly); "wait" s;
	# "door" i, open; "curtains"; "clap"; "bells"; "owl" source; "car"
	# north (from CAR_SPAWN_M off — where cars start: heard by no one — at
	# 100 km/h)
	# the feet on the floors, the ears 1.6 m over them (the player's EYE),
	# as a person's: 0.4 m up they rode at 2.0 m, the height of the
	# doorways' lintels — walking through a door they grazed the lintel,
	# which the engine takes for passing through a wall (every average
	# dropped: the club's bass fell ~10 dB for half a second at its door)
	var g := 0.0 # at grade
	var b := -3.0 # the bunker's floor
	var plan := [
		["wait", 4.0, "piano"], ["bells", "piano"], ["wait", 3.0, "piano"],
		["go", 4.0, 6.0, g, "corridor"], ["go", 4.0, 13.5, g, "corridor"], ["go", 5.5, 17.0, g, "seminar"],
		["door", 1, false, "seminar"], ["curtains", "seminar"], ["wait", 4.0, "seminar"], ["clap", "seminar"], ["wait", 3.0, "seminar"],
		["go", 7.0, 23.0, g, "seminar"], ["go", 7.0, 25.5, g, "square"], ["go", 12.0, 30.0, g, "square"], ["wait", 2.0, "square"],
		["go", 19.0, 31.0, g, "entrance"], ["go", 21.0, 31.0, g, "entrance"], ["door", 4, true, "entrance"], ["wait", 2.0, "entrance"],
		["go", 25.0, 32.0, g, "club"], ["go", 27.0, 33.5, g, "club"], ["wait", 5.0, "club"],
		["go", 21.0, 31.0, g, "entrance"], ["go", 18.5, 31.0, g, "square"], ["go", 13.5, 32.0, g, "square"], ["go", 12.0, 37.0, g, "square"], ["go", 8.0, 46.0, g, "path"],
		["go", 8.0, 50.8, g, "portal"], ["door", 6, true, "portal"], ["wait", 2.0, "portal"],
		["go", 8.0, 56.0, g, "cathedral"], ["go", 8.0, 62.0, g, "cathedral"], ["owl", "owl·cathedral", "cathedral"],
		["curtains", "cathedral"], ["wait", 4.0, "cathedral"], ["clap", "cathedral"], ["wait", 5.0, "cathedral"],
		["go", 8.0, 53.0, g, "portal"], ["go", 8.0, 46.0, g, "path"], ["go", 13.0, 35.0, g, "square"], ["go", 15.0, 25.0, g, "lane"],
		["go", 22.0, 20.0, g, "lane"], ["go", 22.0, 8.5, g, "bunker path"], ["go", 30.5, 7.0, g, "bunker path"],
		["go", 32.7, 7.0, g, "blockhouse"], ["go", 33.6, 7.0, b, "stair"], ["go", 35.0, 7.4, b, "bunker"], ["go", 36.5, 9.0, b, "bunker"],
		["wait", 5.0, "bunker"],
		["go", 35.0, 7.4, b, "bunker"], ["go", 33.6, 7.0, b, "stair"], ["go", 32.7, 7.0, g, "blockhouse"], ["go", 30.5, 7.0, g, "bunker path"],
		["go", 26.0, 7.0, g, "bunker path"], ["wait", 3.0, "bunker path"],
		["go", 20.0, -40.0, g, "south"], ["go", 14.0, -80.0, g, "canyon"], ["go", 14.0, -95.0, g, "canyon"], ["clap", "canyon"], ["wait", 4.0, "canyon"],
		["go", 2.0, -95.0, g, "canyon"], ["clap", "canyon"], ["wait", 4.0, "canyon"],
		["go", 27.0, -95.0, g, "canyon"], ["clap", "canyon"], ["wait", 4.0, "canyon"],
		["go", 14.0, -65.0, g, "south"], ["go", -13.0, -45.0, g, "road"], ["wait", 1.0, "road"],
		["car", true, "road"], ["wait", 5.0, "road"], ["car", false, "road"], ["wait", 40.0, "road"],
		["go", -14.0, 30.0, g, "road"], ["go", -30.0, 30.0, g, "dome"], ["go", -37.0, 30.0, g, "dome"], ["wait", 3.0, "dome"],
		["go", -42.0, 31.0, 0.8, "stage"], ["wait", 2.0, "stage"], ["clap", "stage"], ["wait", 5.0, "stage"],
	]
	if alone != "":
		for k in snd.sources:
			if not String(k).begins_with(alone):
				(snd.sources[k] as OmgSource).stop()
		world.ambience_db = -200.0
		world.rain_db = -200.0
	var pos := Vector3(3.5, 4.0, g)
	# --from=n: where the walk had got to, the doors and curtains as left
	for i in mini(from_step, plan.size()):
		var st: Array = plan[i]
		match st[0]:
			"go":
				var to := Vector3(st[1], st[2], st[3])
				if Vector2(to.x - pos.x, to.y - pos.y).length() > 0.01:
					player.set_heading(atan2(to.y - pos.y, to.x - pos.x))
				pos = to
			"door":
				lv.door_nodes[int(st[1])]["target"] = 1.0 if st[2] else 0.0
			"curtains":
				var best := 0
				for k in lv.curtain_sets.size():
					if (lv.curtain_sets[k]["cord"] - Vector2(pos.x, pos.y)).length() < (lv.curtain_sets[best]["cord"] - Vector2(pos.x, pos.y)).length():
						best = k
				lv.curtain_sets[best]["target"] = 0.0
	_place(pos.x, pos.y, pos.z)
	if from_step == 0:
		player.set_heading(PI / 2.0)
	await get_tree().create_timer(4.0).timeout
	if not _start(base):
		return
	var frames_dir := base + "_frames"
	if video_fps > 0.0:
		DirAccess.make_dir_recursive_absolute(frames_dir)
	var slot := 0
	var t := _now()
	var walked := 0.0
	var log := FileAccess.open(base + ".tour.csv", FileAccess.WRITE)
	log.store_line("t_s,step,kind,phase")
	for i in plan.size():
		if until_step >= 0 and i > until_step:
			break
		if i < from_step:
			continue
		var st: Array = plan[i]
		var kind: String = st[0]
		phase = st[st.size() - 1]
		log.store_line("%.3f,%d,%s,%s" % [_now(), i, kind, phase])
		match kind:
			"go":
				var to := Vector3(st[1], st[2], st[3])
				var flat := Vector2(to.x - pos.x, to.y - pos.y)
				if flat.length() > 0.01:
					player.set_heading(atan2(flat.y, flat.x))
				var from := pos
				var dist := (to - from).length()
				var t0 := _now()
				var done := false
				while not done:
					await get_tree().physics_frame
					t = _now()
					var u := clampf((t - t0) * TOUR_WALK / maxf(dist, 1e-3), 0.0, 1.0)
					var p := from.lerp(to, u)
					walked += (p - pos).length()
					pos = p
					_place(pos.x, pos.y, pos.z)
					if walked >= TOUR_STEP:
						walked -= TOUR_STEP
						if alone == "":
							snd._step(true)
					slot = _tour_frame(frames_dir, slot, t)
					_frame_log(t * 1000.0, pos.x, pos.y)
					done = u >= 1.0
			"wait":
				var until: float = _now() + float(st[1])
				while _now() < until:
					await get_tree().physics_frame
					t = _now()
					slot = _tour_frame(frames_dir, slot, t)
					_frame_log(t * 1000.0, pos.x, pos.y)
			"door":
				lv.door_nodes[int(st[1])]["target"] = 1.0 if st[2] else 0.0
			"curtains":
				var best := 0
				var here := Vector2(pos.x, pos.y)
				for k in lv.curtain_sets.size():
					if (lv.curtain_sets[k]["cord"] - here).length() < (lv.curtain_sets[best]["cord"] - here).length():
						best = k
				lv.curtain_sets[best]["target"] = 0.0
			"clap":
				if alone == "":
					snd.hands.play_clip(snd.clip["clap"], 8.0, 1.0)
			"bells":
				if alone == "":
					snd.sources["bells"].play_clip(snd.clip["bell_big"], 1.0, 1.0)
			"owl":
				if alone == "":
					snd.sources[st[1]].play_clip(snd.clip["owl0"], 1.0, 1.0)
			"car":
				snd._spawn_car()
				var c: Dictionary = snd.cars.back()
				var north: bool = st[1]
				c["x"] = DemoSounds.CAR_LANES[1] if north else DemoSounds.CAR_LANES[0]
				c["y"] = pos.y + (-1.0 if north else 1.0) * DemoSounds.CAR_SPAWN_M
				c["vy"] = 27.8 if north else -27.8
	log.close()
	_stop(base)

# The window's frame for every 1/fps of the tour passed (missed slots
# repeat the frame: the video keeps the audio's time).
func _tour_frame(dir: String, slot: int, t: float) -> int:
	if video_fps <= 0.0:
		return slot
	if float(slot) / video_fps > t:
		return slot
	var img := demo.get_viewport().get_texture().get_image()
	img.resize(960, int(960.0 * img.get_height() / img.get_width()))
	var first := slot
	while float(slot) / video_fps <= t:
		var path := "%s/%06d.jpg" % [dir, slot]
		if slot == first:
			img.save_jpg(path, 0.85)
		else:
			DirAccess.copy_absolute("%s/%06d.jpg" % [dir, first], path)
		slot += 1
	return slot
