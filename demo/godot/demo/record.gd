# Recorded scenarios: what the engine renders in the game, as you'd hear
# it — the real host (threaded engine, output thread, Godot's audio
# thread, the window, the GPU if on), the demo's own sounds, played and
# recorded in real time (OmgWorld.start_recording: exactly the frames
# Godot plays, and every audio callback's timing).
#
#   Godot --path godot -- --record=<dir> --scenario=fly|car|door --engine=v6 [--tone] [--quality=0..3|-1] [--gpu]
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
		elif a == "--tone":
			tone = true
		elif a.begins_with("--quality="):
			quality = int(a.trim_prefix("--quality="))

func run() -> void:
	if demo.loading:
		await demo.loaded
	world.engine_v6 = engine == "v6"
	world.engine_v4 = engine == "v4" or engine == "v5"
	world.engine_v5 = engine == "v5"
	if quality >= -1:
		world.quality = quality
	DirAccess.make_dir_recursive_absolute(out_dir)
	var name := "%s_%s%s" % [scenario, engine, "_tone" if tone else ""]
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
	frames = FileAccess.open(base + ".frames.csv", FileAccess.WRITE)
	frames.store_line("t_ms,x,y,tick_ms,load,taps,rendering,paths,late,ticks,level,direct,edges,edges2,early,late_rays,chains,fold,phase")
	if not world.start_recording(base + ".wav"):
		push_error("recording failed: " + base)
		return false
	t0 = Time.get_ticks_usec()
	return true

func _stop(base: String) -> void:
	var lost := world.stop_recording()
	frames.close()
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
		await get_tree().process_frame
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
