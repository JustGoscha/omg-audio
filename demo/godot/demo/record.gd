# Recorded scenarios: what the engine renders in the game, as you'd hear
# it — the real host (threaded engine, output thread, Godot's audio
# thread, the window, the GPU if on), the demo's own sounds, played and
# recorded in real time (OmgWorld.start_recording: exactly the frames
# Godot plays, and every audio callback's timing).
#
#   Godot --path godot -- --record=<dir> --scenario=fly --engine=v6 [--tone] [--gpu]
#
# Writes <dir>/<scenario>_<engine>[_tone].wav (+ .wav.csv: the audio
# callbacks) and <dir>/<scenario>_<engine>[_tone].frames.csv (per game
# frame: time, the ears' position, the engine's stats), then quits.
#
#   fly   flying (8 m/s, as F-fly) from the field north through the
#         cathedral's south wall to the flute, then hovering
#   --tone  the flute plays a steady 500 Hz instead of its recording, alone
#           (the other sources and the beds off): whatever comes out
#           above ~3 kHz, the engine made
class_name DemoRecord
extends Node

var demo: Node
var world: OmgWorld
var player: DemoPlayer
var out_dir := ""
var scenario := "fly"
var engine := "v6"
var tone := false
var frames: FileAccess
var t0 := 0

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

func run() -> void:
	if demo.loading:
		await demo.loaded
	world.engine_v6 = engine == "v6"
	world.engine_v4 = engine == "v4" or engine == "v5"
	world.engine_v5 = engine == "v5"
	if tone:
		_tone("flute")
		# alone: the other sources and the beds off, so whatever the
		# output holds above ~3 kHz is the engine's making
		for k in demo.sounds.sources:
			if k != "flute":
				(demo.sounds.sources[k] as OmgSource).stop()
		world.ambience_db = -200.0
		world.rain_db = -200.0
	DirAccess.make_dir_recursive_absolute(out_dir)
	var name := "%s_%s%s" % [scenario, engine, "_tone" if tone else ""]
	player.set_physics_process(false) # the script moves the ears
	player.flying = true
	match scenario:
		"fly":
			await _fly(out_dir.path_join(name))
		_:
			push_error("unknown scenario " + scenario)
	get_tree().quit()

# The flute loops a steady 500 Hz (whole cycles: the loop has no seam).
func _tone(source: String) -> void:
	var sr := world.sample_rate()
	var period := int(round(sr / 500.0))
	var n := period * 200
	var s := PackedFloat32Array()
	s.resize(n)
	for i in n:
		s[i] = 0.5 * sin(TAU * float(i % period) / float(period))
	var id := world.add_clip(s)
	var src: OmgSource = demo.sounds.sources.get(source)
	if src:
		src.set_loop_clip(id)

func _place(x: float, y: float, z: float) -> void:
	player.global_position = DemoGeo.P(x, y, z)

func _frame_log(t_ms: float, x: float, y: float) -> void:
	var s: Dictionary = world.stats()
	frames.store_line("%.1f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d" % [t_ms, x, y, s.get("tick_ms", 0.0), s.get("load", 0.0),
		s.get("taps", 0), s.get("rendering", 0), s.get("paths", 0), s.get("late", 0), s.get("ticks", 0)])

func _fly(base: String) -> void:
	var x := 4.0
	var y0 := 40.0
	var y1 := 62.0
	var speed := 8.0
	_place(x, y0, 0.4)
	player.set_heading(PI / 2.0)
	# settle where the flight starts (traces, kernels, the ear)
	await get_tree().create_timer(4.0).timeout
	frames = FileAccess.open(base + ".frames.csv", FileAccess.WRITE)
	frames.store_line("t_ms,x,y,tick_ms,load,taps,rendering,paths,late,ticks")
	if not world.start_recording(base + ".wav"):
		push_error("recording failed: " + base)
		return
	t0 = Time.get_ticks_usec()
	var t := 0.0
	# 1 s still, the flight, 3 s hovering by the flute
	var fly_s := (y1 - y0) / speed
	while t < 1.0 + fly_s + 3.0:
		await get_tree().process_frame
		t = (Time.get_ticks_usec() - t0) / 1e6
		var y := clampf(y0 + (t - 1.0) * speed, y0, y1)
		_place(x, y, 0.4)
		_frame_log(t * 1000.0, x, y)
	var lost := world.stop_recording()
	frames.close()
	print("recorded %s.wav (%d frames lost)" % [base, lost])
