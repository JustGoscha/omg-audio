# Headless smoke test for the GDExtension:
#   /Applications/Godot.app/Contents/MacOS/Godot --headless --path godot -s test.gd
# Exits 0 on success, 1 on failure. (Not `assert`: a failed assert in a
# headless -s script halts _initialize and Godot then idles forever.)
extends SceneTree

func _initialize() -> void:
	var err := _run()
	if err.is_empty():
		print("GODOT SMOKE TEST PASSED")
		quit(0)
	else:
		printerr("GODOT SMOKE TEST FAILED: ", err)
		quit(1)

func _run() -> String:
	var e := OmgEngine.new()
	e.setup(48000.0)

	# 220 Hz tone into the living-room source slot.
	var n := 48000
	var buf := PackedFloat32Array()
	buf.resize(n)
	for k in n:
		buf[k] = sin(TAU * 220.0 * k / 48000.0) * 0.4
	e.set_source_samples(0, buf)
	e.set_listener(3.0, 3.0, 0.0)

	OS.delay_msec(200) # let the 20 Hz sim thread publish params

	var out: PackedVector2Array = e.render(9600) # 200 ms
	var peak := 0.0
	for v in out:
		peak = max(peak, max(abs(v.x), abs(v.y)))
	print("room=", e.listener_room(), " frames=", out.size(), " peak=", peak)
	if out.size() != 9600: return "wrong frame count"
	if peak <= 0.01: return "engine is silent"
	if peak >= 1.0: return "engine is clipping"

	# Turning the head must change the ear signals (binaural sanity).
	e.set_head(PI / 2.0, 0.0, 0.0)
	var out2: PackedVector2Array = e.render(4800)
	var diff := 0.0
	for v in out2:
		diff = max(diff, abs(v.x - v.y))
	if diff <= 1e-4: return "no interaural difference after head turn"

	# Real-time budget: the demo feeds two sources and pulls audio on the
	# game thread, so one second of output must render in under a second.
	e.set_source_samples(2, buf)
	var t0 := Time.get_ticks_usec()
	e.render(48000)
	var speed := 1e6 / float(Time.get_ticks_usec() - t0)
	print("render speed (2 sources): ", snappedf(speed, 0.01), "x realtime")
	if speed <= 1.0: return "render is slower than real time"
	return ""
