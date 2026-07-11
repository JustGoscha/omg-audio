# Headless smoke test for the GDExtension:
#   /Applications/Godot.app/Contents/MacOS/Godot --headless --path godot -s test.gd
extends SceneTree

func _initialize() -> void:
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
	assert(out.size() == 9600, "wrong frame count")
	assert(peak > 0.01, "engine is silent")
	assert(peak < 1.0, "engine is clipping")

	# Turning the head must change the ear signals (binaural sanity).
	e.set_head_yaw(PI / 2.0)
	var out2: PackedVector2Array = e.render(4800)
	var diff := 0.0
	for v in out2:
		diff = max(diff, abs(v.x - v.y))
	assert(diff > 1e-4, "no interaural difference after head turn")

	print("GODOT SMOKE TEST PASSED")
	quit(0)
