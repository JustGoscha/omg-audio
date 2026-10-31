# The flight recorder's game side, running while the spectrogram is on:
# the last minute of where the ears were, every source (where, how loud),
# the engine's state, its strongest traced paths and what the player did —
# beside the audio thread's own minute of output (OmgWorld.flight_*), all
# stamped with that minute's frame clock. M marks a moment: the game
# pauses and goes quiet, the minute's spectrogram opens across the screen;
# drag over it to pick a stretch (or click: 4 s around it), say what you
# heard, and Save writes it all to a folder of its own under captures/
# (README.txt there says what each file holds). M or Esc plays on.
class_name DemoCapture
extends CanvasLayer

const KEEP_S := 60.0
const SRC_HZ := 10.0
const PATH_HZ := 4.0
const PATHS_MAX := 48
const MARGIN_S := 2.0
const BANDS := 160

var demo: Node
var world: OmgWorld
var recording := false
var open := false
var frames := []    # [frame, x, y, z, yaw°, pitch°, room, tick_ms, load, late, paths, rendering, fps]
var src_log := []   # [frame, name, x, y, z, active, peak dB, rms dB]
var path_log := []  # [frame, JSON of [{source, kind, key, energy, points}]]
var events := []    # [frame, what, detail]
var src_t := 0.0
var path_t := 0.0
var last_quality := ""
var start_fr := 0
var end_fr := 0
var rate := 48000
var sel := Vector2(-1, -1) # the picked stretch, seconds into the minute
var dragging := false
var minute_img: Image
var room_marks := [] # [seconds into the minute, room] where it changed

var dim: ColorRect
var spec: TextureRect
var over: Control
var note: LineEdit
var status: Label
var toast: Label
var toast_t := 0.0

func _init(d: Node, w: OmgWorld) -> void:
	demo = d
	world = w
	layer = 30
	process_mode = Node.PROCESS_MODE_ALWAYS

func _ready() -> void:
	dim = ColorRect.new()
	dim.color = Color(0.02, 0.02, 0.04, 0.92)
	dim.set_anchors_preset(Control.PRESET_FULL_RECT)
	dim.visible = false
	add_child(dim)
	var title := Label.new()
	title.text = "MARKED — drag over the spectrogram to pick a stretch (click: 4 s around it) · say what you heard · Enter / Save · M or Esc: play on"
	title.position = Vector2(32, 14)
	title.add_theme_font_size_override("font_size", 18)
	title.add_theme_color_override("font_color", Color(1.0, 0.86, 0.6))
	dim.add_child(title)
	spec = TextureRect.new()
	spec.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	spec.stretch_mode = TextureRect.STRETCH_SCALE
	spec.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	spec.mouse_filter = Control.MOUSE_FILTER_STOP
	spec.gui_input.connect(_spec_input)
	dim.add_child(spec)
	over = Control.new()
	over.mouse_filter = Control.MOUSE_FILTER_IGNORE
	over.draw.connect(_draw_over)
	dim.add_child(over)
	note = LineEdit.new()
	note.placeholder_text = "what did you hear? (e.g. a burst of hiss as I walked out of the club door)"
	note.add_theme_font_size_override("font_size", 18)
	note.text_submitted.connect(func(_t: String) -> void: save())
	dim.add_child(note)
	var save_b := Button.new()
	save_b.text = "Save"
	save_b.pressed.connect(save)
	save_b.name = "SaveButton"
	dim.add_child(save_b)
	var resume_b := Button.new()
	resume_b.text = "Play on"
	resume_b.pressed.connect(resume)
	resume_b.name = "ResumeButton"
	dim.add_child(resume_b)
	status = Label.new()
	status.add_theme_font_size_override("font_size", 16)
	status.add_theme_color_override("font_color", Color(0.75, 0.9, 0.8))
	dim.add_child(status)
	toast = Label.new()
	toast.position = Vector2(32, 32)
	toast.add_theme_font_size_override("font_size", 20)
	toast.add_theme_color_override("font_color", Color(1.0, 0.86, 0.6))
	toast.visible = false
	add_child(toast)
	get_viewport().size_changed.connect(_layout)
	_layout()

func _layout() -> void:
	var vs := get_viewport().get_visible_rect().size
	var w := vs.x - 64.0
	var h := minf(vs.y * 0.62, 640.0)
	spec.position = Vector2(32, 52)
	spec.size = Vector2(w, h)
	over.position = spec.position
	over.size = Vector2(w, h + 44)
	var y := spec.position.y + h + 54
	note.position = Vector2(32, y)
	note.size = Vector2(w - 260, 40)
	var sb: Button = dim.get_node("SaveButton")
	sb.position = Vector2(32 + w - 250, y)
	sb.size = Vector2(110, 40)
	var rb: Button = dim.get_node("ResumeButton")
	rb.position = Vector2(32 + w - 130, y)
	rb.size = Vector2(130, 40)
	status.position = Vector2(32, y + 50)
	status.size = Vector2(w, 60)

# ------------------------------------------------------------- recording

func _process(dt: float) -> void:
	if toast.visible:
		toast_t -= dt
		toast.visible = toast_t > 0.0
	if open:
		return
	var want: bool = demo.debug.spectrum_on
	if want != recording:
		recording = want
		if recording:
			world.flight_start()
			world.debug_paths = true
			event("recording", "spectrogram on: the minute is kept")
		else:
			world.flight_freeze()
			world.debug_paths = demo.paths_view.mode != DemoPathsView.OFF
			frames.clear()
			src_log.clear()
			path_log.clear()
			events.clear()
	if not recording:
		return
	world.debug_paths = true
	var fr: int = world.flight_frames()
	var r: int = maxi(1, world.flight_rate())
	var cut: int = fr - int((KEEP_S + MARGIN_S) * r)
	for rows in [frames, src_log, path_log, events]:
		while not rows.is_empty() and rows[0][0] < cut:
			rows.pop_front()
	var cam: Camera3D = demo.player.cam
	var at := _plan(cam.global_position)
	var fwd := -cam.global_transform.basis.z
	var room: String = DemoLevel.room_at(at.x, at.y, at.z - 1.1)[0]
	var st := world.stats()
	frames.append([fr, snappedf(at.x, 0.001), snappedf(at.y, 0.001), snappedf(at.z, 0.001),
		snappedf(rad_to_deg(atan2(-fwd.z, fwd.x)), 0.1), snappedf(rad_to_deg(asin(clampf(fwd.y, -1.0, 1.0))), 0.1), room,
		snappedf(st.get("tick_ms", 0.0), 0.01), snappedf(st.get("load", 0.0), 0.001), st.get("late", 0), st.get("paths", 0),
		st.get("rendering", 0), Engine.get_frames_per_second()])
	var q: String = world.quality_label()
	if q != last_quality:
		event("quality", q)
		last_quality = q
	src_t += dt
	if src_t >= 1.0 / SRC_HZ:
		src_t = 0.0
		var m: Dictionary = world.meters()
		for s in get_tree().get_nodes_in_group("omg_source"):
			var o := s as OmgSource
			var p := _plan(o.global_position)
			var lv: Vector2 = m.get(o.source_id(), Vector2.ZERO)
			src_log.append([fr, String(o.name), snappedf(p.x, 0.01), snappedf(p.y, 0.01), snappedf(p.z, 0.01), o.active and not o.muted, _db(lv.x), _db(lv.y)])
	path_t += dt
	if path_t >= 1.0 / PATH_HZ:
		path_t = 0.0
		var snap := []
		for p in world.paths().slice(0, PATHS_MAX):
			var pts := []
			for q3 in p["points"]:
				var pp := _plan(q3)
				pts.append([snappedf(pp.x, 0.01), snappedf(pp.y, 0.01), snappedf(pp.z, 0.01)])
			var src = p["source"]
			snap.append({"source": String(src.name) if src != null else "", "kind": p["kind"], "key": p["key"], "energy": snappedf(p["energy"], 1e-6), "points": pts})
		path_log.append([fr, JSON.stringify(snap)])

# Something the player did (a key, a door, the quality), on the minute.
func event(what: String, detail := "") -> void:
	if recording and not open:
		events.append([world.flight_frames(), what, detail])

static func _plan(g: Vector3) -> Vector3:
	return Vector3(g.x, -g.z, g.y)

static func _db(a: float) -> float:
	return snappedf(linear_to_db(maxf(a, 1e-7)), 0.1)

# ------------------------------------------------------------------ mark

func mark() -> void:
	if open:
		return
	if not recording:
		_toast("turn the spectrogram on (Esc › debug › spectrum) — the minute is kept while it shows")
		return
	world.flight_freeze()
	end_fr = world.flight_frames()
	rate = maxi(1, world.flight_rate())
	start_fr = maxi(0, end_fr - int(KEEP_S * rate))
	open = true
	get_tree().paused = true
	AudioServer.set_bus_mute(AudioServer.get_bus_index("Master"), true)
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	_layout()
	var w := int(spec.size.x)
	var bytes: PackedByteArray = world.flight_spectrogram(start_fr, end_fr, w, BANDS)
	minute_img = Image.create_from_data(w, 2 * BANDS + 4, false, Image.FORMAT_RGBA8, bytes)
	spec.texture = ImageTexture.create_from_image(minute_img)
	room_marks.clear()
	var last := ""
	for f in frames:
		if f[0] >= start_fr and f[6] != last:
			last = f[6]
			room_marks.append([_t(f[0]), last])
	sel = Vector2(-1, -1)
	note.text = ""
	status.text = "the last %.0f s · %d frames of the ears, %d source readings, %d path snapshots, %d events" % [
		float(end_fr - start_fr) / rate, frames.size(), src_log.size(), path_log.size(), events.size()]
	dim.visible = true
	note.grab_focus()
	over.queue_redraw()

func resume() -> void:
	if not open:
		return
	open = false
	dim.visible = false
	note.release_focus()
	AudioServer.set_bus_mute(AudioServer.get_bus_index("Master"), false)
	get_tree().paused = false
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	if recording:
		world.flight_start()

func _input(e: InputEvent) -> void:
	if not open or not (e is InputEventKey) or not e.pressed or e.echo:
		return
	if e.keycode == KEY_ESCAPE or (e.keycode == KEY_M and not note.has_focus()):
		resume()
		get_viewport().set_input_as_handled()

# seconds into the minute of frame `fr`
func _t(fr: int) -> float:
	return float(fr - start_fr) / rate

func _span_s() -> float:
	return float(end_fr - start_fr) / rate

func _spec_input(e: InputEvent) -> void:
	var x_to_t := func(x: float) -> float: return clampf(x / spec.size.x, 0.0, 1.0) * _span_s()
	if e is InputEventMouseButton and e.button_index == MOUSE_BUTTON_LEFT:
		if e.pressed:
			dragging = true
			var t: float = x_to_t.call(e.position.x)
			sel = Vector2(t, t)
		else:
			dragging = false
			if absf(sel.y - sel.x) < 0.1:
				sel = Vector2(maxf(0.0, sel.x - 2.0), minf(_span_s(), sel.x + 2.0))
			sel = Vector2(minf(sel.x, sel.y), maxf(sel.x, sel.y))
			status.text = "picked %.1f – %.1f s (%.1f s; %.1f – %.1f s before the mark)" % [sel.x, sel.y, sel.y - sel.x, _span_s() - sel.x, _span_s() - sel.y]
			note.grab_focus()
		over.queue_redraw()
	elif e is InputEventMouseMotion and dragging:
		sel.y = x_to_t.call(e.position.x)
		over.queue_redraw()

func _draw_over() -> void:
	var w := spec.size.x
	var h := spec.size.y
	var span := maxf(_span_s(), 0.001)
	var font := ThemeDB.fallback_font
	# seconds before the mark, every 5 s
	var s := 0
	while s <= int(span):
		var x := w * (1.0 - s / span)
		over.draw_line(Vector2(x, h), Vector2(x, h + 6), Color(1, 1, 1, 0.5))
		over.draw_string(font, Vector2(x - 14, h + 20), "−%d s" % s if s > 0 else "mark", HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(1, 1, 1, 0.6))
		s += 5
	# where the ears were (a room's name where it changed)
	for rm in room_marks:
		var x: float = w * rm[0] / span
		over.draw_line(Vector2(x, h + 24), Vector2(x, h + 40), Color(0.6, 0.85, 1.0, 0.8))
		over.draw_string(font, Vector2(x + 3, h + 38), rm[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.6, 0.85, 1.0, 0.9))
	# what the player did
	for ev in events:
		if ev[0] < start_fr:
			continue
		var x := w * _t(ev[0]) / span
		over.draw_line(Vector2(x, 0), Vector2(x, h), Color(0.4, 1.0, 0.6, 0.5))
		over.draw_string(font, Vector2(x + 3, 14), ev[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.4, 1.0, 0.6, 0.9))
	# the picked stretch
	if sel.x >= 0.0:
		var a := w * minf(sel.x, sel.y) / span
		var b := w * maxf(sel.x, sel.y) / span
		over.draw_rect(Rect2(a, 0, maxf(b - a, 1.0), h), Color(1, 1, 1, 0.16))
		over.draw_line(Vector2(a, 0), Vector2(a, h), Color(1, 1, 1, 0.9), 2.0)
		over.draw_line(Vector2(b, 0), Vector2(b, h), Color(1, 1, 1, 0.9), 2.0)

func _toast(t: String) -> void:
	toast.text = t
	toast.visible = true
	toast_t = 4.0

# ------------------------------------------------------------------ save

# Where captures go: $OMG_CAPTURES, else the repository's captures/ (run
# from the source tree, or an exported demo on this machine with the
# repository at ~/dev/omg-audio), else Godot's user data folder.
static func captures_dir() -> String:
	var env := OS.get_environment("OMG_CAPTURES")
	if env != "":
		return env
	if not OS.has_feature("template"):
		return ProjectSettings.globalize_path("res://").path_join("../../captures").simplify_path()
	var repo := OS.get_environment("HOME").path_join("dev/omg-audio")
	if DirAccess.dir_exists_absolute(repo):
		return repo.path_join("captures")
	return OS.get_user_data_dir().path_join("captures")

func save() -> void:
	if not open:
		return
	var span := _span_s()
	var s0 := sel.x if sel.x >= 0.0 else 0.0
	var s1 := sel.y if sel.x >= 0.0 else span
	var stamp := Time.get_datetime_string_from_system(false, true).replace(":", "-").replace(" ", "_")
	var dir := captures_dir().path_join(stamp)
	if DirAccess.make_dir_recursive_absolute(dir) != OK:
		status.text = "could not make %s" % dir
		return
	var f0 := start_fr + int(s0 * rate)
	var f1 := start_fr + int(s1 * rate)
	var m := int(MARGIN_S * rate)
	world.flight_save(dir.path_join("minute.wav"), start_fr, end_fr)
	world.flight_save(dir.path_join("selection.wav"), f0 - m, f1 + m)
	# the minute's spectrogram with the stretch marked, and the stretch's own
	var img: Image = minute_img.duplicate()
	for t in [s0, s1]:
		var x := clampi(int(img.get_width() * t / maxf(span, 0.001)), 0, img.get_width() - 1)
		for y in img.get_height():
			img.set_pixel(x, y, Color(1, 1, 1))
	img.save_png(dir.path_join("minute.png"))
	var sw := 1600
	var a := maxi(start_fr, f0 - m)
	var b := mini(end_fr, f1 + m)
	var sb: PackedByteArray = world.flight_spectrogram(a, b, sw, 256)
	var simg := Image.create_from_data(sw, 2 * 256 + 4, false, Image.FORMAT_RGBA8, sb)
	for fr in [f0, f1]:
		var x := clampi(int(float(fr - a) / maxf(float(b - a), 1.0) * sw), 0, sw - 1)
		for y in simg.get_height():
			simg.set_pixel(x, y, Color(1, 1, 1))
	simg.save_png(dir.path_join("selection.png"))
	_csv(dir.path_join("ears.csv"), "t_s,frame,x,y,z,yaw_deg,pitch_deg,room,tick_ms,load,late,paths,rendering,fps", frames)
	_csv(dir.path_join("sources.csv"), "t_s,frame,source,x,y,z,active,peak_db,rms_db", src_log)
	_csv(dir.path_join("events.csv"), "t_s,frame,what,detail", events)
	var pf := FileAccess.open(dir.path_join("paths.jsonl"), FileAccess.WRITE)
	for p in path_log:
		if p[0] >= start_fr:
			pf.store_line('{"t_s": %.3f, "frame": %d, "paths": %s}' % [_t(p[0]), p[0], p[1]])
	pf.close()
	_notes(dir, s0, s1, f0, f1, a, b)
	status.text = "saved → %s" % dir
	print("capture: saved ", dir)

func _csv(path: String, header: String, rows: Array) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_line(header)
	for r in rows:
		if r[0] < start_fr:
			continue
		var cells := ["%.3f" % _t(r[0])]
		for c in r:
			cells.append(str(c).replace(",", ";"))
		f.store_line(",".join(cells))
	f.close()

func _notes(dir: String, s0: float, s1: float, f0: int, f1: int, a: int, b: int) -> void:
	var span := _span_s()
	var ears: Array = frames.back() if not frames.is_empty() else [0, 0.0, 0.0, 0.0, 0.0, 0.0, "?"]
	var st := world.stats()
	var bud := world.effective_budget()
	var t := "# Capture %s\n\n" % Time.get_datetime_string_from_system(false, true)
	t += "## What was heard\n\n%s\n\n" % (note.text if note.text.strip_edges() != "" else "(no note)")
	t += "## The stretch\n\n"
	t += "- %.2f – %.2f s into the minute (%.2f s long), %.2f – %.2f s before the mark\n" % [s0, s1, s1 - s0, span - s0, span - s1]
	t += "- frames %d – %d at %d Hz; selection.wav holds %d – %d (%.0f s either side)\n" % [f0, f1, rate, a, b, MARGIN_S]
	t += "- the minute: frames %d – %d (%.1f s)\n\n" % [start_fr, end_fr, span]
	t += "## At the mark\n\n"
	t += "- ears at (%.2f, %.2f, %.2f), facing %.0f°, pitch %.0f°, in %s\n" % [ears[1], ears[2], ears[3], ears[4], ears[5], ears[6]]
	t += "- quality %s (%s) · rays %d, bounces %d, late rays %d, diffraction %s\n" % [world.quality_label(), "gpu" if st.get("gpu", false) else "cpu",
		bud.get("rays", 0), bud.get("bounces", 0), bud.get("late_rays", 0), bud.get("diffraction", false)]
	t += "- sim tick %.1f ms · render load %.0f %% · late %d · %d paths · %d sources rendering\n" % [st.get("tick_ms", 0.0), 100.0 * st.get("load", 0.0),
		st.get("late", 0), st.get("paths", 0), st.get("rendering", 0)]
	t += "- Godot %s · %s\n\n" % [Engine.get_version_info().get("string", "?"), OS.get_name()]
	t += "## Files\n\n"
	t += "- minute.wav: the whole minute as played (float stereo); selection.wav: the stretch with %.0f s either side\n" % MARGIN_S
	t += "- minute.png: the minute's spectrogram (left ear over right, 40 Hz – 16 kHz), the stretch between white lines; selection.png: the stretch's own\n"
	t += "- ears.csv: every frame — where the ears were (plan x east, y north, z up, m), where they looked, the room, the engine's tick, load, late frames\n"
	t += "- sources.csv: every source %d× a second — where, sounding or not, its meter (peak, RMS dB)\n" % int(SRC_HZ)
	t += "- paths.jsonl: the %d strongest traced paths %d× a second (kind 0 direct, 1 edge bend, 2 mirror chain, 3 sampled), points ears → source in plan coordinates; energy is the path's relative mid-band gain — what it would carry, not what its source is playing (sources.csv has that)\n" % [PATHS_MAX, int(PATH_HZ)]
	t += "- events.csv: what the player did (keys, doors, the quality)\n"
	t += "- t_s in every file: seconds into the minute (frames / %d from frame %d)\n" % [rate, start_fr]
	var f := FileAccess.open(dir.path_join("README.md"), FileAccess.WRITE)
	f.store_string(t)
	f.close()
