# The settings screen (Esc), after the web demo's quality and mixer
# panels, in tabs: the engine (quality tier, the tracer's levers pinned
# over the tier, compute CPU / GPU, diffraction, the audio render
# budget), the mixer (a fader, mute and meter per sound), graphics, the
# debug overlays; the engine's live stats under all of them.
class_name DemoSettings
extends CanvasLayer

const INK := Color(0.86, 0.84, 0.78)
const DIM := Color(0.55, 0.55, 0.58)
const GOLD := Color(1.0, 0.86, 0.6)

var demo # the demo root (demo.gd): quality, fine, pixel_h, torch, …
var world: OmgWorld
var open := false
var panel: PanelContainer
var sliders := {}    # lever → [HSlider, value Label]
var groups := {}     # name → Array[Button] (one pressed)
var stats_label: Label
var refresh_t := 0.0
var strips := []     # mixer channels: {srcs, meter: Control, rms, peak}

func _init(d, w: OmgWorld) -> void:
	demo = d
	world = w
	layer = 20

# Type sizes (UI units; the whole 2D layer scales with the window).
const TITLE := 38
const TEXT := 24
const SMALL := 20
const NAME_W := 260.0

func _ready() -> void:
	panel = PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.07, 0.07, 0.08, 0.96)
	sb.border_color = Color(0.22, 0.22, 0.24)
	sb.set_border_width_all(1)
	sb.set_corner_radius_all(12)
	sb.set_content_margin_all(30)
	panel.add_theme_stylebox_override("panel", sb)
	panel.set_anchors_preset(Control.PRESET_CENTER)
	panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	panel.grow_vertical = Control.GROW_DIRECTION_BOTH
	panel.custom_minimum_size = Vector2(980, 0)
	add_child(panel)
	var outer := VBoxContainer.new()
	outer.add_theme_constant_override("separation", 12)
	panel.add_child(outer)

	outer.add_child(_label("settings", TITLE, INK))
	outer.add_child(_label("Esc closes · changes apply at once", SMALL, DIM))
	var tabs := TabContainer.new()
	tabs.add_theme_font_size_override("font_size", TEXT)
	tabs.custom_minimum_size = Vector2(0, 560)
	outer.add_child(tabs)

	var v := _tab(tabs, "engine")
	_section(v, "engine quality", false)
	_choice(v, "quality", ["low", "medium", "high", "ultra"], world.quality, func(i: int) -> void:
		demo.quality = i
		world.quality = i)

	_section(v, "listener tracer · drag to pin a lever over the tier")
	_slider(v, "rays", "rays / tick", 6, 15, true)
	_slider(v, "bounces", "reflections", 1, 16, false)
	_slider(v, "late_rays", "late rays", 4, 13, true)
	var release := Button.new()
	release.text = "release pins → tier"
	release.flat = true
	release.add_theme_font_size_override("font_size", SMALL)
	release.add_theme_color_override("font_color", DIM)
	release.pressed.connect(func() -> void:
		world.rays = 0
		world.bounces = 0
		world.late_rays = 0
		_refresh())
	v.add_child(release)

	_section(v, "engine modules")
	_row(v, "engine", _choice_box("engine", ["v3", "v4"], 1 if world.engine_v4 else 0, func(i: int) -> void:
		world.engine_v4 = i == 1))
	var web := OS.has_feature("web")
	_row(v, "compute", _choice_box("compute", ["cpu", "gpu"], 1 if world.use_gpu else 0, func(i: int) -> void:
		world.use_gpu = i == 1, [false, web]))
	_row(v, "diffraction", _choice_box("diffraction", ["on", "off"], 0 if world.diffraction else 1, func(i: int) -> void:
		world.diffraction = i == 0))
	_row(v, "render budget", _choice_box("budget", ["auto", "1", "2", "3", "4", "5"], 0, func(i: int) -> void:
		world.set_render_budget(i - 1)))

	_mixer(_tab(tabs, "mixer"))

	v = _tab(tabs, "graphics")
	_section(v, "graphics", false)
	_row(v, "look", _choice_box("look", ["pixel", "fine"], 1 if demo.fine else 0, func(i: int) -> void:
		demo.fine = i == 1
		demo._apply_graphics()))
	var heights: Array = demo.HEIGHTS
	_row(v, "pixel height", _choice_box("pixel_h", heights.map(func(h: int) -> String: return str(h)), heights.find(demo.pixel_h), func(i: int) -> void:
		demo.pixel_h = heights[i]
		demo._apply_graphics()))
	_row(v, "flashlight", _choice_box("torch", ["off", "on"], 1 if demo.torch.visible else 0, func(i: int) -> void:
		demo.torch.visible = i == 1))

	v = _tab(tabs, "debug")
	_section(v, "overlays", false)
	_row(v, "performance", _choice_box("perf", ["off", "on"], 0, func(i: int) -> void:
		demo.debug.set_perf(i == 1)))
	_row(v, "spectrum", _choice_box("spectrum", ["off", "on"], 0, func(i: int) -> void:
		demo.debug.set_spectrum(i == 1)))
	_row(v, "sound paths", _choice_box("paths", ["off", "exact", "all"], 0, func(i: int) -> void:
		demo.paths_view.set_mode(i)))

	# the live stats, under every tab
	stats_label = _label("", SMALL, INK)
	stats_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	outer.add_child(stats_label)
	panel.visible = false

# A tab: a padded column.
func _tab(tabs: TabContainer, name: String) -> VBoxContainer:
	var m := MarginContainer.new()
	m.name = name
	for side in ["left", "right", "top", "bottom"]:
		m.add_theme_constant_override("margin_" + side, 18)
	tabs.add_child(m)
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 12)
	m.add_child(v)
	return v

# The mixer: per sound a mute (not traced, costs nothing), a fader over
# its level in the scene (dB SPL at 1 m shown) and its level at the ears;
# ambience, rain and the master as trims, the output's meter on master.
func _mixer(v: VBoxContainer) -> void:
	v.add_theme_constant_override("separation", 2)
	var snd: DemoSounds = demo.sounds
	var src: Dictionary = snd.sources
	var club := []
	for k in src:
		if k.begins_with("club"):
			club.append(src[k])
	for ch in [["music", [src["music"]]], ["voice", [src["voice"]]], ["club", club], ["flute", [src["flute"]]],
			["radio", [src["radio"]]], ["owls", [src["owl·cathedral"], src["owl·house"]]], ["bells", [src["bells"]]],
			["fountain", [src["fountain"]]], ["balls", snd.balls], ["cars", snd.cars_src], ["steps", [snd.feet]],
			["clap", [snd.hands]]]:
		var srcs: Array = ch[1]
		_strip(v, ch[0], srcs, -40.0, 12.0, srcs[0].volume_db, func(x: float, mute: bool) -> String:
			for o in srcs:
				o.volume_db = x
				o.muted = mute
			return "off" if mute else "%+.0f dB · %.0f SPL" % [x, srcs[0].level_db + x])
	v.add_child(HSeparator.new())
	_strip(v, "ambience", [], -40.0, 6.0, world.ambience_db, func(x: float, mute: bool) -> String:
		world.ambience_db = -120.0 if mute else x
		return "off" if mute else "%+.0f dB" % x)
	_strip(v, "rain", [], -30.0, 12.0, world.rain_db, func(x: float, mute: bool) -> String:
		world.rain_db = -120.0 if mute else x
		return "off" if mute else "%+.0f dB" % x)
	_strip(v, "master", ["out"], -30.0, 6.0, world.master_db, func(x: float, mute: bool) -> String:
		world.master_db = -120.0 if mute else x
		return "off" if mute else "%+.0f dB" % x)

func _strip(v: VBoxContainer, name: String, srcs: Array, lo: float, hi: float, at: float, apply: Callable) -> void:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 14)
	var mute := Button.new()
	mute.text = "●"
	mute.flat = true
	mute.toggle_mode = true
	mute.tooltip_text = "mute: a muted sound is not traced or rendered (costs nothing)"
	mute.add_theme_font_size_override("font_size", TEXT)
	mute.add_theme_color_override("font_color", INK)
	mute.add_theme_color_override("font_pressed_color", Color(0.3, 0.3, 0.33))
	mute.add_theme_color_override("font_hover_pressed_color", Color(0.4, 0.4, 0.43))
	h.add_child(mute)
	var l := _label(name, TEXT, INK)
	l.custom_minimum_size = Vector2(130, 0)
	h.add_child(l)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.value = at
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	s.custom_minimum_size = Vector2(0, 30)
	h.add_child(s)
	var strip := {"srcs": srcs, "rms": -90.0, "peak": -90.0}
	var meter := Control.new()
	meter.custom_minimum_size = Vector2(200, 0)
	meter.mouse_filter = Control.MOUSE_FILTER_IGNORE
	meter.modulate.a = 0.0 if srcs.is_empty() else 1.0 # (ambience and rain: no meter)
	meter.draw.connect(func() -> void: _draw_meter(meter, strip))
	h.add_child(meter)
	var val := _label("", SMALL, DIM)
	val.custom_minimum_size = Vector2(210, 0)
	val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	h.add_child(val)
	var update := func() -> void:
		val.text = apply.call(s.value, mute.button_pressed)
		l.modulate.a = 0.4 if mute.button_pressed else 1.0
	s.value_changed.connect(func(_x: float) -> void: update.call())
	mute.toggled.connect(func(_on: bool) -> void: update.call())
	update.call()
	v.add_child(h)
	strip["meter"] = meter
	if not srcs.is_empty():
		strips.append(strip)

# RMS as a bar, the peak as a tick, −60 … 0 dBFS.
func _draw_meter(c: Control, strip: Dictionary) -> void:
	var y := c.size.y * 0.5 - 5.0
	c.draw_rect(Rect2(0, y, c.size.x, 10), Color(0.16, 0.16, 0.19))
	var x_of := func(d: float) -> float: return c.size.x * clampf((d + 60.0) / 60.0, 0.0, 1.0)
	var rms: float = strip["rms"]
	var col := Color(0.45, 0.8, 0.55) if rms < -12.0 else (Color(1.0, 0.8, 0.35) if rms < -3.0 else Color(1.0, 0.35, 0.3))
	c.draw_rect(Rect2(0, y, x_of.call(rms), 10), col)
	c.draw_rect(Rect2(x_of.call(strip["peak"]) - 1.0, y - 2, 3, 14), Color(1, 1, 1, 0.85))

func _update_meters() -> void:
	var m: Dictionary = world.meters()
	for strip in strips:
		var sum2 := 0.0
		var peak := 0.0
		for o in strip["srcs"]:
			var k = "out" if o is String else (o as OmgSource).source_id()
			var lv: Vector2 = m.get(k, Vector2.ZERO)
			peak = maxf(peak, lv.x)
			sum2 += lv.y * lv.y
		var rms := linear_to_db(sqrt(sum2) + 1e-9)
		var pk := linear_to_db(peak + 1e-9)
		# a quick rise, a slower fall (~20 dB/s)
		strip["rms"] = rms if rms > strip["rms"] else maxf(rms, strip["rms"] - 2.0)
		strip["peak"] = pk if pk > strip["peak"] else maxf(pk, strip["peak"] - 2.0)
		(strip["meter"] as Control).queue_redraw()

func toggle() -> void:
	open = not open
	panel.visible = open
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE if open else Input.MOUSE_MODE_CAPTURED
	demo.player.set_physics_process(not open)
	demo.player.set_process_unhandled_input(not open)
	if open:
		_refresh()

func _process(dt: float) -> void:
	if not open:
		return
	_update_meters()
	refresh_t += dt
	if refresh_t >= 0.5:
		refresh_t = 0.0
		_refresh()

# Show what is in effect: the tier's levers unless pinned, the live stats.
func _refresh() -> void:
	var b: Dictionary = world.effective_budget()
	for lever in sliders:
		var s: HSlider = sliders[lever][0]
		var val: int = b[lever]
		s.set_value_no_signal(log(maxf(1.0, val)) / log(2.0) if s.has_meta("log") else val)
		var pinned: bool = world.get(lever) > 0
		(sliders[lever][1] as Label).text = ("%d" % val) + ("  pinned" if pinned else "")
	_press("quality", world.quality)
	var st: Dictionary = world.stats()
	var tick: float = st.get("tick_ms", 0.0)
	stats_label.text = "sim tick %.1f ms (%.0f %%) · %s · render load %.0f %% · %d/%d sources · %d paths · late %d" % [
		tick, tick / 50.0 * 100.0, "gpu" if st.get("gpu", false) else "cpu", 100.0 * st.get("load", 0.0),
		st.get("rendering", 0), st.get("sources", 0), st.get("paths", 0), st.get("late", 0)]

# ---------------------------------------------------------------- widgets

func _label(t: String, size: int, c: Color) -> Label:
	var l := Label.new()
	l.text = t
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", c)
	return l

func _section(v: VBoxContainer, t: String, rule := true) -> void:
	if rule:
		v.add_child(HSeparator.new())
	v.add_child(_label(t, TEXT, GOLD))

func _row(v: VBoxContainer, name: String, right: Control) -> void:
	var h := HBoxContainer.new()
	var l := _label(name, TEXT, INK)
	l.custom_minimum_size = Vector2(NAME_W, 0)
	h.add_child(l)
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	h.add_child(right)
	v.add_child(h)

func _choice(v: VBoxContainer, group: String, names: Array, pressed: int, on: Callable) -> void:
	var box := _choice_box(group, names, pressed, on)
	v.add_child(box)

# A row of toggle buttons, one pressed; `disabled[i]` greys option i.
func _choice_box(group: String, names: Array, pressed: int, on: Callable, disabled := []) -> HBoxContainer:
	var h := HBoxContainer.new()
	var bg := ButtonGroup.new()
	var buttons: Array[Button] = []
	for i in names.size():
		var b := Button.new()
		b.text = names[i]
		b.toggle_mode = true
		b.button_group = bg
		b.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		b.add_theme_font_size_override("font_size", TEXT)
		b.custom_minimum_size = Vector2(0, 48)
		b.button_pressed = i == pressed
		b.disabled = i < disabled.size() and disabled[i]
		b.pressed.connect(func() -> void: on.call(i))
		h.add_child(b)
		buttons.append(b)
	groups[group] = buttons
	return h

func _press(group: String, i: int) -> void:
	var buttons: Array = groups.get(group, [])
	if i >= 0 and i < buttons.size():
		(buttons[i] as Button).set_pressed_no_signal(true)

# A lever slider (log2 scale for ray counts); moving it pins the lever.
func _slider(v: VBoxContainer, lever: String, name: String, lo: int, hi: int, log2: bool) -> void:
	var h := HBoxContainer.new()
	var l := _label(name, TEXT, INK)
	l.custom_minimum_size = Vector2(NAME_W, 0)
	h.add_child(l)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	s.custom_minimum_size = Vector2(0, 40)
	if log2:
		s.set_meta("log", true)
	var val := _label("", TEXT, INK)
	val.custom_minimum_size = Vector2(200, 0)
	val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	s.value_changed.connect(func(x: float) -> void:
		world.set(lever, int(pow(2.0, x)) if log2 else int(x))
		_refresh())
	h.add_child(s)
	h.add_child(val)
	v.add_child(h)
	sliders[lever] = [s, val]
