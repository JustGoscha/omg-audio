# The settings screen (Esc), after the web demo's quality panel: engine
# quality tier, the tracer's levers (pinned over the tier), compute CPU /
# GPU, diffraction, the audio render budget, volumes, graphics, and the
# engine's live stats.
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

func _init(d, w: OmgWorld) -> void:
	demo = d
	world = w
	layer = 20

func _ready() -> void:
	panel = PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.07, 0.07, 0.08, 0.96)
	sb.border_color = Color(0.22, 0.22, 0.24)
	sb.set_border_width_all(1)
	sb.set_corner_radius_all(10)
	sb.set_content_margin_all(22)
	panel.add_theme_stylebox_override("panel", sb)
	panel.set_anchors_preset(Control.PRESET_CENTER)
	panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	panel.grow_vertical = Control.GROW_DIRECTION_BOTH
	panel.custom_minimum_size = Vector2(560, 0)
	add_child(panel)
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 8)
	panel.add_child(v)

	var title := _label("settings", 26, INK)
	v.add_child(title)
	v.add_child(_label("Esc closes · changes apply at once", 14, DIM))

	_section(v, "engine quality")
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
	release.add_theme_color_override("font_color", DIM)
	release.pressed.connect(func() -> void:
		world.rays = 0
		world.bounces = 0
		world.late_rays = 0
		_refresh())
	v.add_child(release)

	_section(v, "engine modules")
	var web := OS.has_feature("web")
	_row(v, "compute", _choice_box("compute", ["cpu", "gpu"], 1 if world.use_gpu else 0, func(i: int) -> void:
		world.use_gpu = i == 1, [false, web]))
	_row(v, "diffraction", _choice_box("diffraction", ["on", "off"], 0 if world.diffraction else 1, func(i: int) -> void:
		world.diffraction = i == 0))
	_row(v, "render budget", _choice_box("budget", ["auto", "1", "2", "3", "4", "5"], 0, func(i: int) -> void:
		world.set_render_budget(i - 1)))

	_section(v, "volume")
	_db_slider(v, "master", -30.0, 6.0, world.master_db, func(x: float) -> void: world.master_db = x)
	_db_slider(v, "ambience", -40.0, 0.0, world.ambience_db, func(x: float) -> void: world.ambience_db = x)

	_section(v, "graphics")
	_row(v, "look", _choice_box("look", ["pixel", "fine"], 1 if demo.fine else 0, func(i: int) -> void:
		demo.fine = i == 1
		demo._apply_graphics()))
	var heights: Array = demo.HEIGHTS
	_row(v, "pixel height", _choice_box("pixel_h", heights.map(func(h: int) -> String: return str(h)), heights.find(demo.pixel_h), func(i: int) -> void:
		demo.pixel_h = heights[i]
		demo._apply_graphics()))
	_row(v, "flashlight", _choice_box("torch", ["off", "on"], 1 if demo.torch.visible else 0, func(i: int) -> void:
		demo.torch.visible = i == 1))

	_section(v, "debug")
	_row(v, "performance graph", _choice_box("perf", ["off", "on"], 0, func(i: int) -> void:
		demo.debug.set_perf(i == 1)))
	_row(v, "audio spectrum", _choice_box("spectrum", ["off", "on"], 0, func(i: int) -> void:
		demo.debug.set_spectrum(i == 1)))

	v.add_child(HSeparator.new())
	stats_label = _label("", 15, INK)
	v.add_child(stats_label)
	panel.visible = false

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

func _section(v: VBoxContainer, t: String) -> void:
	v.add_child(HSeparator.new())
	v.add_child(_label(t, 16, GOLD))

func _row(v: VBoxContainer, name: String, right: Control) -> void:
	var h := HBoxContainer.new()
	var l := _label(name, 16, INK)
	l.custom_minimum_size = Vector2(150, 0)
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
		b.add_theme_font_size_override("font_size", 16)
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
	var l := _label(name, 16, INK)
	l.custom_minimum_size = Vector2(150, 0)
	h.add_child(l)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	if log2:
		s.set_meta("log", true)
	var val := _label("", 16, INK)
	val.custom_minimum_size = Vector2(130, 0)
	val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	s.value_changed.connect(func(x: float) -> void:
		world.set(lever, int(pow(2.0, x)) if log2 else int(x))
		_refresh())
	h.add_child(s)
	h.add_child(val)
	v.add_child(h)
	sliders[lever] = [s, val]

func _db_slider(v: VBoxContainer, name: String, lo: float, hi: float, at: float, on: Callable) -> void:
	var h := HBoxContainer.new()
	var l := _label(name, 16, INK)
	l.custom_minimum_size = Vector2(150, 0)
	h.add_child(l)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.value = at
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	var val := _label("%+.0f dB" % at, 16, INK)
	val.custom_minimum_size = Vector2(130, 0)
	val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	s.value_changed.connect(func(x: float) -> void:
		on.call(x)
		val.text = "%+.0f dB" % x)
	h.add_child(s)
	h.add_child(val)
	v.add_child(h)
