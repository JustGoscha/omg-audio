# Debug overlays (switched on in the settings screen): a performance graph
# — frame time, the engine's sim tick, the audio render load, late audio
# — and the output spectrum as a flame graph (a spectrogram: frequency
# across, the last few seconds rising from the bottom, hotter = louder),
# top right.
class_name DemoDebug
extends CanvasLayer

const W := 680.0
const H := 220.0
const TOP := 36.0       # below the title
const N := 240          # samples kept (~4 s at 60 fps)
const BANDS := 96
const ROWS := 150       # spectrogram history
const ROW_HZ := 50.0    # rows per second (3 s shown)
const F_LO := 30.0
const F_HI := 16000.0

var world: OmgWorld
var perf_on := false
var spectrum_on := false
var perf: Control
var spec: Control
var frame_ms := PackedFloat32Array()
var tick_ms := PackedFloat32Array()
var load_pc := PackedFloat32Array()
var late_last := 0
var late_flags := PackedByteArray()
var analyzer: AudioEffectSpectrumAnalyzerInstance
var bands := PackedFloat32Array()  # loudest per band since the last row
var flame: Image
var flame_tex: ImageTexture
var row_t := 0.0

func _init(w: OmgWorld) -> void:
	world = w
	layer = 15

func _ready() -> void:
	for a in [frame_ms, tick_ms, load_pc]:
		a.resize(N)
	late_flags.resize(N)
	bands.resize(BANDS)
	bands.fill(-120.0)
	flame = Image.create(BANDS, ROWS, false, Image.FORMAT_RGBA8)
	flame.fill(Color(0, 0, 0, 0))
	flame_tex = ImageTexture.create_from_image(flame)
	perf = _panel(Vector2(-W - 16, 16), _draw_perf)
	spec = _panel(Vector2(-W - 16, 16 + H + 16), _draw_spectrum)
	_apply()

func set_perf(on: bool) -> void:
	perf_on = on
	_apply()

func set_spectrum(on: bool) -> void:
	spectrum_on = on
	if on and analyzer == null:
		# an analyser on the master bus (the engine plays through it)
		var bus := AudioServer.get_bus_index("Master")
		var fx := AudioEffectSpectrumAnalyzer.new()
		fx.fft_size = AudioEffectSpectrumAnalyzer.FFT_SIZE_2048
		AudioServer.add_bus_effect(bus, fx)
		analyzer = AudioServer.get_bus_effect_instance(bus, AudioServer.get_bus_effect_count(bus) - 1)
	_apply()

func _apply() -> void:
	if perf == null:
		return
	perf.visible = perf_on
	spec.visible = spectrum_on
	spec.position.y = (16 + H + 16) if perf_on else 16

func _panel(at: Vector2, draw: Callable) -> Control:
	var c := Control.new()
	c.anchor_left = 1.0
	c.anchor_right = 1.0
	c.position = at
	c.size = Vector2(W, H)
	c.mouse_filter = Control.MOUSE_FILTER_IGNORE
	c.draw.connect(draw.bind(c))
	add_child(c)
	return c

func _process(dt: float) -> void:
	if perf_on:
		var st := world.stats()
		_push(frame_ms, dt * 1000.0)
		_push(tick_ms, st.get("tick_ms", 0.0))
		_push(load_pc, 100.0 * st.get("load", 0.0))
		var late: int = st.get("late", 0)
		late_flags.remove_at(0)
		late_flags.append(1 if late > late_last else 0)
		late_last = late
		perf.queue_redraw()
	if spectrum_on and analyzer != null:
		# log-spaced bands, dB, the loudest of each between two rows
		for i in BANDS:
			var f0 := F_LO * pow(F_HI / F_LO, float(i) / BANDS)
			var f1 := F_LO * pow(F_HI / F_LO, float(i + 1) / BANDS)
			var m := analyzer.get_magnitude_for_frequency_range(f0, f1, AudioEffectSpectrumAnalyzerInstance.MAGNITUDE_MAX)
			bands[i] = maxf(bands[i], linear_to_db(maxf(m.x, m.y) + 1e-9))
		row_t += dt
		if row_t >= 1.0 / ROW_HZ:
			row_t = fmod(row_t, 1.0 / ROW_HZ)
			# everything moves up a row; the newest enters at the bottom
			flame.blit_rect(flame.get_region(Rect2i(0, 1, BANDS, ROWS - 1)), Rect2i(0, 0, BANDS, ROWS - 1), Vector2i.ZERO)
			for i in BANDS:
				flame.set_pixel(i, ROWS - 1, _flame_color(clampf((bands[i] + 90.0) / 84.0, 0.0, 1.0)))
			bands.fill(-120.0)
			flame_tex.update(flame)
			spec.queue_redraw()

func _push(a: PackedFloat32Array, v: float) -> void:
	a.remove_at(0)
	a.append(v)

func _bg(c: Control, title: String) -> void:
	c.draw_rect(Rect2(Vector2.ZERO, c.size), Color(0.03, 0.03, 0.05, 0.72))
	c.draw_string(ThemeDB.fallback_font, Vector2(12, 26), title, HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color(0.7, 0.72, 0.78))

func _draw_perf(c: Control) -> void:
	_bg(c, "frame ms (white) · sim tick ms (gold) · load % (cyan) · late (red)")
	var top := TOP
	var gh := H - top - 34.0
	var y_of := func(ms: float) -> float: return top + gh * (1.0 - clampf(ms / 60.0, 0.0, 1.0))
	# guides: 60, 30 fps and the 50 ms sim tick budget
	for g in [[16.7, "60"], [33.3, "30"], [50.0, "tick budget"]]:
		var y: float = y_of.call(g[0])
		c.draw_line(Vector2(0, y), Vector2(W, y), Color(1, 1, 1, 0.12))
		c.draw_string(ThemeDB.fallback_font, Vector2(W - 120, y - 3), g[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 17, Color(1, 1, 1, 0.35))
	var dx := W / (N - 1)
	for i in N:
		if late_flags[i]:
			c.draw_line(Vector2(i * dx, top), Vector2(i * dx, top + gh), Color(1, 0.2, 0.2, 0.6), 2.0)
	for series in [[frame_ms, Color(0.95, 0.95, 0.95), 1.0], [tick_ms, Color(1.0, 0.8, 0.4), 1.0], [load_pc, Color(0.4, 0.85, 1.0), 0.6]]:
		var a: PackedFloat32Array = series[0]
		var pts := PackedVector2Array()
		for i in N:
			pts.append(Vector2(i * dx, y_of.call(a[i] * series[2])))
		c.draw_polyline(pts, series[1], 2.0)
	c.draw_string(ThemeDB.fallback_font, Vector2(12, H - 10), "%.1f ms · sim %.1f ms · load %.0f %%" % [frame_ms[N - 1], tick_ms[N - 1], load_pc[N - 1]],
		HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color(0.9, 0.9, 0.95))

# Black → red → orange → yellow → white with the level (0..1); quiet
# is see-through.
func _flame_color(t: float) -> Color:
	var c: Color
	if t < 0.4:
		c = Color(0.1, 0, 0).lerp(Color(0.85, 0.08, 0.02), t / 0.4)
	elif t < 0.7:
		c = Color(0.85, 0.08, 0.02).lerp(Color(1.0, 0.6, 0.05), (t - 0.4) / 0.3)
	elif t < 0.9:
		c = Color(1.0, 0.6, 0.05).lerp(Color(1.0, 0.95, 0.3), (t - 0.7) / 0.2)
	else:
		c = Color(1.0, 0.95, 0.3).lerp(Color(1, 1, 1), (t - 0.9) / 0.1)
	c.a = clampf(t * 4.0, 0.0, 1.0)
	return c

func _draw_spectrum(c: Control) -> void:
	_bg(c, "output spectrum · 30 Hz – 16 kHz · last %d s, newest at the bottom" % int(ROWS / ROW_HZ))
	var top := TOP
	var gh := H - top - 28.0
	c.draw_texture_rect(flame_tex, Rect2(0, top, W, gh), false)
	for f in [100.0, 1000.0, 10000.0]:
		var x := W * log(f / F_LO) / log(F_HI / F_LO)
		c.draw_line(Vector2(x, top), Vector2(x, top + gh), Color(1, 1, 1, 0.1))
		c.draw_string(ThemeDB.fallback_font, Vector2(x - 16, H - 6), "%dk" % int(f / 1000) if f >= 1000 else "%d" % int(f),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 18, Color(1, 1, 1, 0.6))
