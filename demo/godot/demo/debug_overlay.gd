# Debug overlays (switched on in the settings screen): a performance graph
# — frame time, the engine's sim tick, the audio render load, late audio
# — and the output spectrogram per ear: time running right to left (the
# newest at the right edge), log frequency up, inferno colours; left above
# right. Top right. Its columns are the output's own (the flight
# recorder's minute, OmgWorld.flight_columns), one every 5 ms of audio
# whatever the frame rate, each band in a window as long as its frequency
# needs, as the ear resolves it — 43 ms under 200 Hz, 21 to 800 Hz, 11 to
# 3 kHz, 5 above: a click is a column wide.
class_name DemoDebug
extends CanvasLayer

const W := 680.0
const SW := W           # the spectrogram's width
const H := 220.0
const SH := 330.0       # the spectrum panel: two flame graphs
const TOP := 36.0       # below the title
const N := 240          # samples kept (~4 s at 60 fps)
const BANDS := 256      # log bands, 40 Hz – 16 kHz (0.4 semitone each)
const COL_PX := 1.0     # a column's width on screen
const COLS := int(SW / COL_PX)
const COL_HZ := 200.0   # columns per second of audio (3.4 s shown)
const F_LO := 40.0
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
var lut := PackedColorArray()
var flame: Array[Image] = []
var flame_tex: Array[ImageTexture] = []
var next_fr := -1         # the output frame the next column starts at

func _init(w: OmgWorld) -> void:
	world = w
	layer = 15

func _ready() -> void:
	for a in [frame_ms, tick_ms, load_pc]:
		a.resize(N)
	late_flags.resize(N)
	for ear in 2:
		var img := Image.create(COLS, BANDS, false, Image.FORMAT_RGBA8)
		img.fill(_inferno(0.0))
		flame.append(img)
		flame_tex.append(ImageTexture.create_from_image(img))
	perf = _panel(Vector2(-W - 16, 16), H, _draw_perf)
	spec = _panel(Vector2(-SW - 16, 16 + H + 16), SH, _draw_spectrum, SW)
	# the image smoothed onto the panel, not blown up into blocks
	spec.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	_apply()

func set_perf(on: bool) -> void:
	perf_on = on
	_apply()

func set_spectrum(on: bool) -> void:
	spectrum_on = on
	if on:
		world.flight_start()
		next_fr = -1
	_apply()

func _apply() -> void:
	if perf == null:
		return
	perf.visible = perf_on
	spec.visible = spectrum_on
	spec.position.y = (16 + H + 16) if perf_on else 16

func _panel(at: Vector2, h: float, draw: Callable, w := W) -> Control:
	var c := Control.new()
	c.anchor_left = 1.0
	c.anchor_right = 1.0
	c.position = at
	c.size = Vector2(w, h)
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
	if spectrum_on:
		# the columns the output has played since the last frame
		var hop := maxi(1, int(world.flight_rate() / COL_HZ))
		if next_fr < 0:
			next_fr = maxi(0, world.flight_frames() - COLS * hop)
		var r: Dictionary = world.flight_columns(next_fr, hop, BANDS, COLS)
		var k: int = r["columns"]
		next_fr = r["next"]
		if k > 0:
			var cols := Image.create_from_data(k, 2 * BANDS + 4, false, Image.FORMAT_RGBA8, r["rgba"])
			# everything moves k columns left; the newest enter at the right
			for ear in 2:
				var img: Image = flame[ear]
				if k < COLS:
					img.blit_rect(img.get_region(Rect2i(k, 0, COLS - k, BANDS)), Rect2i(0, 0, COLS - k, BANDS), Vector2i.ZERO)
				img.blit_rect(cols, Rect2i(0, ear * (BANDS + 4), k, BANDS), Vector2i(COLS - k, 0))
				flame_tex[ear].update(img)
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

# Inferno (black → purple → crimson → orange → yellow), the web's
# SPEC_LUT stops, as a 256-entry table.
func _inferno(t: float) -> Color:
	if lut.is_empty():
		var stops := [[0.0, 0, 0, 4], [0.125, 22, 11, 57], [0.25, 66, 10, 104], [0.375, 106, 23, 110],
			[0.5, 147, 38, 103], [0.625, 188, 55, 84], [0.75, 221, 81, 58], [0.875, 243, 120, 25],
			[0.9375, 249, 189, 38], [1.0, 252, 255, 164]]
		for i in 256:
			var x := i / 255.0
			var k := 0
			while stops[k + 1][0] < x:
				k += 1
			var u: float = (x - stops[k][0]) / (stops[k + 1][0] - stops[k][0])
			var a: Array = stops[k]
			var b: Array = stops[k + 1]
			lut.append(Color8(roundi(lerpf(a[1], b[1], u)), roundi(lerpf(a[2], b[2], u)), roundi(lerpf(a[3], b[3], u))))
	return lut[clampi(roundi(t * 255.0), 0, 255)]

func _draw_spectrum(c: Control) -> void:
	_bg(c, "output spectrogram · 40 Hz – 16 kHz · last %d s" % int(COLS / COL_HZ))
	var top := TOP
	var gap := 6.0
	var gh := (SH - top - 10.0 - gap) * 0.5
	for ear in 2:
		var y0 := top + ear * (gh + gap)
		c.draw_texture_rect(flame_tex[ear], Rect2(0, y0, SW, gh), false)
		# frequency gridlines: 100 Hz, 1 kHz, 10 kHz
		for f in [100.0, 1000.0, 10000.0]:
			var y := y0 + gh * (1.0 - log(f / F_LO) / log(F_HI / F_LO))
			c.draw_line(Vector2(0, y), Vector2(SW, y), Color(0.48, 0.52, 0.59, 0.27))
			c.draw_string(ThemeDB.fallback_font, Vector2(6, y - 3), "%dk" % int(f / 1000) if f >= 1000 else "%d" % int(f),
				HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Color(1, 1, 1, 0.5))
		c.draw_string(ThemeDB.fallback_font, Vector2(SW - 30, y0 + 24), ["L", "R"][ear], HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color(1, 1, 1, 0.8))
