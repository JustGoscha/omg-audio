# The omg-audio town at night, in Godot: the web demo's level and sound
# design on the engine's Godot nodes. The level is generated (demo/level.gd),
# its architecture tagged with acoustic materials; OmgWorld collects that
# geometry and traces it — what you see is what the engine hears.
#
# Click: capture mouse / throw a ball · WASD walk · Shift run · Space jump
# C crouch · F fly · E door · B bells · V whisper · R rain · L labels
# Q sound quality · G graphics (pixel ⇄ fine) · [ ] pixel height · H hud
# Esc release mouse.
#
# Graphics default to a cheap, low-res "pixel" look: the 3D view renders
# into a SubViewport 360 px high (width from the window's aspect),
# upscaled to fill the window with nearest-neighbour filtering — no
# anti-aliasing anywhere, nearest-filtered textures, no SSAO/SSIL/
# volumetric fog; the moon and the lamps cast hard shadows from a small
# shadow atlas (chunky, like the pixels). [ and ] change the height; G
# switches to the fine look (full resolution, MSAA, SSAO, SSIL,
# volumetric fog, soft shadows from a 4096 atlas).
extends Node3D

const SKY := "res://../assets/sky/night.jpg"
const LAMP_SHADOW_M := 30.0 # lamp shadows fade out beyond this distance

var world: OmgWorld
var level: DemoLevel
var player: DemoPlayer
var sounds: DemoSounds
var hud: Label
var hint: Label
var labels: Array[Label3D] = []
var rain_name := "off"
# OmgWorld.quality: 0 low … 3 ultra. Medium on the CPU walk: ~10–40 ms a
# tick here on its own thread (High peaks over the 50 ms tick in rooms)
var quality := 1
var t_flicker := 0.0
var env: Environment
var view: SubViewport
var view_rect: TextureRect
var view_cam: Camera3D
var fine := false
# 3D internal height in pixel mode (keys [ and ] walk HEIGHTS)
const HEIGHTS := [180, 240, 270, 360, 480, 540, 720]
var pixel_h := 360

func _ready() -> void:
	_environment()
	_viewport()
	var level_root := Node3D.new()
	level_root.name = "Level"
	add_child(level_root)
	level = DemoLevel.new(level_root)
	level.labels = "--labels" in OS.get_cmdline_user_args()
	level.build()
	if "--audit" in OS.get_cmdline_user_args():
		var audit: Dictionary = level.geo.audit()
		print("texture → acoustic material:")
		for line in audit["pairs"]:
			print("  ", line)
		for w in audit["warnings"]:
			push_warning(w)

	player = DemoPlayer.new()
	player.name = "Player"
	add_child(player)
	player.global_position = DemoGeo.P(3.0, 3.0, 0.05)
	player.set_heading(PI / 2.0)
	# every lamp casts shadows in both looks; past LAMP_SHADOW_M the
	# shadow fades out (the light stays), so only the lamps around you
	# render shadow maps
	for l in level_root.find_children("*", "OmniLight3D", true, false):
		var o := l as OmniLight3D
		o.shadow_enabled = true
		o.distance_fade_enabled = true
		o.distance_fade_begin = 4000.0
		o.distance_fade_shadow = LAMP_SHADOW_M
		o.distance_fade_length = 8.0
	_apply_graphics()

	world = OmgWorld.new()
	world.name = "OmgWorld"
	world.geometry_root = NodePath("../Level")
	world.ambience_file = DemoSounds.A + "night-nature48.ogg"
	world.ambience_db = -16.0
	world.rain_bank_file = DemoSounds.A + "drops48.ogg"
	# Trace on the CPU by default: this town costs the CPU walk ~4 ms a
	# tick on its own thread, while the GPU walker competes with the
	# renderer for the GPU (measured on an M1 Max: 60–105 fps on the CPU
	# walk vs 15–100 with the GPU walk). `-- --gpu` switches it on.
	world.use_gpu = "--gpu" in OS.get_cmdline_user_args()
	world.quality = quality
	add_child(world)
	var g := world.rebuild_geometry()
	print("omg: acoustic geometry %d triangles, %d boxes from %d meshes" % [g["triangles"], g["boxes"], g["meshes"]])

	sounds = DemoSounds.new()
	add_child(sounds)
	sounds.setup(world, level_root, level, player)
	_hud()
	_labels()
	var args := OS.get_cmdline_user_args()
	if "--labels" in args:
		for l in labels:
			l.visible = true
	for i in args.size():
		var a: String = args[i]
		if a == "--fine":
			fine = true
			_apply_graphics()
		if a.begins_with("--shots="):
			_shots(a.trim_prefix("--shots="))
		elif a == "--shots" and i + 1 < args.size():
			_shots(args[i + 1])
		if a == "--print-fps":
			_print_fps()

# `-- --print-fps`: ~10 s of frame rate (start room, then the square,
# slowly turning), printed, then quit.
func _print_fps() -> void:
	player.set_physics_process(false)
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED) # measure headroom
	for i in 60:
		await get_tree().process_frame
	var res := []
	for spot in [["living room", Vector3(3, 3, 0)], ["square", Vector3(12, 29, 0)]]:
		var p: Vector3 = spot[1]
		player.global_position = DemoGeo.P(p.x, p.y, p.z)
		for i in 30:
			await get_tree().process_frame
		var t0 := Time.get_ticks_usec()
		var frames := 0
		var worst := 0.0
		var last := t0
		while Time.get_ticks_usec() - t0 < 5_000_000:
			player.rotation.y += 0.004
			await get_tree().process_frame
			var now := Time.get_ticks_usec()
			worst = maxf(worst, (now - last) / 1000.0)
			last = now
			frames += 1
		var secs := (Time.get_ticks_usec() - t0) / 1e6
		res.append("%s: %.0f fps avg, worst frame %.1f ms" % [spot[0], frames / secs, worst])
		for a in OS.get_cmdline_user_args():
			if a.begins_with("--fps-shot="): # what the window showed
				get_viewport().get_texture().get_image().save_png(a.trim_prefix("--fps-shot=") + "_%s.png" % spot[0].replace(" ", "_"))
	print("FPS (%s graphics, window %s px, 3D rendered at %s, screen scale %.1f, vsync off): %s" % ["fine" if fine else "pixel %dp" % pixel_h,
		DisplayServer.window_get_size(), view.size, DisplayServer.screen_get_scale(), " · ".join(res)])
	print("stats: ", world.stats())
	get_tree().quit()

# Screenshot mode for layout review:
#   Godot --path godot -- --shots <dir> [--labels]
# For every room and the town square: <room>_plan.png (orthographic, top
# down, cut just below the ceiling so the room reads like a floor plan)
# and <room>_view1.png / _view2.png (eye level, standing on the room's
# free floor and looking at its focal point), at full resolution, brightly lit; then quits. --labels
# names the furniture kinds and sound sources in the pictures.
#
# room → [plan rect (x0, y0, x1, y1), cut height (world z), floor z,
#         views: [[eye (x, y, z), heading°, pitch°], …]]
# heading: degrees CCW from east (0 = +x, 90 = north, -90 = south).
const SHOT_ROOMS := {
	"living_room": [Rect2(0, 0, 8, 6), 2.6, 0.0, [[Vector3(4, 5.5, 1.6), -90.0, -12.0], [Vector3(6.4, 5.6, 1.6), 210.0, -15.0]]],
	"corridor": [Rect2(3.2, 6, 1.6, 8), 2.3, 0.0, [[Vector3(4, 6.4, 1.6), 90.0, -5.0]]],
	"great_hall": [Rect2(0, 14, 14, 10), 6.9, 0.0, [[Vector3(4, 14.5, 1.6), 50.0, -8.0], [Vector3(1.0, 23.3, 1.7), -13.0, -10.0]]],
	"entrance": [Rect2(20, 28, 2, 6), 2.5, 0.0, [[Vector3(21.0, 28.7, 1.6), 90.0, -14.0]]],
	"club": [Rect2(22, 26, 10, 12), 4.4, 0.0, [[Vector3(22.6, 31, 1.6), -35.0, -8.0], [Vector3(31.5, 37.5, 1.6), 220.0, -12.0]]],
	"old_house": [Rect2(24, 16, 7, 7), 2.7, 0.0, [[Vector3(26.5, 22.6, 1.6), -90.0, -12.0], [Vector3(30.6, 22.6, 1.6), 220.0, -15.0]]],
	"old_house_upper": [Rect2(24, 16, 7, 7), 5.5, 3.0, [[Vector3(25.0, 22.4, 4.6), -40.0, -15.0], [Vector3(30.6, 22.6, 4.6), 220.0, -15.0]]],
	"cathedral": [Rect2(0, 52, 16, 22), 14.5, 0.0, [[Vector3(8, 52.6, 1.7), 90.0, 8.0], [Vector3(6.6, 53.2, 1.7), 85.5, -2.0]]],
	"bunker": [Rect2(34, 6, 6, 8), -0.9, -3.0, [[Vector3(34.6, 7.2, -1.4), 55.0, -10.0], [Vector3(38.5, 6.6, -1.4), 100.0, -12.0]]],
	"bunker_stair": [Rect2(31.4, 6, 2.6, 2), 2.3, -3.0, [[Vector3(31.0, 7, 1.6), 0.0, -35.0]]],
	"kiosk": [Rect2(14.5, 34, 2, 2), 3.5, 0.0, [[Vector3(11.5, 32.5, 1.6), 30.0, 0.0]]],
	"belfry": [Rect2(7, 318, 6, 6), 26.5, 24.2, [[Vector3(18, 305, 1.7), 115.0, 22.0], [Vector3(12.8, 323.8, 25.8), 225.0, -10.0]]],
	"town_square": [Rect2(4, 14, 32, 32), 60.0, 0.0, [[Vector3(12, 29, 1.6), 50.0, 4.0], [Vector3(30, 44, 1.6), 235.0, 0.0]]],
}

var shooting := false

func _shots(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	shooting = true
	player.set_physics_process(false)
	set_process_unhandled_input(false)
	hud.get_parent().visible = false
	# full resolution, linear textures, bright and even (a review light,
	# not the night)
	_size_view(true)
	DemoMats.set_filtering(false)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.85, 0.85, 0.9)
	env.ambient_light_energy = 1.4
	env.fog_enabled = false
	var sun := DirectionalLight3D.new()
	sun.light_energy = 0.9
	sun.rotation_degrees = Vector3(-70, 30, 0)
	add_child(sun)
	for i in 20:
		await get_tree().process_frame
	for room in SHOT_ROOMS:
		var d: Array = SHOT_ROOMS[room]
		var r: Rect2 = d[0]
		# (a) plan: orthographic straight down, cut just under the ceiling
		var c := r.get_center()
		view_cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		var aspect := view.size.x / maxf(1.0, view.size.y)
		view_cam.size = maxf(r.size.y, r.size.x / aspect) * 1.12 + 1.0
		view_cam.near = 0.02
		view_cam.far = 60.0
		view_cam.global_transform = Transform3D(Basis.from_euler(Vector3(-PI / 2, 0, 0)), DemoGeo.P(c.x, c.y, d[1]))
		await _snap(dir.path_join(room + "_plan.png"))
		# (b) eye level from the doorway(s)
		view_cam.projection = Camera3D.PROJECTION_PERSPECTIVE
		view_cam.fov = 75.0
		view_cam.near = 0.05
		view_cam.far = 2000.0
		var k := 1
		for vw in d[3]:
			var e: Vector3 = vw[0]
			var heading := deg_to_rad(vw[1])
			var b := Basis.from_euler(Vector3(deg_to_rad(vw[2]), heading - PI / 2, 0), EULER_ORDER_YXZ)
			view_cam.global_transform = Transform3D(b, DemoGeo.P(e.x, e.y, e.z))
			await _snap(dir.path_join("%s_view%d.png" % [room, k]))
			k += 1
	print("shots: %d rooms → %s" % [SHOT_ROOMS.size(), dir])
	get_tree().quit()

func _snap(path: String) -> void:
	for i in 12:
		await get_tree().process_frame
	view.get_texture().get_image().save_png(path)
	print("shot ", path.get_file())

# The 3D view: a SubViewport sharing this world, rendered small and
# shown by a full-window TextureRect with nearest filtering (the root
# viewport draws only the 2D layer: the upscaled view and the HUD at
# full resolution).
func _viewport() -> void:
	get_viewport().disable_3d = true
	view = SubViewport.new()
	view.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	view.msaa_3d = Viewport.MSAA_DISABLED
	view.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	view.use_taa = false
	add_child(view)
	view_cam = Camera3D.new()
	view.add_child(view_cam)
	view_cam.make_current()
	var layer := CanvasLayer.new()
	layer.layer = -1
	add_child(layer)
	view_rect = TextureRect.new()
	view_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	view_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view_rect.stretch_mode = TextureRect.STRETCH_SCALE
	view_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	view_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	view_rect.texture = view.get_texture()
	layer.add_child(view_rect)
	get_viewport().size_changed.connect(func() -> void: _size_view(fine))

# Pixel mode: pixel_h rows, as many columns as the window's aspect gives
# (the image fills the window); fine/full: the window's own pixels.
func _size_view(full: bool) -> void:
	var win := get_viewport().get_visible_rect().size
	if full:
		view.size = Vector2i(win)
	else:
		view.size = Vector2i(roundi(pixel_h * win.x / maxf(1.0, win.y)), pixel_h)

func _apply_graphics() -> void:
	_size_view(fine)
	view.msaa_3d = Viewport.MSAA_2X if fine else Viewport.MSAA_DISABLED
	view_rect.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR if fine else CanvasItem.TEXTURE_FILTER_NEAREST
	env.ssao_enabled = fine
	env.ssil_enabled = fine
	env.volumetric_fog_enabled = fine
	view.positional_shadow_atlas_size = 4096 if fine else 2048
	var q := RenderingServer.SHADOW_QUALITY_SOFT_MEDIUM if fine else RenderingServer.SHADOW_QUALITY_HARD
	RenderingServer.directional_soft_shadow_filter_set_quality(q)
	RenderingServer.positional_soft_shadow_filter_set_quality(q)
	DemoMats.set_filtering(not fine)

func _environment() -> void:
	env = Environment.new()
	var sky := Sky.new()
	var pano := PanoramaSkyMaterial.new()
	var img := Image.load_from_file(ProjectSettings.globalize_path(SKY))
	if img != null and not img.is_empty():
		pano.panorama = ImageTexture.create_from_image(img)
	pano.energy_multiplier = 0.55
	sky.sky_material = pano
	env.sky = sky
	env.background_mode = Environment.BG_SKY
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 0.55
	env.ambient_light_sky_contribution = 0.7
	env.ambient_light_color = Color(0.32, 0.38, 0.55)
	env.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.tonemap_exposure = 1.25
	env.ssao_radius = 1.2
	env.ssao_intensity = 2.0
	env.glow_enabled = true
	env.glow_intensity = 0.7
	env.glow_bloom = 0.04
	env.glow_hdr_threshold = 0.9
	env.fog_enabled = true
	env.fog_light_color = Color(0.09, 0.11, 0.17)
	env.fog_density = 0.0012
	env.fog_sky_affect = 0.15
	env.fog_aerial_perspective = 0.4
	env.volumetric_fog_density = 0.006
	env.volumetric_fog_albedo = Color(0.7, 0.75, 0.85)
	env.volumetric_fog_anisotropy = 0.5
	env.volumetric_fog_length = 70.0
	env.volumetric_fog_sky_affect = 0.0
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	# the moon: a cool key light with long soft shadows
	var moon := DirectionalLight3D.new()
	moon.light_color = Color(0.62, 0.72, 1.0)
	moon.light_energy = 0.45
	moon.light_volumetric_fog_energy = 0.6
	moon.shadow_enabled = true
	moon.shadow_blur = 1.5
	moon.directional_shadow_max_distance = 120.0
	moon.directional_shadow_mode = DirectionalLight3D.SHADOW_ORTHOGONAL
	moon.rotation_degrees = Vector3(-38, 140, 0)
	add_child(moon)

func _hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	hud.position = Vector2(14, 10)
	hud.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0))
	hud.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.8))
	hud.add_theme_constant_override("outline_size", 4)
	layer.add_child(hud)
	hint = Label.new()
	hint.text = "click: capture mouse / throw · WASD walk · Shift run · Space jump · C crouch · F fly · E door · B bells · V whisper · R rain · L labels · Q sound quality · G graphics · [ ] pixel height · H hud"
	hint.add_theme_color_override("font_color", Color(0.75, 0.8, 0.9, 0.85))
	hint.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.8))
	hint.add_theme_constant_override("outline_size", 4)
	hint.anchor_top = 1.0
	hint.anchor_bottom = 1.0
	hint.position = Vector2(14, -34)
	layer.add_child(hint)
	var cross := Label.new()
	cross.text = "·"
	cross.set_anchors_preset(Control.PRESET_CENTER)
	cross.add_theme_font_size_override("font_size", 28)
	layer.add_child(cross)

func _labels() -> void:
	for n in sounds.sources:
		var s: OmgSource = sounds.sources[n]
		var l := Label3D.new()
		l.text = n
		l.font_size = 48
		l.pixel_size = 0.0008
		l.fixed_size = true
		l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		l.modulate = Color(1.0, 0.85, 0.5, 0.9)
		l.no_depth_test = true
		s.add_child(l)
		l.position = Vector3(0, 0.6, 0)
		l.visible = false
		labels.append(l)

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed and e.button_index == MOUSE_BUTTON_LEFT:
		if Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		else:
			sounds.throw_ball()
	elif e is InputEventKey and e.pressed and not e.echo:
		match e.keycode:
			KEY_ESCAPE:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
			KEY_E:
				var p := player.plan()
				level.toggle_nearest_door(p.x, p.y)
			KEY_B:
				sounds.ring_bells()
			KEY_V:
				sounds.whisper()
			KEY_R:
				rain_name = sounds.cycle_rain()
			KEY_L:
				for l in labels:
					l.visible = not l.visible
			KEY_H:
				hud.visible = not hud.visible
				hint.visible = hud.visible
			KEY_Q:
				quality = (quality + 1) % 4 # medium → high → ultra → low
				world.quality = quality
			KEY_G:
				fine = not fine
				_apply_graphics()
			KEY_BRACKETLEFT:
				pixel_h = HEIGHTS[maxi(0, HEIGHTS.find(pixel_h) - 1)]
				_apply_graphics()
			KEY_BRACKETRIGHT:
				pixel_h = HEIGHTS[mini(HEIGHTS.size() - 1, HEIGHTS.find(pixel_h) + 1)]
				_apply_graphics()

func _process(dt: float) -> void:
	# the pixel view's camera follows the player's
	if not shooting:
		view_cam.global_transform = player.cam.global_transform
		view_cam.fov = player.cam.fov
		view_cam.near = player.cam.near
		view_cam.far = player.cam.far
	level.update_doors(dt)
	# candle flicker in the apse, club wash breathing to the beat
	t_flicker += dt
	for c in level.markers.get("candles", []):
		(c as OmniLight3D).light_energy = 3.6 + 0.5 * sin(t_flicker * 13.0 + c.position.x) + 0.3 * sin(t_flicker * 7.3)
	var i := 0
	for l in level.markers.get("club_lights", []):
		(l as OmniLight3D).light_energy = 2.0 + 1.6 * maxf(0.0, sin(t_flicker * TAU * 124.0 / 60.0 + i * PI / 2.0))
		i += 1
	if hud.visible:
		var p := player.plan()
		var room: String = DemoLevel.room_at(p.x, p.y, p.z + 0.5)[0]
		var s := world.stats()
		var q: String = ["low", "medium", "high", "ultra"][quality]
		hud.text = "%d fps · %s graphics\n%s · RT60 %.2f s · enclosure %.0f%%\nrain %s · %s%s\nsim %.1f ms (%s, %s) · render load %.0f%% · budget %d paths / %d hrtf / %d voices · %d/%d sources rendering · %d paths · late %d" % [
			Engine.get_frames_per_second(), "fine" if fine else "pixel %dp" % pixel_h,
			room, s.get("rt60", 0.0), 100.0 * s.get("enclosure", 0.0), rain_name,
			"flying" if player.flying else "walking", " · crouched" if player.crouch > 0.5 else "",
			s.get("tick_ms", 0.0), q, "gpu" if s.get("gpu", false) else "cpu",
			100.0 * s.get("load", 0.0), s.get("taps", 0), s.get("points", 0), s.get("voices", 0),
			s.get("rendering", 0), s.get("sources", 0), s.get("paths", 0), s.get("late", 0)]
