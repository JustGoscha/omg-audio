# PBR material library for the demo level. Textures are CC0 Poly Haven
# sets in ../assets/textures (diffuse + OpenGL normal + AO/rough/metal),
# loaded at runtime from the shared assets folder (no Godot import step).
# Level geometry carries world-meter UVs, so `tile` is meters per repeat.
#
# Every surface material DECLARES its acoustics: an `omg_material` meta
# naming an OmgWorld preset (brick looks like brick and sounds like
# brick). OmgWorld reads it from any mesh using the material, so a
# texture cannot drift from its sound; DemoGeo.B defaults to it too.
class_name DemoMats
extends RefCounted

const TEX_DIR := "res://../assets/textures/"

static var _cache := {}
static var _tex := {}

static func _texture(file: String) -> Texture2D:
	if _tex.has(file):
		return _tex[file]
	var img := Image.load_from_file(ProjectSettings.globalize_path(TEX_DIR + file))
	var t: Texture2D = null
	if img != null and not img.is_empty():
		img.generate_mipmaps()
		t = ImageTexture.create_from_image(img)
	else:
		push_warning("demo: missing texture " + file)
	_tex[file] = t
	return t

# A textured PBR surface: `set_name` is the Poly Haven asset name.
static func pbr(set_name: String, tile: float, tint := Color.WHITE, rough := -1.0, acoustic := "") -> BaseMaterial3D:
	var key := "%s|%s|%s|%s|%s" % [set_name, tile, tint, rough, acoustic]
	if _cache.has(key):
		return _cache[key]
	var m := ORMMaterial3D.new()
	m.albedo_texture = _texture(set_name + "_diff_1k.jpg")
	m.albedo_color = tint
	m.normal_enabled = true
	m.normal_texture = _texture(set_name + "_nor_gl_1k.jpg")
	m.normal_scale = 1.0
	m.orm_texture = _texture(set_name + "_arm_1k.jpg")
	if rough >= 0.0:
		m.roughness = rough
	m.uv1_scale = Vector3(1.0 / tile, 1.0 / tile, 1.0)
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	_declare(m, set_name, acoustic)
	_cache[key] = m
	return m

# name the material (for the audit) and declare its acoustics
static func _declare(m: Material, name: String, acoustic: String) -> void:
	m.resource_name = name
	if acoustic != "":
		m.set_meta("omg_material", acoustic)

static func acoustic_of(m: Material) -> String:
	return m.get_meta("omg_material", "") if m != null else ""

# Pixel look: nearest-filtered textures (mipmapped, so the distance
# stays calm); the fine look: anisotropic linear.
static func set_filtering(nearest: bool) -> void:
	for m in _cache.values():
		if m is BaseMaterial3D:
			m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS if nearest \
				else BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC

static func plain(color: Color, rough := 0.8, metal := 0.0, acoustic := "") -> StandardMaterial3D:
	var key := "plain|%s|%s|%s|%s" % [color, rough, metal, acoustic]
	if _cache.has(key):
		return _cache[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	m.metallic = metal
	_declare(m, "plain " + color.to_html(false), acoustic)
	_cache[key] = m
	return m

static func glow(color: Color, energy := 2.0) -> StandardMaterial3D:
	var key := "glow|%s|%s" % [color, energy]
	if _cache.has(key):
		return _cache[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.emission_enabled = true
	m.emission = color
	m.emission_energy_multiplier = energy
	_declare(m, "glow " + color.to_html(false), "")
	_cache[key] = m
	return m

static func glass(tint := Color(0.75, 0.85, 0.9, 0.18)) -> StandardMaterial3D:
	var key := "glass|%s" % [tint]
	if _cache.has(key):
		return _cache[key]
	var m := StandardMaterial3D.new()
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = tint
	m.roughness = 0.04
	m.metallic = 0.2
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	_declare(m, "glass", "glass")
	_cache[key] = m
	return m

# ---- the level's palette (texture → acoustics) --------------------------------------------------

static func brick() -> BaseMaterial3D: return pbr("castle_brick_02_red", 2.2, Color.WHITE, -1.0, "brick")
static func sandstone(tile := 2.6, tint := Color(0.92, 0.88, 0.82)) -> BaseMaterial3D: return pbr("white_sandstone_bricks_03", tile, tint, -1.0, "stone")
# plaster is a finish: it sounds like the wall it covers (brick by default)
static func plaster(tint := Color(0.93, 0.9, 0.84), on := "plaster") -> BaseMaterial3D: return pbr("plastered_wall_04", 2.5, tint, -1.0, on)
static func concrete(tint := Color(0.85, 0.85, 0.85)) -> BaseMaterial3D: return pbr("brushed_concrete", 3.0, tint, -1.0, "concrete")
static func blocks(tint := Color(0.8, 0.8, 0.82)) -> BaseMaterial3D: return pbr("concrete_block_wall", 2.4, tint, -1.0, "concrete")
static func slate() -> BaseMaterial3D: return pbr("roof_slates_02", 2.0, Color(0.75, 0.78, 0.85), -1.0, "stone")
static func cobbles(tile := 2.4, tint := Color.WHITE) -> BaseMaterial3D: return pbr("cobblestone_floor_08", tile, tint, -1.0, "paving")
static func grass() -> BaseMaterial3D:
	# no ORM map: a lawn is fully rough (the packed map's gloss made the
	# plain mirror the sky at grazing angles)
	if _cache.has("grass"):
		return _cache["grass"]
	var m := StandardMaterial3D.new()
	m.albedo_texture = _texture("leafy_grass_diff_1k.jpg")
	m.albedo_color = Color(0.62, 0.72, 0.58)
	m.normal_enabled = true
	m.normal_texture = _texture("leafy_grass_nor_gl_1k.jpg")
	m.roughness = 1.0
	m.specular_mode = BaseMaterial3D.SPECULAR_DISABLED
	m.uv1_scale = Vector3(1.0 / 3.5, 1.0 / 3.5, 1.0)
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	_declare(m, "leafy_grass", "grass")
	_cache["grass"] = m
	return m
static func asphalt() -> BaseMaterial3D: return pbr("asphalt_02", 3.0, Color(0.7, 0.7, 0.72), -1.0, "asphalt")
static func planks(tint := Color.WHITE, acoustic := "wood") -> BaseMaterial3D: return pbr("brown_planks_05", 2.0, tint, -1.0, acoustic)
static func parquet() -> BaseMaterial3D: return pbr("herringbone_parquet", 1.6, Color.WHITE, -1.0, "wood")
static func floorboards() -> BaseMaterial3D: return pbr("wood_floor", 2.2, Color.WHITE, -1.0, "wood")
static func marble(tile := 2.0, tint := Color(0.85, 0.85, 0.88)) -> BaseMaterial3D: return pbr("marble_tiles", tile, tint, -1.0, "marble")
static func fabric(c: Color) -> StandardMaterial3D: return plain(c, 0.95, 0.0, "soft")
static func carpet(c: Color) -> StandardMaterial3D: return plain(c, 1.0, 0.0, "carpet")
static func metal(c := Color(0.55, 0.57, 0.6)) -> StandardMaterial3D: return plain(c, 0.35, 0.9, "metal")
