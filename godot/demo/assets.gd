# Where the demo's assets live: the repository's assets/ beside the
# project when run from the checkout (res://../assets/), or packed into the
# .pck as res://assets/ in an exported build (tools/build_godot_web.sh and
# build_godot_mac.sh copy them in for the export).
class_name DemoAssets
extends RefCounted

# packed (web, or any exported build: feature "template") vs the checkout
static var packed: bool = OS.has_feature("web") or OS.has_feature("template")
static var root: String = "res://assets/" if packed else "res://../assets/"

# A path Image.load_from_file / FileAccess can open: on disk natively
# (res://.. leaves the project), inside the .pck on the web.
static func file(rel: String) -> String:
	var p := root + rel
	return p if packed else ProjectSettings.globalize_path(p)
