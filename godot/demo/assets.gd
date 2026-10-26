# Where the demo's assets live: the repository's assets/ beside the
# project when run from the checkout (res://../assets/), or packed into the
# .pck as res://assets/ in the web export (tools/build_godot_web.sh copies
# them in for the export).
class_name DemoAssets
extends RefCounted

static var root: String = "res://assets/" if OS.has_feature("web") else "res://../assets/"

# A path Image.load_from_file / FileAccess can open: on disk natively
# (res://.. leaves the project), inside the .pck on the web.
static func file(rel: String) -> String:
	var p := root + rel
	return p if OS.has_feature("web") else ProjectSettings.globalize_path(p)
