# First-person walker, controls as the web demo: click captures the mouse,
# WASD walk, Shift sprint (keeps accelerating), Space jump, C crouch,
# F fly (W flies where you look, Space up, C down, Shift up to 150 m/s).
class_name DemoPlayer
extends CharacterBody3D

const WALK := 3.0
const FLY := 8.0
const FLY_TOP := 150.0
const EYE := 1.6
const GRAVITY := 9.8

var cam: Camera3D
var shape: CollisionShape3D
var capsule: CapsuleShape3D
var pitch := 0.0
var crouch := 0.0
var sprint_v := 0.0
var flying := false
var want_jump := false
# per-frame outputs for the sound logic
var stepped := 0.0      # horizontal meters walked this frame (on the ground)
var landed := false     # touched down this frame after a fall/jump
var was_air := false

func _init() -> void:
	capsule = CapsuleShape3D.new()
	capsule.radius = 0.22
	capsule.height = 1.75
	shape = CollisionShape3D.new()
	shape.shape = capsule
	shape.position = Vector3(0, 0.875, 0)
	add_child(shape)
	cam = Camera3D.new()
	cam.position = Vector3(0, EYE, 0)
	cam.fov = 72.0
	cam.near = 0.05
	cam.far = 6000.0
	add_child(cam)
	var ears := OmgListener.new()
	cam.add_child(ears)
	_body()
	floor_max_angle = deg_to_rad(62.0)
	floor_snap_length = 0.45
	safe_margin = 0.02

# A blocky body under the eyes (seen looking down): legs, torso, arms —
# no head (the eyes are in it). Not acoustic: the engine hears from one
# point in the middle of the head, which a head around it would wall in.
var body: Node3D

func _body() -> void:
	body = Node3D.new()
	body.set_meta("omg_material", "none")
	add_child(body)
	var shirt := StandardMaterial3D.new()
	shirt.albedo_color = Color(0.22, 0.3, 0.42)
	shirt.roughness = 0.9
	var cloth := StandardMaterial3D.new()
	cloth.albedo_color = Color(0.13, 0.13, 0.16)
	cloth.roughness = 0.9
	# (x right, y up, z back: the player faces −z)
	for b in [[Vector3(-0.17, 0.0, -0.08), Vector3(-0.03, 0.84, 0.08), cloth], [Vector3(0.03, 0.0, -0.08), Vector3(0.17, 0.84, 0.08), cloth],
			[Vector3(-0.19, 0.84, -0.12), Vector3(0.19, 1.42, 0.12), shirt],
			[Vector3(-0.23, 0.86, -0.06), Vector3(-0.19, 1.4, 0.06), shirt], [Vector3(0.19, 0.86, -0.06), Vector3(0.23, 1.4, 0.06), shirt]]:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		var a: Vector3 = b[0]
		var c: Vector3 = b[1]
		bm.size = c - a
		mi.mesh = bm
		mi.material_override = b[2]
		mi.position = (a + c) * 0.5
		body.add_child(mi)

# heading as the web uses it: world math angle, π/2 = facing north (+y)
func heading() -> float:
	return rotation.y + PI / 2.0

func set_heading(h: float) -> void:
	rotation.y = h - PI / 2.0

# plan-space feet position
func plan() -> Vector3:
	return Vector3(global_position.x, -global_position.z, global_position.y)

func look_dir() -> Vector3:
	return -cam.global_transform.basis.z

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotation.y -= e.relative.x * 0.0025
		pitch = clampf(pitch - e.relative.y * 0.0025, -1.4, 1.4)
	elif e is InputEventKey and e.pressed and not e.echo:
		if e.keycode == KEY_SPACE:
			want_jump = true
		elif e.keycode == KEY_F:
			toggle_fly()

func toggle_fly() -> void:
	flying = not flying
	velocity = Vector3.ZERO
	shape.disabled = flying

func _boost(dt: float) -> float:
	if Input.is_key_pressed(KEY_SHIFT):
		if sprint_v < WALK * 0.8:
			sprint_v = WALK * 0.8
		else:
			var acc := maxf(1.0, 0.46 * sprint_v) if flying else 1.0
			sprint_v = minf(sprint_v + acc * dt, FLY_TOP if flying else 27.0)
	else:
		sprint_v = maxf(0.0, sprint_v - 30.0 * dt)
	return sprint_v

func _input_vec() -> Vector2:
	var f := 0.0
	var r := 0.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP): f += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN): f -= 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT): r += 1.0
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT): r -= 1.0
	var v := Vector2(f, r)
	return v.normalized() if v.length() > 1.0 else v

func _physics_process(dt: float) -> void:
	stepped = 0.0
	landed = false
	var inp := _input_vec()
	var fwd := -global_transform.basis.z
	var right := global_transform.basis.x
	if flying:
		crouch = maxf(0.0, crouch - dt * 8.0)
		want_jump = false
		var climb := (1.0 if Input.is_key_pressed(KEY_SPACE) else 0.0) - (1.0 if Input.is_key_pressed(KEY_C) or Input.is_key_pressed(KEY_CTRL) else 0.0)
		var speed := FLY + _boost(dt)
		var dir := look_dir() * inp.x + right * inp.y + Vector3.UP * climb
		global_position += dir * speed * dt
		global_position.y = clampf(global_position.y, -3.0, 400.0)
	else:
		var want_c := 1.0 if Input.is_key_pressed(KEY_C) or Input.is_key_pressed(KEY_CTRL) else 0.0
		crouch += (want_c - crouch) * minf(1.0, dt * 8.0)
		capsule.height = 1.75 - 0.6 * crouch
		shape.position.y = capsule.height * 0.5
		var speed := (WALK + (_boost(dt) if inp != Vector2.ZERO else 0.0)) * (1.0 - 0.45 * crouch)
		var hv := (fwd * inp.x + right * inp.y) * speed
		velocity.x = hv.x
		velocity.z = hv.z
		if is_on_floor():
			if want_jump and crouch < 0.4:
				velocity.y = 4.2
			elif velocity.y < 0.0:
				velocity.y = 0.0
		velocity.y -= GRAVITY * dt
		want_jump = false
		var before := global_position
		move_and_slide()
		var on := is_on_floor()
		if on:
			stepped = Vector2(global_position.x - before.x, global_position.z - before.z).length()
			if was_air:
				landed = true
		was_air = not on
		# never fall out of the world
		if global_position.y < -20.0:
			global_position = Vector3(3, 0.1, -3)
			velocity = Vector3.ZERO
	cam.position.y = EYE - 0.7 * crouch
	body.scale.y = (EYE - 0.7 * crouch) / EYE
	cam.rotation.x = pitch
