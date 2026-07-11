// First-person walking simulator over the omg-audio engine.
// Desktop: pointer-lock mouse look + WASD. Mobile: left-thumb joystick to
// walk, device orientation (or right-half drag) to look.
//
// World coords (match the Rust scene): x/y ground plane, z up.
// three.js coords: X = wx, Y = wz (height), Z = -wy.

import * as THREE from './vendor/three.module.js';

// Scene geometry mirrored from crates/omg-scene/src/walkthrough.rs (visuals
// + movement only; the acoustic truth lives in the wasm sim).
const ROOMS = [
  { name: 'Living Room', min: [0, 0], max: [8, 6], h: 2.7, floor: 0x4a3b2e },
  { name: 'Corridor', min: [3.2, 6], max: [4.8, 14], h: 2.4, floor: 0x323844 },
  { name: 'Great Hall', min: [0, 14], max: [14, 24], h: 7, floor: 0x2a3140 },
  { name: 'Entrance', min: [20, 28], max: [22, 34], h: 2.6, floor: 0x3a2a3a },
  { name: 'Club', min: [22, 26], max: [32, 38], h: 4.5, floor: 0x2e1f33 },
  { name: 'Old House', min: [24, 16], max: [31, 23], h: 5.6, floor: 0x39322a }, // 2 storeys
  { name: 'Colonnade', min: [16, 15], max: [16.5, 21], h: 2.5, solid: true },
  { name: 'Kiosk', min: [14.5, 34], max: [16.5, 36], h: 2.7, solid: true },
  { name: 'Outside', min: [-8, -8], max: [42, 46], outdoor: true, floor: 0x1c2a20 },
];
// axis: 0 = opening in an x=const wall, 1 = opening in a y=const wall
const DOORS = [
  { pos: [4, 6], axis: 1 },
  { pos: [4, 14], axis: 1 },
  { pos: [7, 24], axis: 1 },
  { pos: [20, 31], axis: 0 },
  { pos: [22, 31], axis: 0 },
  { pos: [26.5, 23], axis: 1 }, // old house front door
];
const DOOR_HALF = 0.55;
const DOOR_H = 2.1;
// windows: glass — visible, not walkable; direct sound passes lightly damped
const WINDOWS = [
  { pos: [3.0, 0.0], axis: 1, hw: 1.3 },
  { pos: [32.0, 32.0], axis: 0, hw: 1.8 },
  { pos: [26.0, 38.0], axis: 1, hw: 1.8 },
  { pos: [29.3, 23.0], axis: 1, hw: 1.4 }, // house ground floor → square
  { pos: [24.0, 19.5], axis: 0, hw: 1.4 }, // house ground floor → hall
];
const SOURCES = [
  { name: 'music', pos: [2, 3], color: 0xffaa3c },
  { name: 'voice', pos: [10.5, 20.5], color: 0x6ee0a0 },
  { name: 'club', pos: [27, 32], color: 0xff5a9e,
    emitters: [[23.5, 27.5], [30.5, 27.5], [23.5, 36.5], [30.5, 36.5]] },
];
const MARGIN = 0.35;
const EYE = 1.6;

const state = {
  projs: [],
  fx: null,
  meters: { l: 0, r: 0, agc: 1, tts: 0, pts: 0, hist: [] },
  pose: { x: 3, y: 3 },
  heading: Math.PI / 2, // world math angle; π/2 = facing +y (into the scene)
  pitch: 0,
  keys: new Set(),
  joy: null,
  look: null,
  orientation: null,
  orientationOffset: null,
  simState: null,
  running: false,
  rainLevel: 0, // index into RAIN_LEVELS
  chanHist: [], // per-channel meter frames, ~1 s
  mixerRows: null,
};

// simulated weather: intensity targets the engine ramps toward (~6 s),
// so rain starts and stops like weather, not like a fader
const RAIN_LEVELS = [
  { label: '☂ off', intensity: 0.0 },
  { label: '☂ drizzle', intensity: 0.3 },
  { label: '☂ rain', intensity: 0.65 },
  { label: '☂ downpour', intensity: 1.0 },
];

function cycleRain() {
  state.rainLevel = (state.rainLevel + 1) % RAIN_LEVELS.length;
  const lvl = RAIN_LEVELS[state.rainLevel];
  document.getElementById('rain').textContent = lvl.label;
  if (state.node) state.node.port.postMessage({ type: 'rain', intensity: lvl.intensity });
}

// ------------------------------------------------------------- mixer panel
//
// Source faders are POWER faders on a real SPL scale (dB SPL @ 1 m):
// 20 ≈ a needle drop, 60 ≈ speech, 90 ≈ fortissimo piano, 110 ≈ club PA,
// 130 ≈ jet engine. The engine's calibration anchor is gain 1.0 ≈ 90 dB
// SPL, and each source's authored gain gives its baseline; the fader sets
// the REAL emitted energy: gain = 10^((SPL − baseline) / 20).
// Ambience/rain/master are trim faders in plain dB.
const MIXER = [
  { name: 'music', srcs: [0], base: 90, meters: [0], spl: true },
  { name: 'voice', srcs: [1], base: 84, meters: [1], spl: true },
  { name: 'club', srcs: [2], base: 104, meters: [2], spl: true },
  { name: 'balls', srcs: [3, 4, 5], base: 89, meters: [3, 4, 5], spl: true },
  { name: 'ambience', target: 'ambient', meters: [6] },
  { name: 'rain', target: 'rainGain', meters: [7] },
  { name: 'master', target: 'master', meters: 'lr' },
];
const SPL_MIN = 20, SPL_MAX = 130, TRIM_MIN = -30, TRIM_MAX = 12;

function buildMixer() {
  const panel = document.getElementById('mixer');
  panel.innerHTML = '<div class="scale"><span>20</span><span>needle</span>' +
    '<span>60 speech</span><span>club 110</span><span>jet 130</span></div>';
  state.mixerRows = MIXER.map((ch) => {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('div');
    label.innerHTML = ch.name + '<br><span class="spl"></span>';
    const mid = document.createElement('div');
    const fader = document.createElement('input');
    fader.type = 'range';
    fader.min = 0; fader.max = 1000; fader.className = 'fader';
    fader.value = ch.spl
      ? ((ch.base - SPL_MIN) / (SPL_MAX - SPL_MIN)) * 1000
      : ((0 - TRIM_MIN) / (TRIM_MAX - TRIM_MIN)) * 1000;
    const meter = document.createElement('div');
    meter.className = 'meter';
    meter.innerHTML = '<div class="rms"></div><div class="pk"></div>';
    mid.append(fader, meter);
    const lvl = document.createElement('div');
    lvl.className = 'lvl';
    lvl.textContent = '−∞';
    row.append(label, mid, lvl);
    panel.append(row);
    const apply = () => {
      const v = fader.value / 1000;
      let gain, text;
      if (ch.spl) {
        const spl = SPL_MIN + v * (SPL_MAX - SPL_MIN);
        gain = 10 ** ((spl - ch.base) / 20);
        text = `${spl.toFixed(0)} dB SPL`;
      } else {
        const db = TRIM_MIN + v * (TRIM_MAX - TRIM_MIN);
        gain = v === 0 ? 0 : 10 ** (db / 20);
        text = `${db >= 0 ? '+' : ''}${db.toFixed(0)} dB`;
      }
      label.querySelector('.spl').textContent = text;
      if (state.node) {
        state.node.port.postMessage({ type: 'mixer', target: ch.target, srcs: ch.srcs, gain });
      }
    };
    fader.oninput = apply;
    fader.onpointerup = () => fader.blur();
    apply();
    return { ch, row, lvl, rms: meter.querySelector('.rms'), pk: meter.querySelector('.pk') };
  });
}

// meter history: ~43 frames ≈ 1 s
function updateMixerMeters() {
  if (!state.mixerRows || document.getElementById('mixer').hidden) return;
  const hist = state.chanHist;
  if (!hist.length) return;
  for (const r of state.mixerRows) {
    let rms = 0, pk = 0;
    if (r.ch.meters === 'lr') {
      // master: current output block peaks (post-AGC, post-master)
      rms = (state.meters.l + state.meters.r) / 2;
      pk = Math.max(state.meters.l, state.meters.r);
    } else {
      let n = 0;
      for (const frame of hist) {
        for (const m of r.ch.meters) {
          rms += frame[m * 2 + 1] ** 2;
          pk = Math.max(pk, frame[m * 2]);
          n++;
        }
      }
      rms = Math.sqrt(rms / Math.max(1, n));
    }
    const toDb = (x) => Math.max(-60, 20 * Math.log10(x + 1e-9));
    const rdb = toDb(rms);
    const pdb = toDb(pk);
    r.rms.style.width = `${((rdb + 60) / 60) * 100}%`;
    r.pk.style.left = `${((pdb + 60) / 60) * 100}%`;
    r.lvl.textContent = rdb <= -59.5 ? '−∞' : `${rdb.toFixed(1)} dB`;
  }
}

// ------------------------------------------------------------ walkability

// Collision: actual wall segments (built alongside the 3D walls; door
// gaps are simply absent). Room-union tests stopped working once Outside
// surrounds all buildings — everything was "inside a room".
const COLLIDERS = []; // { axis, plane, lo, hi }  axis 0: x=plane, 1: y=plane
const PLAYER_R = 0.22;
const WORLD = { min: [-7.6, -7.6], max: [41.6, 45.6] };

function crossesWall(x0, y0, x1, y1, z = 1.6) {
  for (const c of COLLIDERS) {
    if (z > c.h || z < (c.zlo || 0)) continue; // above wall / below upper storey
    const [p0, p1] = c.axis === 0 ? [x0, x1] : [y0, y1];
    const d = p1 - p0;
    if (Math.abs(d) < 1e-9) {
      // moving parallel: keep a standoff from the wall plane
      if (Math.abs(p1 - c.plane) < PLAYER_R) {
        const along = c.axis === 0 ? y1 : x1;
        if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) continue; // grazing ok
      }
      continue;
    }
    const t = (c.plane - p0) / d;
    if (t < 0 || t > 1) {
      // also block ending too close to the plane
      if (Math.abs(p1 - c.plane) > PLAYER_R) continue;
      const along = c.axis === 0 ? y1 : x1;
      if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) return true;
      continue;
    }
    const along = c.axis === 0 ? y0 + t * (y1 - y0) : x0 + t * (x1 - x0);
    if (along > c.lo - PLAYER_R && along < c.hi + PLAYER_R) return true;
  }
  return false;
}

function walkableMove(x0, y0, x1, y1) {
  if (x1 < WORLD.min[0] || x1 > WORLD.max[0] || y1 < WORLD.min[1] || y1 > WORLD.max[1]) return false;
  return !crossesWall(x0, y0, x1, y1);
}

// ------------------------------------------------------------ three scene

const glCanvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d12);
scene.fog = new THREE.Fog(0x0a0d12, 24, 100);
const camera = new THREE.PerspectiveCamera(72, 1, 0.05, 120);
camera.rotation.order = 'YXZ';

const v3 = (wx, wy, wz) => new THREE.Vector3(wx, wz, -wy);

function fit() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', fit);
fit();

scene.add(new THREE.HemisphereLight(0x8899bb, 0x1a1410, 0.9));
scene.add(new THREE.AmbientLight(0x404860, 0.5));

const wallMat = new THREE.MeshLambertMaterial({ color: 0x39465c });
const glassMat = new THREE.MeshLambertMaterial({
  color: 0x7ac8ff, transparent: true, opacity: 0.28, side: THREE.DoubleSide,
});
const edgeMat = new THREE.LineBasicMaterial({ color: 0x6a7f9f });
const ceilMat = new THREE.MeshLambertMaterial({ color: 0x232b38, side: THREE.DoubleSide });

function addBox(cx, cy, cz, sx, sy, sz) {
  const g = new THREE.BoxGeometry(sx, sz, sy); // (wx, wz, wy) sizes
  const m = new THREE.Mesh(g, wallMat);
  m.position.copy(v3(cx, cy, cz));
  scene.add(m);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat);
  e.position.copy(m.position);
  scene.add(e);
}

// Wall along y = const, spanning x0..x1, with door cuts (axis-1 doors)
// and window insets (glass panes; collision keeps the full span solid).
function wallY(y, x0, x1, h) {
  let spans = [[x0, x1]];
  for (const { pos: [dx, dy], axis } of DOORS) {
    if (axis !== 1 || Math.abs(dy - y) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      dx - DOOR_HALF > a && dx + DOOR_HALF < b
        ? [[a, dx - DOOR_HALF], [dx + DOOR_HALF, b]]
        : [[a, b]],
    );
    if (dx - DOOR_HALF > x0 && dx + DOOR_HALF < x1 && h > DOOR_H) {
      addBox(dx, y, (DOOR_H + h) / 2, 2 * DOOR_HALF, 0.15, h - DOOR_H); // lintel
    }
  }
  for (const { pos: [wx, wy], axis, hw } of WINDOWS) {
    if (axis !== 1 || Math.abs(wy - y) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      wx - hw > a && wx + hw < b ? [[a, wx - hw], [wx + hw, b]] : [[a, b]],
    );
    if (wx - hw > x0 && wx + hw < x1) {
      addBox(wx, y, 0.45, 2 * hw, 0.15, 0.9);              // sill
      addBox(wx, y, (2.2 + h) / 2, 2 * hw, 0.15, h - 2.2); // head
      const pane = new THREE.Mesh(new THREE.BoxGeometry(2 * hw, 1.3, 0.06), glassMat);
      pane.position.copy(v3(wx, y, 1.55));
      scene.add(pane);
    }
  }
  for (const [a, b] of spans) {
    addBox((a + b) / 2, y, h / 2, b - a, 0.15, h);
  }
  COLLIDERS.push({ axis: 1, plane: y, lo: x0, hi: x1, h });
  // reopen the doorway gaps in the collider
  for (const { pos: [dx, dy], axis } of DOORS) {
    if (axis === 1 && Math.abs(dy - y) < 0.01 && dx > x0 && dx < x1) {
      openColliderGap(1, y, dx);
    }
  }
}

// Wall along x = const, spanning y0..y1, with door cuts (axis-0 doors)
// and window insets.
function wallX(x, y0, y1, h) {
  let spans = [[y0, y1]];
  for (const { pos: [dx, dy], axis } of DOORS) {
    if (axis !== 0 || Math.abs(dx - x) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      dy - DOOR_HALF > a && dy + DOOR_HALF < b
        ? [[a, dy - DOOR_HALF], [dy + DOOR_HALF, b]]
        : [[a, b]],
    );
    if (dy - DOOR_HALF > y0 && dy + DOOR_HALF < y1 && h > DOOR_H) {
      addBox(x, dy, (DOOR_H + h) / 2, 0.15, 2 * DOOR_HALF, h - DOOR_H); // lintel
    }
  }
  for (const { pos: [wx, wy], axis, hw } of WINDOWS) {
    if (axis !== 0 || Math.abs(wx - x) > 0.01) continue;
    spans = spans.flatMap(([a, b]) =>
      wy - hw > a && wy + hw < b ? [[a, wy - hw], [wy + hw, b]] : [[a, b]],
    );
    if (wy - hw > y0 && wy + hw < y1) {
      addBox(x, wy, 0.45, 0.15, 2 * hw, 0.9);
      addBox(x, wy, (2.2 + h) / 2, 0.15, 2 * hw, h - 2.2);
      const pane = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.3, 2 * hw), glassMat);
      pane.position.copy(v3(x, wy, 1.55));
      scene.add(pane);
    }
  }
  for (const [a, b] of spans) {
    addBox(x, (a + b) / 2, h / 2, 0.15, b - a, h);
  }
  COLLIDERS.push({ axis: 0, plane: x, lo: y0, hi: y1, h });
  for (const { pos: [dx, dy], axis } of DOORS) {
    if (axis === 0 && Math.abs(dx - x) < 0.01 && dy > y0 && dy < y1) {
      openColliderGap(0, x, dy);
    }
  }
}

// split a collider around a doorway so it stays walkable
function openColliderGap(axis, plane, at) {
  for (let i = COLLIDERS.length - 1; i >= 0; i--) {
    const c = COLLIDERS[i];
    if (c.axis !== axis || Math.abs(c.plane - plane) > 0.01) continue;
    if (at - DOOR_HALF > c.lo && at + DOOR_HALF < c.hi) {
      COLLIDERS.splice(i, 1,
        { ...c, hi: at - DOOR_HALF },
        { ...c, lo: at + DOOR_HALF });
    }
  }
}

function buildWorld() {
  // floors + ceilings
  for (const r of ROOMS) {
    if (r.solid) continue;
    const w = r.max[0] - r.min[0];
    const d = r.max[1] - r.min[1];
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(w, d),
      new THREE.MeshLambertMaterial({ color: r.floor }),
    );
    floor.rotation.x = -Math.PI / 2;
    // outdoor ground sits below room floors (they overlap: z-fighting)
    floor.position.copy(v3(r.min[0] + w / 2, r.min[1] + d / 2, r.outdoor ? -0.03 : 0.02));
    scene.add(floor);
    if (!r.outdoor) {
      const ceil = new THREE.Mesh(new THREE.PlaneGeometry(w, d), ceilMat);
      ceil.rotation.x = Math.PI / 2;
      ceil.position.copy(v3(r.min[0] + w / 2, r.min[1] + d / 2, r.h));
      scene.add(ceil);
    }
  }
  const grid = new THREE.GridHelper(80, 80, 0x24402e, 0x16281d);
  grid.position.copy(v3(17, 19, 0.0));
  scene.add(grid);

  // hand-authored wall list (avoids coplanar duplicates between rooms)
  wallY(0, 0, 8, 2.7);            // living south
  wallX(0, 0, 6, 2.7);            // living west
  wallX(8, 0, 6, 2.7);            // living east
  wallY(6, 0, 8, 2.7);            // living north (door 0)
  wallX(3.2, 6, 14, 2.4);         // corridor west
  wallX(4.8, 6, 14, 2.4);         // corridor east
  wallY(14, 0, 14, 7);            // hall south (door 1)
  wallX(0, 14, 24, 7);            // hall west
  wallX(14, 14, 24, 7);           // hall east
  wallY(24, 0, 14, 7);            // hall north (door 2, to outside)

  // club building (entered from outside via the entrance lobby)
  wallX(20, 28, 34, 2.6);         // entrance west (door 3)
  wallY(28, 20, 22, 2.6);         // entrance south
  wallY(34, 20, 22, 2.6);         // entrance north
  wallX(22, 26, 38, 4.5);         // club west (door 4, shared with entrance)
  wallX(32, 26, 38, 4.5);         // club east
  wallY(26, 22, 32, 4.5);         // club south
  wallY(38, 22, 32, 4.5);         // club north

  // old house: ground floor enterable, second storey visual mass
  wallY(16, 24, 31, 2.8);          // south
  wallX(24, 16, 23, 2.8);          // west (window)
  wallX(31, 16, 23, 2.8);          // east
  wallY(23, 24, 31, 2.8);          // north (door + window, faces the square)
  // upper storey: solid shell + decorative panes, blocks flying balls only
  const upper = (cx, cy, sx, sy) => {
    addBox(cx, cy, 4.2, sx, sy, 2.8);
    COLLIDERS.push({
      axis: sx < sy ? 0 : 1,
      plane: sx < sy ? cx : cy,
      lo: (sx < sy ? cy : cx) - Math.max(sx, sy) / 2,
      hi: (sx < sy ? cy : cx) + Math.max(sx, sy) / 2,
      h: 5.6, zlo: 2.8,
    });
  };
  upper(27.5, 16, 7, 0.15);
  upper(24, 19.5, 0.15, 7);
  upper(31, 19.5, 0.15, 7);
  upper(27.5, 23, 7, 0.15);
  {
    const slab = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), ceilMat);
    slab.rotation.x = Math.PI / 2;
    slab.position.copy(v3(27.5, 19.5, 2.8));
    scene.add(slab);
    const roof = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), ceilMat);
    roof.rotation.x = Math.PI / 2;
    roof.position.copy(v3(27.5, 19.5, 5.6));
    scene.add(roof);
    for (const [wx, wy, ax] of [[26, 23, 1], [29.3, 23, 1], [24, 19.5, 0]]) {
      const pane = new THREE.Mesh(
        ax ? new THREE.BoxGeometry(2.2, 1.2, 0.06) : new THREE.BoxGeometry(0.06, 1.2, 2.2),
        glassMat,
      );
      pane.position.copy(v3(wx, wy, 4.1)); // upper-floor windows (visual)
      scene.add(pane);
    }
  }
  // solid street furniture (reflectors/occluders)
  for (const r of ROOMS.filter((r) => r.solid)) {
    const w = r.max[0] - r.min[0];
    const d = r.max[1] - r.min[1];
    addBox(r.min[0] + w / 2, r.min[1] + d / 2, r.h / 2, w, d, r.h);
    COLLIDERS.push({ axis: 0, plane: r.min[0], lo: r.min[1], hi: r.max[1], h: r.h });
    COLLIDERS.push({ axis: 0, plane: r.max[0], lo: r.min[1], hi: r.max[1], h: r.h });
    COLLIDERS.push({ axis: 1, plane: r.min[1], lo: r.min[0], hi: r.max[0], h: r.h });
    COLLIDERS.push({ axis: 1, plane: r.max[1], lo: r.min[0], hi: r.max[0], h: r.h });
  }

  // sources: pulsing emissive markers + labels + local light
  for (const src of SOURCES) {
    src.meshes = [];
    for (const [ex, ey] of src.emitters ?? [src.pos]) {
      const isRig = !!src.emitters;
      const mesh = new THREE.Mesh(
        isRig
          ? new THREE.BoxGeometry(0.7, 1.1, 0.55) // PA speaker cabinet
          : new THREE.SphereGeometry(0.16, 24, 16),
        new THREE.MeshBasicMaterial({ color: src.color }),
      );
      mesh.position.copy(v3(ex, ey, isRig ? 1.9 : 1.5));
      scene.add(mesh);
      src.meshes.push(mesh);
    }
    const light = new THREE.PointLight(src.color, 6, 8);
    light.position.copy(v3(src.pos[0], src.pos[1], 1.8));
    scene.add(light);

    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 64;
    const cc = c.getContext('2d');
    cc.font = '40px sans-serif';
    cc.textAlign = 'center';
    cc.fillStyle = '#' + src.color.toString(16).padStart(6, '0');
    cc.fillText(src.name, 128, 44);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true }),
    );
    sprite.scale.set(1.6, 0.4, 1);
    sprite.position.copy(v3(src.pos[0], src.pos[1], 2.15));
    scene.add(sprite);
  }
}
buildWorld();

// ------------------------------------------------------------ start

const statusEl = document.getElementById('status');
const hintEl = document.getElementById('controls-hint');

document.getElementById('start').onclick = async () => {
  const err = document.getElementById('err');
  try {
    await startAudio();
    document.getElementById('overlay').remove();
    document.getElementById('recenter').hidden = false;
    document.getElementById('rain').hidden = false;
    document.getElementById('mixerbtn').hidden = false;
    buildMixer();
    document.getElementById('mixerbtn').onclick = (e) => {
      e.target.blur();
      const p = document.getElementById('mixer');
      p.hidden = !p.hidden;
    };
    setupControls();
    state.running = true;
  } catch (e) {
    err.textContent = String(e);
    console.error(e);
  }
};

// Render from the first frame: idle aerial orbit until the user starts.
requestAnimationFrame(frame);

async function startAudio() {
  const audio = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
  const fetchBuf = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
    return r.arrayBuffer();
  };
  statusEl.textContent = 'loading…';
  const [wasm1, wasm2, grid, speakers, ariaRaw, aliceRaw, clubRaw, fxW, fxT, fxB, ambRaw] =
    await Promise.all([
    fetchBuf('omg_web.wasm'),
    fetchBuf('omg_web.wasm'),
    fetchBuf('../assets/hrir_grid.bin'),
    fetchBuf('../assets/hrir_dodeca20.bin'),
    fetchBuf('../assets/aria48.ogg'),
    fetchBuf('../assets/alice48.ogg'),
    fetchBuf('../assets/club48.ogg'),
    fetchBuf('../assets/fx_whistle.ogg'),
    fetchBuf('../assets/fx_thump.ogg'),
    fetchBuf('../assets/fx_boom.ogg'),
    fetchBuf('../assets/night-nature48.ogg'),
  ]);

  const decodeMono = async (buf, target = 0.6) => {
    const ab = await audio.decodeAudioData(buf);
    const ch = ab.getChannelData(0);
    let peak = 1e-6;
    for (let i = 0; i < ch.length; i++) peak = Math.max(peak, Math.abs(ch[i]));
    const out = new Float32Array(ch.length);
    const g = target / peak;
    for (let i = 0; i < ch.length; i++) out[i] = ch[i] * g;
    return out;
  };
  const [aria, alice, club, whistle, thumpFx, boomFx, ambience] = await Promise.all([
    decodeMono(ariaRaw),
    decodeMono(aliceRaw),
    decodeMono(clubRaw),
    decodeMono(fxW, 0.18), // whistle: background-y
    decodeMono(fxT, 0.55),
    decodeMono(fxB, 1.9), // boom: BIG (AGC + tanh keep it safe)
    (async () => {
      const ab = await audio.decodeAudioData(ambRaw);
      const L = ab.getChannelData(0);
      let R = ab.numberOfChannels > 1 ? ab.getChannelData(1) : null;
      if (!R) {
        // mono bed: decorrelate the second channel with an offset read
        R = new Float32Array(L.length);
        const off = Math.floor(L.length / 3);
        for (let i = 0; i < L.length; i++) R[i] = L[(i + off) % L.length];
      }
      let peak = 1e-6;
      for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
      const out = new Float32Array(L.length * 2);
      const g = 0.55 / peak;
      for (let i = 0; i < L.length; i++) {
        out[i * 2] = L[i] * g;
        out[i * 2 + 1] = R[i] * g;
      }
      return out;
    })(),
  ]);
  const silent = new Float32Array(480); // projectile slot: fx voices only

  await audio.audioWorklet.addModule('worklet.js');
  const node = new AudioWorkletNode(audio, 'omg-engine', {
    numberOfInputs: 0,
    outputChannelCount: [2],
  });
  node.connect(audio.destination);
  state.node = node;
  document.getElementById('rain').onclick = (e) => { e.target.blur(); cycleRain(); };
  const analyser = audio.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.35;
  node.connect(analyser);
  state.analyser = analyser;
  state.specBins = new Uint8Array(analyser.frequencyBinCount);
  node.port.postMessage(
    { type: 'wasm', bytes: wasm1, grid, speakers,
      sources: [aria.buffer, alice.buffer, club.buffer, silent.buffer],
      fx: [whistle.buffer, thumpFx.buffer, boomFx.buffer],
      ambient: ambience.buffer },
    [wasm1, grid, speakers, aria.buffer, alice.buffer, club.buffer, silent.buffer,
     whistle.buffer, thumpFx.buffer, boomFx.buffer, ambience.buffer],
  );
  await new Promise((res) => {
    node.port.onmessage = (e) => {
      if (e.data.type === 'ready') res();
      if (e.data.type === 'meters') {
        state.meters.l = e.data.l;
        state.meters.r = e.data.r;
        state.meters.agc = e.data.agc;
        state.meters.pts = e.data.pts || 0;
        state.meters.tts = e.data.tts || 0;
        if (e.data.chans) {
          state.chanHist.push(e.data.chans);
          if (state.chanHist.length > 43) state.chanHist.shift();
        }
        state.meters.hist.push(e.data.agc);
        if (state.meters.hist.length > 220) state.meters.hist.shift();
      }
    };
  });

  const worker = new Worker('worker.js');
  worker.postMessage({ type: 'init', bytes: wasm2 }, [wasm2]);
  worker.onmessage = (e) => {
    if (e.data.type !== 'tick') return;
    node.port.postMessage({ type: 'params', blocks: e.data.blocks }, e.data.blocks);
    state.simState = new Float32Array(e.data.state);
    node.port.postMessage({
      type: 'ambient', gain: state.simState[62], fc: state.simState[63],
    });
  };

  state.fx = (src, kind, action = 'play') =>
    node.port.postMessage({ type: 'fx', src, kind, action });
  setInterval(() => {
    worker.postMessage({
      type: 'pose', x: state.pose.x, y: state.pose.y, yaw: 0,
      projs: state.projs.map((p) => [p.slot, p.x, p.y, p.z]),
    });
  }, 50);
  setInterval(() => {
    node.port.postMessage({ type: 'head', yaw: state.heading });
  }, 16);

  await audio.resume();
}

// ------------------------------------------------------------ controls

function setupControls() {
  const isTouch = matchMedia('(pointer: coarse)').matches;

  if (!isTouch) {
    hintEl.textContent = 'click: capture mouse, then click to throw · WASD to walk';
    glCanvas.addEventListener('click', () => {
      if (document.pointerLockElement !== glCanvas) glCanvas.requestPointerLock();
      else throwBall();
    });
    addEventListener('mousemove', (e) => {
      if (document.pointerLockElement !== glCanvas) return;
      state.heading -= e.movementX * 0.0025;
      state.pitch = Math.max(-1.4, Math.min(1.4, state.pitch - e.movementY * 0.0025));
    });
    addEventListener('keydown', (e) => {
      if (e.code === 'Space') throwBall();
      else if (e.code === 'KeyR') cycleRain();
      else state.keys.add(e.code);
    });
    addEventListener('keyup', (e) => state.keys.delete(e.code));
  } else {
    hintEl.textContent = 'left thumb: walk · turn your body (or drag right) to look';
    glCanvas.addEventListener('pointerdown', (e) => {
      if (e.clientX < innerWidth / 2 && !state.joy) {
        state.joy = { id: e.pointerId, sx: e.clientX, sy: e.clientY, dx: 0, dy: 0 };
      } else if (!state.look) {
        state.look = { id: e.pointerId, sx: e.clientX, sy: e.clientY,
                       heading0: state.heading, pitch0: state.pitch };
      }
      glCanvas.setPointerCapture(e.pointerId);
    });
    glCanvas.addEventListener('pointermove', (e) => {
      if (state.joy && e.pointerId === state.joy.id) {
        state.joy.dx = e.clientX - state.joy.sx;
        state.joy.dy = e.clientY - state.joy.sy;
      } else if (state.look && e.pointerId === state.look.id) {
        state.heading = state.look.heading0 - (e.clientX - state.look.sx) * 0.008;
        state.pitch = Math.max(-1.4, Math.min(1.4,
          state.look.pitch0 - (e.clientY - state.look.sy) * 0.008));
      }
    });
    const up = (e) => {
      if (state.joy && e.pointerId === state.joy.id) state.joy = null;
      if (state.look && e.pointerId === state.look.id) state.look = null;
    };
    glCanvas.addEventListener('pointerup', up);
    glCanvas.addEventListener('pointercancel', up);

    const btn = document.createElement('button');
    btn.textContent = '🎯 throw';
    btn.style.cssText =
      'position:absolute;right:12px;bottom:270px;z-index:6;padding:10px 16px;' +
      'background:#2a3648;color:#c8d2e1;';
    document.body.appendChild(btn);
    btn.addEventListener('click', throwBall);

    // Android Chrome: compass-fused orientation, no permission prompt.
    addEventListener('deviceorientationabsolute', (e) => {
      if (e.alpha == null) return;
      state.orientation = e.alpha;
      if (state.orientationOffset == null) {
        state.orientationOffset = state.heading + (e.alpha * Math.PI) / 180;
      }
      state.heading = state.orientationOffset - (e.alpha * Math.PI) / 180;
    });
  }

  document.getElementById('recenter').onclick = () => {
    if (state.orientation != null) {
      state.orientationOffset = Math.PI / 2 + (state.orientation * Math.PI) / 180;
    } else {
      state.heading = Math.PI / 2;
      state.pitch = 0;
    }
  };
}

function throwBall() {
  if (state.projs.length >= 3 || !state.fx) return;
  const usedSlots = new Set(state.projs.map((p) => p.slot));
  let slot = 0;
  while (usedSlots.has(slot)) slot++;
  const ch = Math.cos(state.heading);
  const sh = Math.sin(state.heading);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const speed = 13;
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xfff1a8 }),
  );
  const light = new THREE.PointLight(0xffcf7a, 3, 6);
  scene.add(mesh);
  scene.add(light);
  state.projs.push({
    slot,
    x: state.pose.x + ch * 0.4, y: state.pose.y + sh * 0.4, z: EYE,
    vx: ch * cp * speed, vy: sh * cp * speed, vz: sp * speed + 2.5,
    landedAt: 0, boomAt: 0, mesh, light,
  });
  state.fx(3 + slot, 0, 'play'); // whistle
}

function roomHeightAt(x, y) {
  for (const r of ROOMS) {
    if (!r.outdoor && x > r.min[0] && x < r.max[0] && y > r.min[1] && y < r.max[1]) return r.h;
  }
  return Infinity;
}

function updateProjectiles(dt, now) {
  for (const p of [...state.projs]) updateProjectile(p, dt, now);
}

function updateProjectile(p, dt, now) {
  const fx = (kind, action = 'play') => state.fx(3 + p.slot, kind, action);

  if (p.boomAt) {
    // explosion flash decays, then the source goes silent and disappears
    const k = (now - p.boomAt) / 1000;
    p.light.intensity = Math.max(0, 60 * (1 - k / 0.5));
    p.mesh.scale.setScalar(1 + k * 14);
    p.mesh.material.opacity = Math.max(0, 1 - k / 0.6);
    if (k > 2.4) {
      scene.remove(p.mesh);
      scene.remove(p.light);
      state.projs = state.projs.filter((q) => q !== p);
    }
    return;
  }

  p.vz -= 9.8 * dt;
  const nx = p.x + p.vx * dt;
  const ny = p.y + p.vy * dt;
  let nz = p.z + p.vz * dt;

  // wall bounce (height-aware: clears walls it flies over)
  if (crossesWall(p.x, p.y, nx, p.y, p.z)) {
    p.vx = -p.vx * 0.6;
    if (Math.hypot(p.vx, p.vy, p.vz) > 1.5) fx(1);
  } else p.x = nx;
  if (crossesWall(p.x, p.y, p.x, ny, p.z)) {
    p.vy = -p.vy * 0.6;
    if (Math.hypot(p.vx, p.vy, p.vz) > 1.5) fx(1);
  } else p.y = ny;

  // ceiling (from inside), roof landing (from above), ground
  const hRoom = roomHeightAt(p.x, p.y);
  if (hRoom !== Infinity) {
    if (p.z <= hRoom - 0.15 && nz > hRoom - 0.15 && p.vz > 0) {
      p.vz = -p.vz * 0.5;
      nz = hRoom - 0.15;
      fx(1);
    } else if (p.z >= hRoom + 0.1 && nz < hRoom + 0.1) {
      nz = hRoom + 0.12;
      if (Math.abs(p.vz) > 1.2) fx(1);
      p.vz = -p.vz * 0.5;
      p.vx *= 0.8;
      p.vy *= 0.8;
      if (!p.landedAt) p.landedAt = now;
    }
  }
  if (nz < 0.12) {
    nz = 0.12;
    if (Math.abs(p.vz) > 1.2) fx(1);
    p.vz = -p.vz * 0.55;
    p.vx *= 0.75;
    p.vy *= 0.75;
    if (!p.landedAt) p.landedAt = now;
  }
  p.z = nz;

  if (p.landedAt && now - p.landedAt > 3000 && !p.boomAt) {
    p.boomAt = now;
    fx(0, 'stop');
    fx(2, 'play');
    p.mesh.material = new THREE.MeshBasicMaterial({
      color: 0xffb46a, transparent: true, opacity: 1,
    });
  }

  p.mesh.position.copy(v3(p.x, p.y, p.z));
  p.light.position.copy(v3(p.x, p.y, p.z + 0.1));
}

const WALK_SPEED = 3.0;
let lastT = 0;

function movePose(t) {
  const dt = Math.min((t - lastT) / 1000, 0.05);
  lastT = t;
  let fwd = 0;
  let strafe = 0;

  if (state.keys.has('KeyW') || state.keys.has('ArrowUp')) fwd += 1;
  if (state.keys.has('KeyS') || state.keys.has('ArrowDown')) fwd -= 1;
  if (state.keys.has('KeyD') || state.keys.has('ArrowRight')) strafe += 1;
  if (state.keys.has('KeyA') || state.keys.has('ArrowLeft')) strafe -= 1;

  if (state.joy) {
    const mag = Math.hypot(state.joy.dx, state.joy.dy);
    if (mag > 8) {
      const k = Math.min(mag, 80) / 80;
      fwd += (-state.joy.dy / Math.max(mag, 1)) * k;
      strafe += (state.joy.dx / Math.max(mag, 1)) * k;
    }
  }
  if (!fwd && !strafe) return;

  // forward = along heading, strafe-right = heading − 90°
  const mag = Math.hypot(fwd, strafe);
  const len = Math.min(mag, 1);
  const f = (fwd / mag) * len;
  const r = (strafe / mag) * len;
  const ch = Math.cos(state.heading);
  const sh = Math.sin(state.heading);
  const step = WALK_SPEED * dt;
  const nx = state.pose.x + (ch * f + sh * r) * step;
  const ny = state.pose.y + (sh * f - ch * r) * step;
  const { x, y } = state.pose;
  if (walkableMove(x, y, nx, ny)) {
    state.pose.x = nx;
    state.pose.y = ny;
  } else if (walkableMove(x, y, nx, y)) {
    state.pose.x = nx;
  } else if (walkableMove(x, y, x, ny)) {
    state.pose.y = ny;
  }
}

// ------------------------------------------------------------ render loop

const mm = document.getElementById('minimap').getContext('2d');

function frame(t) {
  if (!state.running) {
    // idle flyover while the start overlay is up
    const a = t / 12000;
    camera.position.copy(v3(17 + 26 * Math.cos(a), 19 + 26 * Math.sin(a), 16));
    camera.lookAt(v3(17, 19, 1));
  } else {
    const dt = Math.min((t - (state.prevT || t)) / 1000, 0.05);
    state.prevT = t;
    movePose(t);
    updateProjectiles(dt, t);
    camera.position.copy(v3(state.pose.x, state.pose.y, EYE));
    camera.rotation.y = state.heading - Math.PI / 2;
    camera.rotation.x = state.pitch;
  }

  const tt = t / 1000;
  for (const src of SOURCES) {
    const s = 1 + 0.25 * Math.sin(tt * 2 * Math.PI * 1.6);
    for (const m of src.meshes) m.scale.setScalar(s);
  }

  renderer.render(scene, camera);
  if (state.running) {
    drawMinimap();
    drawMeters();
    drawSpec();
  }

  const st = state.simState;
  if (st) {
    statusEl.textContent = `${ROOMS[st[2] | 0].name} · RT60 ${st[3].toFixed(2)}s`;
  }
  requestAnimationFrame(frame);
}

function drawMeters() {
  updateMixerMeters();
  const c = document.getElementById('meters');
  if (!c) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#0e1116cc';
  g.fillRect(0, 0, W, H);
  const db = (x) => Math.max(-60, 20 * Math.log10(x + 1e-6));
  const bar = (x0, val, color, label) => {
    const h = ((db(val) + 60) / 60) * (H - 56);
    g.fillStyle = '#1c2430';
    g.fillRect(x0, 28, 26, H - 56);
    g.fillStyle = color;
    g.fillRect(x0, 28 + (H - 56) - h, 26, h);
    g.fillStyle = '#7a8496';
    g.font = '13px sans-serif';
    g.textAlign = 'center';
    g.fillText(label, x0 + 13, H - 12);
  };
  bar(12, state.meters.l, '#50dcff', 'L');
  bar(48, state.meters.r, '#50dcff', 'R');
  // AGC ("ear gain") history: dB line, −24…+14
  g.strokeStyle = '#ffaa3c';
  g.lineWidth = 2;
  g.beginPath();
  const hist = state.meters.hist;
  for (let i = 0; i < hist.length; i++) {
    const gx = 86 + (i / 220) * (W - 98);
    const gdb = 20 * Math.log10(hist[i]);
    const gy = 28 + (1 - (gdb + 24) / 38) * (H - 56);
    i ? g.lineTo(gx, gy) : g.moveTo(gx, gy);
  }
  g.stroke();
  g.fillStyle = '#ffaa3c';
  g.textAlign = 'left';
  g.fillText(`ear ${(20 * Math.log10(state.meters.agc)).toFixed(1)} dB`, 86, 18);
  if (state.meters.tts > 0.02) {
    // hearing fatigue after ultra-loud exposure (muffle depth)
    g.fillStyle = '#ff6a5a';
    g.fillText(`muffled ${Math.round(state.meters.tts * 100)}%`, 200, 18);
  }
  if (state.meters.pts) {
    // adaptive point-HRTF budget (per source), set by the worklet from load
    g.fillStyle = '#7a8496';
    g.fillText(`hrtf ×${state.meters.pts}`, 86, H - 12);
  }
  // 0 dB reference line
  g.strokeStyle = '#3a4658';
  g.lineWidth = 1;
  const y0 = 28 + (1 - 24 / 38) * (H - 56);
  g.beginPath();
  g.moveTo(86, y0);
  g.lineTo(W - 12, y0);
  g.stroke();
}

// Inferno colormap (perceptually uniform: black → purple → crimson →
// orange → yellow), piecewise-linear between matplotlib anchor stops.
const SPEC_LUT = (() => {
  const stops = [
    [0.0, 0, 0, 4], [0.125, 22, 11, 57], [0.25, 66, 10, 104],
    [0.375, 106, 23, 110], [0.5, 147, 38, 103], [0.625, 188, 55, 84],
    [0.75, 221, 81, 58], [0.875, 243, 120, 25], [0.9375, 249, 189, 38],
    [1.0, 252, 255, 164],
  ];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (stops[k + 1][0] < t) k++;
    const u = (t - stops[k][0]) / (stops[k + 1][0] - stops[k][0]);
    for (let ch = 0; ch < 3; ch++) {
      lut[i * 3 + ch] = Math.round(stops[k][ch + 1] + u * (stops[k + 1][ch + 1] - stops[k][ch + 1]));
    }
  }
  return lut;
})();

function drawSpec() {
  const c = document.getElementById('spec');
  if (!c || !state.analyser) return;
  const g = c.getContext('2d');
  const W = c.width;
  const H = c.height;
  state.analyser.getByteFrequencyData(state.specBins);
  // scroll left, draw the new column at the right edge
  g.drawImage(c, -2, 0);
  const bins = state.specBins;
  const fs = 48000;
  const nyq = fs / 2;
  if (!state.specCol) state.specCol = g.createImageData(2, H);
  const px = state.specCol.data;
  for (let y = 0; y < H; y++) {
    // log frequency axis: 40 Hz (bottom) … 16 kHz (top)
    const frac = 1 - y / H;
    const f = 40 * Math.pow(16000 / 40, frac);
    const bin = Math.min(bins.length - 1, Math.round((f / nyq) * bins.length));
    // analyser bytes are already dB-scaled; mild gamma keeps the floor dark
    const li = Math.round(Math.pow(bins[bin] / 255, 1.25) * 255) * 3;
    for (let x = 0; x < 2; x++) {
      const o = (y * 2 + x) * 4;
      px[o] = SPEC_LUT[li];
      px[o + 1] = SPEC_LUT[li + 1];
      px[o + 2] = SPEC_LUT[li + 2];
      px[o + 3] = 255;
    }
  }
  g.putImageData(state.specCol, W - 2, 0);
  // frequency gridlines: 100 Hz, 1 kHz, 10 kHz
  g.fillStyle = '#7a849644';
  for (const f of [100, 1000, 10000]) {
    const y = H * (1 - Math.log(f / 40) / Math.log(16000 / 40));
    g.fillRect(0, y, W, 1);
  }
}

function drawMinimap() {
  const W = 300;
  const H = 480;
  mm.clearRect(0, 0, W, H);
  mm.fillStyle = '#0e1116cc';
  mm.fillRect(0, 0, W, H);
  const xs = ROOMS.flatMap((r) => [r.min[0], r.max[0]]);
  const ys = ROOMS.flatMap((r) => [r.min[1], r.max[1]]);
  const [mx0, mx1, my0, my1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const s = Math.min(W / (mx1 - mx0 + 2), H / (my1 - my0 + 2));
  const sx = (wx) => (wx - mx0 + 1) * s;
  const sy = (wy) => H - (wy - my0 + 1) * s;
  const room = state.simState ? state.simState[2] | 0 : -1;

  ROOMS.forEach((r, i) => {
    if (r.solid) {
      mm.fillStyle = '#3a4658';
      mm.fillRect(sx(r.min[0]), sy(r.max[1]),
                  (r.max[0] - r.min[0]) * s, (r.max[1] - r.min[1]) * s);
      return;
    }
    mm.strokeStyle = i === room ? '#78c8ff' : '#5a6980';
    mm.lineWidth = i === room ? 3 : 1.5;
    mm.strokeRect(sx(r.min[0]), sy(r.max[1]),
                  (r.max[0] - r.min[0]) * s, (r.max[1] - r.min[1]) * s);
  });
  DOORS.forEach(({ pos: [dx, dy], axis }) => {
    mm.strokeStyle = '#0e1116';
    mm.lineWidth = 5;
    mm.beginPath();
    if (axis === 1) {
      mm.moveTo(sx(dx - 0.55), sy(dy));
      mm.lineTo(sx(dx + 0.55), sy(dy));
    } else {
      mm.moveTo(sx(dx), sy(dy - 0.55));
      mm.lineTo(sx(dx), sy(dy + 0.55));
    }
    mm.stroke();
  });
  WINDOWS.forEach(({ pos: [wx, wy], axis, hw }) => {
    mm.strokeStyle = '#7ac8ff';
    mm.lineWidth = 3;
    mm.beginPath();
    if (axis === 1) {
      mm.moveTo(sx(wx - hw), sy(wy));
      mm.lineTo(sx(wx + hw), sy(wy));
    } else {
      mm.moveTo(sx(wx), sy(wy - hw));
      mm.lineTo(sx(wx), sy(wy + hw));
    }
    mm.stroke();
  });
  SOURCES.forEach((src) => {
    mm.fillStyle = '#' + src.color.toString(16).padStart(6, '0');
    mm.beginPath();
    mm.arc(sx(src.pos[0]), sy(src.pos[1]), 4, 0, Math.PI * 2);
    mm.fill();
  });
  const lx = sx(state.pose.x);
  const ly = sy(state.pose.y);
  const a = -state.heading;
  mm.fillStyle = '#50dcff';
  mm.beginPath();
  mm.moveTo(lx + 9 * Math.cos(a), ly + 9 * Math.sin(a));
  mm.lineTo(lx + 6 * Math.cos(a + 2.5), ly + 6 * Math.sin(a + 2.5));
  mm.lineTo(lx + 6 * Math.cos(a - 2.5), ly + 6 * Math.sin(a - 2.5));
  mm.closePath();
  mm.fill();
}
