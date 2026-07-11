// Simulation worker: wasm WorldSim ticked at 20 Hz with the latest listener
// pose from the main thread; posts flat ParamBlocks (transferred) + a small
// state buffer for the canvas viz.
let w = null;
let pose = { x: 3.0, y: 3.0, yaw: 0.0, projs: [] };

onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    const { instance } = await WebAssembly.instantiate(m.bytes, {});
    w = instance.exports;
    w.sim_setup();
    setInterval(tick, 50);
  } else if (m.type === 'pose') {
    pose = m;
  }
};

function tick() {
  if (!w) return;
  const dyn = new Float32Array(w.memory.buffer, w.sim_dyn_ptr(), 12);
  dyn.fill(0);
  (pose.projs || []).forEach((p) => {
    const slot = p[0];
    if (slot >= 0 && slot < 3) dyn.set([p[1], p[2], p[3], 1], slot * 4);
  });
  w.sim_tick(pose.x, pose.y, pose.yaw);
  const blocks = [];
  for (let i = 0; i < 6; i++) {
    const len = w.sim_params_len(i);
    const src = new Float32Array(w.memory.buffer, w.sim_params_ptr(i), len);
    blocks.push(src.slice().buffer);
  }
  const state = new Float32Array(w.memory.buffer, w.sim_state_ptr(), 64).slice();
  postMessage({ type: 'tick', blocks, state: state.buffer },
              [...blocks, state.buffer]);
}
