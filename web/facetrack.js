// Camera head tracking: the webcam watches your face and your ACTUAL
// head movements drive the engine's head — the view (camera/cursor)
// stays put, only the simulated ears move. Turn your head a little and
// the sound field counter-rotates immediately; lean left and the
// listening position shifts left. MediaPipe's FaceLandmarker fits a
// face model per video frame (a few ms on a GPU delegate) and returns
// a facial transformation matrix; we decompose it into yaw/pitch/roll
// + translation, subtract the neutral pose captured at start (or on
// recenter), and hand the deltas to main.js, which smooths between
// camera frames at render rate.
//
// Loaded from CDN on demand — the feature is opt-in via the 🎥 button
// and degrades to a console error when offline or without a camera.

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
  'face_landmarker/float16/1/face_landmarker.task';

// Decompose the (column-major) facial transformation matrix. Camera
// space, un-mirrored front camera: +x = the user's LEFT (image right),
// +y up, camera looks down −z (the face sits at z ≈ −0.3 m). The face's
// forward axis is the matrix's third column, its up axis the second.
// Angles come out in the ENGINE's conventions: yaw+ = turned left,
// pitch+ = looking up, roll+ = tilted right (right ear down).
function headPose(m) {
  const yaw = Math.atan2(m[8], m[10]);
  const pitch = Math.atan2(m[9], Math.hypot(m[8], m[10]));
  const roll = -Math.atan2(m[4], m[5]);
  // translation is in centimeters
  return { yaw, pitch, roll, tx: m[12] / 100, ty: m[13] / 100, tz: m[14] / 100 };
}

export async function startFaceTracking(onPose) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 60 },
    },
    audio: false,
  });
  const video = document.createElement('video');
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();

  let landmarker;
  try {
    const { FilesetResolver, FaceLandmarker } = await import(`${CDN}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
      outputFaceBlendshapes: false,
    });
    try {
      landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
    } catch {
      landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
    }
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    throw e;
  }

  let neutral = null;
  let last = null;
  let stopped = false;
  let lastTs = 0;
  const stats = { fps: 0, ms: 0, tracking: false };
  let frames = 0;
  let statT = performance.now();

  const step = () => {
    if (stopped) return;
    // monotonically increasing timestamps are required in VIDEO mode
    const ts = Math.max(performance.now(), lastTs + 1e-3);
    lastTs = ts;
    const t0 = performance.now();
    const res = landmarker.detectForVideo(video, ts);
    stats.ms = performance.now() - t0;
    const m = res.facialTransformationMatrixes?.[0]?.data;
    stats.tracking = !!m;
    if (m) {
      const pose = headPose(m);
      last = pose;
      neutral ??= pose;
      onPose({
        yaw: pose.yaw - neutral.yaw,
        pitch: pose.pitch - neutral.pitch,
        roll: pose.roll - neutral.roll,
        // camera x is the user's left; leaning toward the screen makes
        // z less negative, i.e. Δz > 0 = leaning forward
        dx: pose.tx - neutral.tx,
        dy: pose.ty - neutral.ty,
        dz: pose.tz - neutral.tz,
      });
    }
    frames++;
    const now = performance.now();
    if (now - statT > 1000) {
      stats.fps = (frames * 1000) / (now - statT);
      frames = 0;
      statT = now;
    }
    schedule();
  };
  const schedule = () => {
    if (stopped) return;
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  schedule();

  return {
    stats,
    // re-zero on the current pose — sit comfortably, hit recenter
    recenter() {
      if (last) neutral = last;
    },
    stop() {
      stopped = true;
      stream.getTracks().forEach((t) => t.stop());
      landmarker.close();
    },
  };
}
