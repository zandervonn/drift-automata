import * as THREE from 'three';

// SmoothLife on the GPU. State is a single-channel field stored in RGBA16F
// render targets, ping-ponged each frame.
//
// COORDINATE MODEL.
//   • The simulation buffer is sized to the current canvas (1 cell per
//     canvas pixel). Reallocated on window resize.
//   • Cells live in WORLD coords. Each cell at buffer pixel (i, j) sits at
//     world coord = view.cx + (i - bufW/2) / view.zoom, similarly for y.
//   • view = { cx, cy, zoom }. zoom = canvas pixels per world unit.
//   • Each frame, before stepping, the buffer is REMAPPED from the previous
//     view to the current view: cells whose world coord is outside the new
//     viewport drop off (culled forever); cells inside the new viewport are
//     resampled into their new buffer position. Pan slides cells; zoom-in
//     magnifies them and culls anything beyond the new tighter viewport.
//   • Noise (per-rule deviation + pastel background) is sampled at world
//     coords using the current view. It's procedural and continuous, so the
//     pastel layers cover every canvas pixel — both where cells live and
//     out-of-buffer (e.g. the moment after zoom-in before fresh cells fill).
let SIM_W = Math.max(2, window.innerWidth );
let SIM_H = Math.max(2, window.innerHeight);
console.log(`[smoothlife] viewport buffer ${SIM_W}x${SIM_H} (${(SIM_W*SIM_H/1e6).toFixed(2)}M cells)`);

const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x14141e, 1);
document.body.appendChild(renderer.domElement);

if (!renderer.capabilities.isWebGL2) {
  document.body.innerHTML = '<p style="color:#f66;padding:24px">WebGL2 is required.</p>';
  throw new Error('WebGL2 required');
}

{
  const gl  = renderer.getContext();
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const gpu    = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '(masked)';
  const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL)   : '(masked)';
  console.log(`[smoothlife] webgl2 ok — GPU: ${gpu}  vendor: ${vendor}`);
}

const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const fsQuad = new THREE.PlaneGeometry(2, 2);

// ClampToEdge: the field is no longer toroidal. When the kernel reaches past
// the SIM bounds it samples the (typically empty) edge cells, which behaves
// like a hard boundary instead of wrapping the playground onto itself.
const rtOpts = {
  format: THREE.RGBAFormat,
  type: THREE.HalfFloatType,
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  wrapS: THREE.ClampToEdgeWrapping,
  wrapT: THREE.ClampToEdgeWrapping,
  depthBuffer: false,
  stencilBuffer: false,
};
let rtRead  = new THREE.WebGLRenderTarget(SIM_W, SIM_H, rtOpts);
let rtWrite = new THREE.WebGLRenderTarget(SIM_W, SIM_H, rtOpts);

// ---------------------------------------------------------------------------
// Per-rule metadata. devMax is the slider max for noise deviation; the display
// shader also uses it to normalize each pastel layer's intensity.
// ---------------------------------------------------------------------------

const RULES = [
  { key: 'b1',      base: 0.278, devMax: 0.30, baseRange: [0,    1,    0.001], color: '#ffc6d3' }, // pale rose
  { key: 'b2',      base: 0.365, devMax: 0.30, baseRange: [0,    1,    0.001], color: '#ffd8b8' }, // peach
  { key: 'd1',      base: 0.267, devMax: 0.30, baseRange: [0,    1,    0.001], color: '#fff1a8' }, // pale gold
  { key: 'd2',      base: 0.445, devMax: 0.30, baseRange: [0,    1,    0.001], color: '#d6f0a8' }, // pale lime
  { key: 'alpha_n', base: 0.028, devMax: 0.10, baseRange: [0.001,0.3,  0.001], color: '#b8e6d2' }, // mint
  { key: 'alpha_m', base: 0.147, devMax: 0.10, baseRange: [0.001,0.3,  0.001], color: '#b8dcf0' }, // sky
  { key: 'dt',      base: 0.20,  devMax: 0.20, baseRange: [0.01, 0.5,  0.01 ], color: '#cdc6f0' }, // lavender
  { key: 'ra',      base: 4.0,   devMax: 2.0,  baseRange: [1.0,  18,   0.5  ], color: '#e8c6f0' }, // lilac
  { key: 'rb',      base: 12.0,  devMax: 4.0,  baseRange: [3.0,  24,   0.5  ], color: '#f0c6c6' }, // dusty pink
];
const RULE_KEYS = RULES.map(r => r.key);

// ---------------------------------------------------------------------------
// Shaders (GLSL ES 3.00 — three.js prepends `#version 300 es` for us)
// ---------------------------------------------------------------------------

const VERT = /* glsl */`
precision highp float;
in vec3 position;
in vec2 uv;
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

// Ashima/Stefan Gustavson 2D simplex noise — returns ~[-1, 1].
const SNOISE_GLSL = /* glsl */`
vec3 _sn_mod289_3(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec2 _sn_mod289_2(vec2 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec3 _sn_permute(vec3 x) { return _sn_mod289_3(((x*34.0)+1.0)*x); }
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                     -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = _sn_mod289_2(i);
  vec3 p = _sn_permute(_sn_permute(i.y + vec3(0.0, i1.y, 1.0))
                       + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m; m = m*m;
  vec3 px2 = 2.0 * fract(p * C.www) - 1.0;
  vec3 h   = abs(px2) - 0.5;
  vec3 ox  = floor(px2 + 0.5);
  vec3 a0  = px2 - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  vec3 g;
  g.x  = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}
`;

const ruleUniformBlock = (k, includeDisplay) => `
uniform float u_${k}_base;
uniform float u_${k}_dev;
uniform float u_${k}_scale;
uniform vec2  u_${k}_offset;
uniform int   u_${k}_flat;${includeDisplay ? `
uniform float u_${k}_dev_max;
uniform vec3  u_${k}_pastel;` : ''}
`;

const stepRuleUniforms    = RULE_KEYS.map(k => ruleUniformBlock(k, false)).join('');
const displayRuleUniforms = RULE_KEYS.map(k => ruleUniformBlock(k, true )).join('');

// Noise frequency multiplier: at slider scale=4, this gives a handful of
// cycles per typical viewport — visible but not busy.
const NOISE_SPATIAL_K = 0.0005;

// worldP for noise is in pure WORLD units (1 cell = 1 world unit). Zoom does
// not enter worldP, so noise patterns stay locked to world position; pan
// shifts cells through the rule landscape, zoom just lets you see more or
// fewer cycles of that landscape on screen at once.
const stepRuleEvals = RULE_KEYS.map(k =>
  `  float ${k} = u_${k}_base + (u_${k}_flat == 1 ? 0.0 : u_${k}_dev * snoise(worldP * (u_${k}_scale * ${NOISE_SPATIAL_K}) + u_${k}_offset));`
).join('\n');

const displayPastelLayers = RULE_KEYS.map(k => `
  {
    float n = (u_${k}_flat == 1) ? 0.0 : snoise(worldP * (u_${k}_scale * ${NOISE_SPATIAL_K}) + u_${k}_offset);
    float devNorm = u_${k}_dev_max > 0.0 ? clamp(u_${k}_dev / u_${k}_dev_max, 0.0, 1.0) : 0.0;
    float intensity = abs(n) * devNorm * (50.0/255.0);
    bg = mix(bg, u_${k}_pastel, intensity);
  }`).join('');

const STEP_FRAG = /* glsl */`
precision highp float;
uniform sampler2D u_state;
uniform vec2  u_res;            // buffer = canvas resolution (in pixels)
uniform float u_rb_max;
uniform vec2  u_view_center;    // world coord at buffer center
uniform float u_view_zoom;      // canvas pixels per world unit
${stepRuleUniforms}
in  vec2 vUv;
out vec4 outColor;

${SNOISE_GLSL}

float sigma1(float x, float a, float alpha) {
  return 1.0 / (1.0 + exp(-(x - a) * 4.0 / alpha));
}
float sigmaN(float x, float a, float b, float alpha_n_) {
  return sigma1(x, a, alpha_n_) * (1.0 - sigma1(x, b, alpha_n_));
}
float sigmaM(float x, float y, float m, float alpha_m_) {
  return mix(x, y, sigma1(m, 0.5, alpha_m_));
}
float transition(float n, float m,
                 float b1_, float b2_, float d1_, float d2_,
                 float alpha_n_, float alpha_m_) {
  return sigmaN(n,
                sigmaM(b1_, d1_, m, alpha_m_),
                sigmaM(b2_, d2_, m, alpha_m_),
                alpha_n_);
}

void main() {
  // Each buffer pixel maps to a world coord through the current view —
  // 1 buffer pixel = 1/zoom world units. Noise samples here; rules are
  // determined by the cell's world coord under the current view.
  vec2 worldP = u_view_center + (vUv * u_res - u_res * 0.5) / u_view_zoom;
${stepRuleEvals}
  // ra and rb above are in WORLD units. The convolution kernel works in
  // buffer pixels, so multiply by view.zoom — the kernel "feels" the same
  // world scale at every zoom, which means structures emerge at a fixed
  // world size and visually scale with zoom (same as the noise).
  float ra_px = ra * u_view_zoom;
  float rb_px = rb * u_view_zoom;
  vec2 px = 1.0 / u_res;
  float diskSum  = 0.0;
  float diskArea = 0.0;
  float ringSum  = 0.0;
  float ringArea = 0.0;
  int R = max(1, int(ceil(u_rb_max * u_view_zoom)));
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      float fx = float(dx);
      float fy = float(dy);
      float r  = sqrt(fx*fx + fy*fy);
      float inDisk = clamp(ra_px + 0.5 - r, 0.0, 1.0);
      float inRing = clamp(rb_px + 0.5 - r, 0.0, 1.0) - inDisk;
      float v = texture(u_state, vUv + vec2(fx, fy) * px).r;
      diskSum  += v * inDisk;
      diskArea += inDisk;
      ringSum  += v * inRing;
      ringArea += inRing;
    }
  }
  float m = diskSum / max(diskArea, 1e-6);
  float n = ringSum / max(ringArea, 1e-6);
  float prev = texture(u_state, vUv).r;
  float t    = transition(n, m, b1, b2, d1, d2, alpha_n, alpha_m);
  float next = clamp(prev + dt * (2.0 * t - 1.0), 0.0, 1.0);
  outColor = vec4(next, next, next, 1.0);
}
`;

const DISPLAY_FRAG = /* glsl */`
precision highp float;
uniform sampler2D u_state;
uniform vec2  u_res;            // buffer = canvas resolution
uniform vec2  u_view_center;    // world coord at buffer center
uniform float u_view_zoom;      // canvas pixels per world unit
uniform vec3  u_bg_color;
uniform vec3  u_cell_color;
${displayRuleUniforms}
in  vec2 vUv;
out vec4 outColor;

${SNOISE_GLSL}

void main() {
  // The buffer matches the canvas, so sampling state at vUv reads exactly the
  // cell living at this canvas pixel. World coord is computed for noise so
  // the rule landscape stays continuous as you pan/zoom.
  vec2 worldP = u_view_center + (vUv * u_res - u_res * 0.5) / u_view_zoom;

  vec3 bg = u_bg_color;
${displayPastelLayers}

  float v = texture(u_state, vUv).r;
  outColor = vec4(mix(bg, u_cell_color, v), 1.0);
}
`;

const COPY_FRAG = /* glsl */`
precision highp float;
uniform sampler2D u_src;
in  vec2 vUv;
out vec4 outColor;
void main() {
  outColor = texture(u_src, vUv);
}
`;

const STAMP_FRAG = /* glsl */`
precision highp float;
uniform sampler2D u_src;
uniform vec2  u_res;
uniform vec2  u_pos;
uniform float u_radius_px;
uniform float u_value;
uniform int   u_mode;
in  vec2 vUv;
out vec4 outColor;
void main() {
  vec2 d = (vUv - u_pos) * u_res;
  float r = length(d);
  float w = 1.0 - smoothstep(u_radius_px - 1.0, u_radius_px + 1.0, r);
  float src = texture(u_src, vUv).r;
  float painted = (u_mode == 0)
      ? max(src, u_value * w)
      : src * (1.0 - w);
  outColor = vec4(vec3(painted), 1.0);
}
`;

// Re-align the buffer from the previous view to the current view. For each
// destination buffer pixel, find the world coord (current view) and look up
// where that world coord sat in the source buffer (previous view). Cells
// whose world coord is outside the source viewport read 0 (empty); cells
// whose world coord is outside the new viewport are simply not written
// (effectively culled).
const REMAP_FRAG = /* glsl */`
precision highp float;
uniform sampler2D u_src;
uniform vec2  u_res;
uniform vec2  u_src_center;
uniform float u_src_zoom;
uniform vec2  u_dst_center;
uniform float u_dst_zoom;
in  vec2 vUv;
out vec4 outColor;
void main() {
  vec2 dstPx = vUv * u_res - u_res * 0.5;
  vec2 worldP = u_dst_center + dstPx / u_dst_zoom;
  vec2 srcPx = (worldP - u_src_center) * u_src_zoom + u_res * 0.5;
  vec2 srcUv = srcPx / u_res;
  if (srcUv.x < 0.0 || srcUv.x > 1.0 || srcUv.y < 0.0 || srcUv.y > 1.0) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    outColor = texture(u_src, srcUv);
  }
}
`;

function makeMat(frag, uniforms) {
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms,
    vertexShader: VERT,
    fragmentShader: frag,
  });
}

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

const stepUniforms = {
  u_state:       { value: null },
  u_res:         { value: new THREE.Vector2(SIM_W, SIM_H) },
  u_rb_max:      { value: 16.0 },
  u_view_center: { value: new THREE.Vector2(0, 0) },
  u_view_zoom:   { value: 1.0 },
};
for (const r of RULES) {
  stepUniforms[`u_${r.key}_base`]   = { value: r.base };
  stepUniforms[`u_${r.key}_dev`]    = { value: 0.0 };
  stepUniforms[`u_${r.key}_scale`]  = { value: 4.0 };
  stepUniforms[`u_${r.key}_offset`] = { value: new THREE.Vector2(0, 0) };
  stepUniforms[`u_${r.key}_flat`]   = { value: 1 };
}

const displayUniforms = {
  u_state:       { value: null },
  u_res:         { value: new THREE.Vector2(SIM_W, SIM_H) },
  u_view_center: { value: new THREE.Vector2(0, 0) },
  u_view_zoom:   { value: 1.0 },
  u_bg_color:    { value: new THREE.Vector3(0.078, 0.078, 0.118) },  // ~#14141e
  u_cell_color:  { value: new THREE.Vector3(1.000, 0.847, 0.420) },  // ~#ffd86b
};
for (const r of RULES) {
  displayUniforms[`u_${r.key}_base`]    = { value: r.base };
  displayUniforms[`u_${r.key}_dev`]     = { value: 0.0 };
  displayUniforms[`u_${r.key}_scale`]   = { value: 4.0 };
  displayUniforms[`u_${r.key}_offset`]  = { value: new THREE.Vector2(0, 0) };
  displayUniforms[`u_${r.key}_flat`]    = { value: 1 };
  displayUniforms[`u_${r.key}_dev_max`] = { value: r.devMax };
  displayUniforms[`u_${r.key}_pastel`]  = { value: new THREE.Vector3(1, 1, 1) };
}

const copyUniforms = { u_src: { value: null } };
const stampUniforms = {
  u_src:       { value: null },
  u_res:       { value: new THREE.Vector2(SIM_W, SIM_H) },
  u_pos:       { value: new THREE.Vector2(0.5, 0.5) },
  u_radius_px: { value: 14.0 },
  u_value:     { value: 1.0 },
  u_mode:      { value: 0 },
};
const remapUniforms = {
  u_src:        { value: null },
  u_res:        { value: new THREE.Vector2(SIM_W, SIM_H) },
  u_src_center: { value: new THREE.Vector2(0, 0) },
  u_src_zoom:   { value: 1.0 },
  u_dst_center: { value: new THREE.Vector2(0, 0) },
  u_dst_zoom:   { value: 1.0 },
};

function makeScene(material) {
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(fsQuad, material));
  return scene;
}
const stepScene    = makeScene(makeMat(STEP_FRAG,    stepUniforms));
const displayScene = makeScene(makeMat(DISPLAY_FRAG, displayUniforms));
const copyScene    = makeScene(makeMat(COPY_FRAG,    copyUniforms));
const stampScene   = makeScene(makeMat(STAMP_FRAG,   stampUniforms));
const remapScene   = makeScene(makeMat(REMAP_FRAG,   remapUniforms));

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

function makeSeed(kind) {
  const data = new Uint8Array(SIM_W * SIM_H * 4);
  if (kind === 'clear') return data;
  if (kind === 'noise') {
    for (let i = 0; i < SIM_W * SIM_H; i++) {
      const v = Math.random() < 0.5 ? 0 : 255;
      data[i*4] = v; data[i*4+1] = v; data[i*4+2] = v; data[i*4+3] = 255;
    }
    return data;
  }
  // 'circles' — density scaled to grid area, capped for big fields
  const count = Math.min(2000, Math.round(60 * (SIM_W * SIM_H) / (256 * 256)));
  for (let c = 0; c < count; c++) {
    const cx = Math.random() * SIM_W;
    const cy = Math.random() * SIM_H;
    const rad = 6 + Math.random() * 22;
    const r2 = rad * rad;
    const ymin = Math.floor(cy - rad - 1);
    const ymax = Math.ceil (cy + rad + 1);
    const xmin = Math.floor(cx - rad - 1);
    const xmax = Math.ceil (cx + rad + 1);
    for (let y = ymin; y <= ymax; y++) {
      for (let x = xmin; x <= xmax; x++) {
        const dx = x - cx, dy = y - cy;
        if (dx*dx + dy*dy < r2) {
          const xx = ((x % SIM_W) + SIM_W) % SIM_W;
          const yy = ((y % SIM_H) + SIM_H) % SIM_H;
          const idx = (yy * SIM_W + xx) * 4;
          data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
        }
      }
    }
  }
  return data;
}

function uploadSeed(kind) {
  const seed = makeSeed(kind);
  const tex = new THREE.DataTexture(seed, SIM_W, SIM_H, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  copyUniforms.u_src.value = tex;
  renderer.setRenderTarget(rtRead);
  renderer.render(copyScene, orthoCam);
  renderer.setRenderTarget(null);
  tex.dispose();
}

uploadSeed('clear');

// ---------------------------------------------------------------------------
// Stepping & display
// ---------------------------------------------------------------------------

// View used for the most recent buffer state. Each frame we remap from this
// to the current view, then step. Cells whose world coord is outside the new
// viewport drop off the buffer; that's what gives the cull-on-zoom-in
// behavior the user wants.
const prevView = { cx: 0, cy: 0, zoom: 1.0 };

function viewChanged() {
  return view.cx   !== prevView.cx
      || view.cy   !== prevView.cy
      || view.zoom !== prevView.zoom;
}

function commitView() {
  prevView.cx   = view.cx;
  prevView.cy   = view.cy;
  prevView.zoom = view.zoom;
}

function remapBuffer() {
  remapUniforms.u_src.value = rtRead.texture;
  remapUniforms.u_res.value.set(SIM_W, SIM_H);
  remapUniforms.u_src_center.value.set(prevView.cx, prevView.cy);
  remapUniforms.u_src_zoom.value = prevView.zoom;
  remapUniforms.u_dst_center.value.set(view.cx, view.cy);
  remapUniforms.u_dst_zoom.value = view.zoom;
  renderer.setRenderTarget(rtWrite);
  renderer.render(remapScene, orthoCam);
  renderer.setRenderTarget(null);
  [rtRead, rtWrite] = [rtWrite, rtRead];
  commitView();
}

function step() {
  if (viewChanged()) remapBuffer();
  stepUniforms.u_state.value = rtRead.texture;
  stepUniforms.u_res.value.set(SIM_W, SIM_H);
  stepUniforms.u_view_center.value.set(view.cx, view.cy);
  stepUniforms.u_view_zoom.value = view.zoom;
  renderer.setRenderTarget(rtWrite);
  renderer.render(stepScene, orthoCam);
  renderer.setRenderTarget(null);
  [rtRead, rtWrite] = [rtWrite, rtRead];
}

// Stamp using a canvas UV (= buffer UV, since buffer is canvas-sized). Brush
// radius is in canvas pixels so it feels like a constant on-screen brush at
// any zoom.
function stamp(canvasU, canvasV, mode) {
  if (viewChanged()) remapBuffer();
  stampUniforms.u_src.value = rtRead.texture;
  stampUniforms.u_res.value.set(SIM_W, SIM_H);
  stampUniforms.u_pos.value.set(canvasU, canvasV);
  stampUniforms.u_radius_px.value = params.brushRadius;
  stampUniforms.u_value.value = params.brushValue;
  stampUniforms.u_mode.value = mode;
  renderer.setRenderTarget(rtWrite);
  renderer.render(stampScene, orthoCam);
  renderer.setRenderTarget(null);
  [rtRead, rtWrite] = [rtWrite, rtRead];
}

function display() {
  displayUniforms.u_state.value = rtRead.texture;
  renderer.setRenderTarget(null);
  renderer.render(displayScene, orthoCam);
}

// ---------------------------------------------------------------------------
// Params + GUI
// ---------------------------------------------------------------------------

const params = {
  paused:        false,
  stepsPerFrame: 1,
  brushRadius:  14,
  brushValue:    1.0,
  bgColor:      '#14141e',
  cellColor:    '#ffd86b',
  noise:         {},
  seedCircles: () => uploadSeed('circles'),
  seedNoise:   () => uploadSeed('noise'),
  clear:       () => uploadSeed('clear'),
  resetParams: () => resetParams(),
};
// Initial spatial variation on a few rules so the pastel background reads on
// first load. Other rules start flat (no spatial variation) — toggle their
// `flat` off in the GUI to bring them in.
const NOISE_INTRO = {
  b1:      { flat: false, dev: 0.06,  scale: 3.0, offsetX:  0.0, offsetY:  0.0 },
  d2:      { flat: false, dev: 0.08,  scale: 2.5, offsetX:  7.3, offsetY:  0.0 },
  alpha_n: { flat: false, dev: 0.025, scale: 4.5, offsetX:  0.0, offsetY: 11.7 },
};
for (const r of RULES) {
  const intro = NOISE_INTRO[r.key] ?? {};
  params.noise[r.key] = {
    base:    r.base,
    dev:     intro.dev     ?? 0.0,
    scale:   intro.scale   ?? 4.0,
    offsetX: intro.offsetX ?? 0.0,
    offsetY: intro.offsetY ?? 0.0,
    flat:    intro.flat    ?? true,
    color:   r.color,
  };
}
const noiseDefaults     = JSON.parse(JSON.stringify(params.noise));
const bgColorDefault    = params.bgColor;
const cellColorDefault  = params.cellColor;

function resetParams() {
  for (const r of RULES) {
    Object.assign(params.noise[r.key], noiseDefaults[r.key]);
  }
  params.bgColor   = bgColorDefault;
  params.cellColor = cellColorDefault;
  applyParams();
  refreshUI();
}

function applyParams() {
  for (const r of RULES) {
    const p = params.noise[r.key];
    const flatI = p.flat ? 1 : 0;

    stepUniforms[`u_${r.key}_base`].value   = p.base;
    stepUniforms[`u_${r.key}_dev`].value    = p.dev;
    stepUniforms[`u_${r.key}_scale`].value  = p.scale;
    stepUniforms[`u_${r.key}_offset`].value.set(p.offsetX, p.offsetY);
    stepUniforms[`u_${r.key}_flat`].value   = flatI;

    displayUniforms[`u_${r.key}_base`].value    = p.base;
    displayUniforms[`u_${r.key}_dev`].value     = p.dev;
    displayUniforms[`u_${r.key}_scale`].value   = p.scale;
    displayUniforms[`u_${r.key}_offset`].value.set(p.offsetX, p.offsetY);
    displayUniforms[`u_${r.key}_flat`].value    = flatI;
    displayUniforms[`u_${r.key}_dev_max`].value = r.devMax;

    const c = new THREE.Color(p.color);
    displayUniforms[`u_${r.key}_pastel`].value.set(c.r, c.g, c.b);
  }

  // u_rb_max bounds the convolution loop. With noise on, local rb can swing
  // up to base + dev (snoise peaks at ~1). When flat, the kernel collapses
  // back to the base radius.
  const rb = params.noise.rb;
  stepUniforms.u_rb_max.value = rb.flat ? rb.base : (rb.base + rb.dev);

  const bg = new THREE.Color(params.bgColor);
  displayUniforms.u_bg_color.value.set(bg.r, bg.g, bg.b);
  const cc = new THREE.Color(params.cellColor);
  displayUniforms.u_cell_color.value.set(cc.r, cc.g, cc.b);
}

// Custom bottom-bar UI. Markup lives in index.html; this just wires controls
// to params + applyParams and exposes setPaused/refreshUI for outside callers
// (keyboard shortcuts, resetParams).
const $ = sel => document.querySelector(sel);
const fmt = (v, dp) => Number(v).toFixed(dp);

let playBtnEl;
function setPaused(p) {
  params.paused = p;
  if (playBtnEl) {
    playBtnEl.textContent = p ? '▶' : '⏸';
    playBtnEl.title = p ? 'Play (Space)' : 'Pause (Space)';
  }
}

function bindRange(rngSel, valSel, setter, dp = 0) {
  const rng = $(rngSel);
  const val = $(valSel);
  const upd = () => {
    const v = parseFloat(rng.value);
    setter(v);
    if (val) val.textContent = fmt(v, dp);
  };
  rng.addEventListener('input', upd);
  upd();
  return rng;
}

function buildRulesTable() {
  const tbody = $('#rules-tbody');
  tbody.innerHTML = '';
  for (const r of RULES) {
    const p = params.noise[r.key];
    const tr = document.createElement('tr');
    tr.dataset.rule = r.key;
    tr.innerHTML = `
      <td class="rule-name">${r.key}</td>
      <td class="rule-cell"><label class="switch"><input type="checkbox" data-prop="active"${!p.flat ? ' checked' : ''}><span class="switch-track"></span></label></td>
      <td class="rule-cell"><input type="color" data-prop="color" value="${p.color}"></td>
      <td class="rule-cell">
        <input type="range" data-prop="dev" min="0" max="${r.devMax}" step="${r.devMax / 200}" value="${p.dev}">
        <span class="rule-val" data-val="dev">${fmt(p.dev, 3)}</span>
      </td>
      <td class="rule-cell">
        <input type="range" data-prop="scale" min="0.25" max="32" step="0.25" value="${p.scale}">
        <span class="rule-val" data-val="scale">${fmt(p.scale, 2)}</span>
      </td>
      <td class="rule-cell">
        <input type="range" data-prop="offsetX" min="-100" max="100" step="0.1" value="${p.offsetX}">
        <span class="rule-val" data-val="offsetX">${fmt(p.offsetX, 1)}</span>
      </td>
      <td class="rule-cell">
        <input type="range" data-prop="offsetY" min="-100" max="100" step="0.1" value="${p.offsetY}">
        <span class="rule-val" data-val="offsetY">${fmt(p.offsetY, 1)}</span>
      </td>
    `;
    tbody.appendChild(tr);
  }

  tbody.addEventListener('input', e => {
    const tr   = e.target.closest('tr');
    const prop = e.target.dataset.prop;
    if (!tr || !prop) return;
    const p = params.noise[tr.dataset.rule];
    if (prop === 'active')      p.flat  = !e.target.checked;
    else if (prop === 'color')  p.color = e.target.value;
    else                        p[prop] = parseFloat(e.target.value);
    const valEl = tr.querySelector(`[data-val="${prop}"]`);
    if (valEl) {
      const dp = prop === 'dev' ? 3 : prop === 'scale' ? 2 : 1;
      valEl.textContent = fmt(p[prop], dp);
    }
    applyParams();
  });
}

function refreshUI() {
  $('#rng-speed').value = params.stepsPerFrame;
  $('#val-speed').textContent = params.stepsPerFrame;
  $('#rng-brush').value = params.brushRadius;
  $('#val-brush').textContent = params.brushRadius;
  $('#rng-value').value = params.brushValue;
  $('#val-value').textContent = fmt(params.brushValue, 2);
  $('#col-bg').value   = params.bgColor;
  $('#col-cell').value = params.cellColor;
  for (const r of RULES) {
    const p  = params.noise[r.key];
    const tr = document.querySelector(`tr[data-rule="${r.key}"]`);
    if (!tr) continue;
    tr.querySelector('[data-prop="active"]').checked = !p.flat;
    tr.querySelector('[data-prop="color"]').value    = p.color;
    for (const k of ['dev','scale','offsetX','offsetY']) {
      tr.querySelector(`[data-prop="${k}"]`).value = p[k];
      const dp = k === 'dev' ? 3 : k === 'scale' ? 2 : 1;
      tr.querySelector(`[data-val="${k}"]`).textContent = fmt(p[k], dp);
    }
  }
}

function bindUI() {
  playBtnEl = $('#btn-play');
  playBtnEl.addEventListener('click', () => setPaused(!params.paused));
  setPaused(params.paused);

  $('#btn-step').addEventListener('click', () => step());

  bindRange('#rng-speed', '#val-speed', v => { params.stepsPerFrame = v|0; }, 0);
  bindRange('#rng-brush', '#val-brush', v => { params.brushRadius   = v|0; }, 0);
  bindRange('#rng-value', '#val-value', v => { params.brushValue    = v;   }, 2);

  $('#btn-clear'  ).addEventListener('click', () => uploadSeed('clear'));
  $('#btn-circles').addEventListener('click', () => uploadSeed('circles'));
  $('#btn-noise'  ).addEventListener('click', () => uploadSeed('noise'));

  $('#btn-reset-view').addEventListener('click', resetView);

  $('#col-bg'  ).addEventListener('input', e => { params.bgColor   = e.target.value; applyParams(); });
  $('#col-cell').addEventListener('input', e => { params.cellColor = e.target.value; applyParams(); });

  const panel = $('#rules-panel');
  const btn   = $('#btn-rules');
  btn.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  buildRulesTable();
  $('#btn-reset-rules').addEventListener('click', () => resetParams());
}

bindUI();
applyParams();

// ---------------------------------------------------------------------------
// View (pan + zoom in world coords), pointer painting
// ---------------------------------------------------------------------------

// view.cx/cy = world coord at canvas center (float). view.zoom = canvas
// pixels per world unit (display magnification). Cell world size is fixed;
// zoom scales the cells visually — at zoom=1 a cell is 1 canvas pixel; at
// zoom=4 a cell is 4 canvas pixels; at zoom=0.25 a cell is 0.25 canvas
// pixels.
const view = { cx: 0, cy: 0, zoom: 1.0 };
const ZOOM_MIN = 0.05;
const ZOOM_MAX = 64;

// Canvas mouse position → world coord under cursor.
function canvasToWorld(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  const px = (e.clientX - rect.left) - rect.width  / 2;
  const py = rect.height / 2 - (e.clientY - rect.top); // flip y so up = +world.y
  return [view.cx + px / view.zoom, view.cy + py / view.zoom];
}

function applyView() {
  view.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, view.zoom));
  displayUniforms.u_view_center.value.set(view.cx, view.cy);
  displayUniforms.u_view_zoom.value = view.zoom;
  displayUniforms.u_res.value.set(SIM_W, SIM_H);
}
function resetView() {
  view.cx = 0; view.cy = 0; view.zoom = 1.0;
  applyView();
}
applyView();

// Mouse to canvas UV (also = buffer UV, since buffer = canvas).
function canvasUv(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  return [
    (e.clientX - rect.left) / rect.width,
    1 - (e.clientY - rect.top) / rect.height,
  ];
}

let painting = 0;     // 0 none, 1 paint, 2 erase
let panning  = false;
let panLast  = { x: 0, y: 0 };

renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

renderer.domElement.addEventListener('pointerdown', e => {
  if (e.button === 1) {                       // middle = pan
    panning = true;
    panLast.x = e.clientX; panLast.y = e.clientY;
    e.preventDefault();
    return;
  }
  painting = (e.button === 2) ? 2 : 1;        // right = erase, left = paint
  const [u, v] = canvasUv(e);
  stamp(u, v, painting === 2 ? 1 : 0);
});

renderer.domElement.addEventListener('pointermove', e => {
  if (panning) {
    const dx = e.clientX - panLast.x;
    const dy = e.clientY - panLast.y;
    panLast.x = e.clientX; panLast.y = e.clientY;
    // dx canvas pixels right → world center moves dx/zoom world units left.
    view.cx -= dx / view.zoom;
    view.cy += dy / view.zoom;                  // canvas y flipped vs world y
    applyView();
    return;
  }
  if (!painting) return;
  const [u, v] = canvasUv(e);
  stamp(u, v, painting === 2 ? 1 : 0);
});

window.addEventListener('pointerup',     () => { painting = 0; panning = false; });
window.addEventListener('pointercancel', () => { painting = 0; panning = false; });

renderer.domElement.addEventListener('wheel', e => {
  e.preventDefault();
  // Anchor zoom to the cursor: keep the world point under the cursor fixed
  // so the user zooms into whatever they're pointing at. view.zoom changes
  // canvas-px-per-world-unit; the buffer is untouched.
  const [wx, wy] = canvasToWorld(e);
  const factor = Math.exp(-e.deltaY * 0.0015);  // wheel-up → zoom in (zoom grows)
  view.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, view.zoom * factor));
  // Re-anchor: the world point (wx, wy) must still sit under the same canvas pixel.
  const rect = renderer.domElement.getBoundingClientRect();
  const px = (e.clientX - rect.left) - rect.width  / 2;
  const py = rect.height / 2 - (e.clientY - rect.top);
  view.cx = wx - px / view.zoom;
  view.cy = wy - py / view.zoom;
  applyView();
}, { passive: false });

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

window.addEventListener('keydown', e => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.code === 'Space')  { setPaused(!params.paused); e.preventDefault(); }
  if (e.code === 'KeyR')   { uploadSeed('circles'); }
  if (e.code === 'KeyN')   { uploadSeed('noise'); }
  if (e.code === 'KeyC')   { uploadSeed('clear'); }
  if (e.code === 'KeyF')   { resetView(); }
  if (e.code === 'Period') { step(); }
});

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

function reallocateBuffers(w, h) {
  if (w === SIM_W && h === SIM_H) return;
  rtRead.dispose();
  rtWrite.dispose();
  SIM_W = w;
  SIM_H = h;
  rtRead  = new THREE.WebGLRenderTarget(SIM_W, SIM_H, rtOpts);
  rtWrite = new THREE.WebGLRenderTarget(SIM_W, SIM_H, rtOpts);
  uploadSeed('clear');
  // Anything painted is gone — fresh viewport-sized buffer.
  commitView();
}

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  reallocateBuffers(window.innerWidth, window.innerHeight);
  applyView();
});

let frame = 0;
function animate() {
  requestAnimationFrame(animate);
  if (!params.paused) {
    for (let i = 0; i < params.stepsPerFrame; i++) step();
  } else if (viewChanged()) {
    remapBuffer();
  }
  display();
  if (frame === 0) console.log('[smoothlife] first frame rendered');
  frame++;
}
animate();
