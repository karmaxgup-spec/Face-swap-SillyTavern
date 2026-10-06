import * as ST from '../../../../script.js';
import { getContext, extension_settings, renderExtensionTemplateAsync } from '../../../extensions.js';
import { saveBase64AsFile } from '../../../utils.js';

const MODULE = 'faceswap';
const FOLDER = 'ST-FaceSwap';
const BASE = `/scripts/extensions/third-party/${FOLDER}`;
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const ARC_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
const FFHQ_512 = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]];

const defaults = {
    enabled: false,
    reference: '',
    modelsPath: `${BASE}/models`,
    remoteBase: 'https://huggingface.co/YOUR_USER/ST-FaceSwap-models/resolve/main',
    provider: 'webgpu',
    detFile: 'det_2.5g.onnx',
    arcFile: 'w600k_r50.onnx',
    swapFile: 'inswapper_128_fp16.onnx',
    emapFile: 'emap.bin',
    enhanceFile: 'GFPGANv1.4.onnx',
    enhance: true,
    enhanceWeight: 0.4,
    colorMatch: true,
    minFace: 112,
    allFaces: false,
    identityWarn: true,
    maxSide: 1024,
};

const S = () => extension_settings[MODULE];

/* ------------------------------------------------------------------ */
/* onnxruntime-web (library only; models are loaded/released per run)  */
/* ------------------------------------------------------------------ */

let ortPromise = null;
function loadOrt() {
    if (window.ort) return Promise.resolve(window.ort);
    if (ortPromise) return ortPromise;
    ortPromise = new Promise((resolve, reject) => {
        const el = document.createElement('script');
        el.src = ORT_CDN + 'ort.all.min.js';
        el.onload = () => { window.ort.env.wasm.wasmPaths = ORT_CDN; resolve(window.ort); };
        el.onerror = () => { ortPromise = null; reject(new Error('Could not load onnxruntime-web from CDN')); };
        document.head.appendChild(el);
    });
    return ortPromise;
}

async function openSession(file) {
    const ort = await loadOrt();
    const bytes = await resolveFileBytes(file);
    const eps = S().provider === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'];
    try {
        return await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: 'all' });
    } catch (e) {
        throw new Error(`Failed to load model "${file}": ${e.message ?? e}`);
    }
}

/* Ready-made models: try local modelsPath first, fall back to remoteBase.
   Bytes are cached on disk (Cache API) via the Download button, never kept as
   live sessions — every swap still cold-starts and releases everything. */
const CACHE_NAME = 'faceswap-models-v1';

async function fetchBytesCached(url) {
    try {
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(url);
        if (hit) return new Uint8Array(await hit.arrayBuffer());
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        const bytes = new Uint8Array(await res.arrayBuffer());
        try {
            await cache.put(url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
        } catch { /* quota etc. — bytes still usable, just not cached */ }
        return bytes;
    } catch (e) {
        if (e.message?.startsWith('HTTP')) throw e;
        // Cache API unavailable (private mode etc.) — plain fetch, nothing kept.
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        return new Uint8Array(await res.arrayBuffer());
    }
}

async function tryFetchBytes(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return new Uint8Array(await res.arrayBuffer());
}

function remoteUrl(file) {
    return `${S().remoteBase.replace(/\/$/, '')}/${file}`;
}

function localUrl(file) {
    return `${S().modelsPath.replace(/\/$/, '')}/${file}`;
}

// Order: browser disk cache (remote URL) -> local path -> remote download.
// Returns a fresh Uint8Array each call; caller lets it GC after session creation.
async function resolveFileBytes(file) {
    const errors = [];
    // 1) disk-cached remote copy (pre-downloaded on install)
    if (S().remoteBase) {
        try {
            const cache = await caches.open(CACHE_NAME);
            const hit = await cache.match(remoteUrl(file));
            if (hit) return new Uint8Array(await hit.arrayBuffer());
        } catch { /* ignore, fall through */ }
    }
    // 2) local models folder (legacy / offline override)
    if (S().modelsPath) {
        try {
            return await tryFetchBytes(localUrl(file));
        } catch (e) { errors.push(`local: ${e.message}`); }
    }
    // 3) remote download + populate disk cache
    if (S().remoteBase) {
        try {
            return await fetchBytesCached(remoteUrl(file));
        } catch (e) { errors.push(`remote: ${e.message}`); }
    }
    throw new Error(`Could not get "${file}" (${errors.join('; ') || 'no source configured'}). Set Models folder or Remote base in settings.`);
}

async function getEmap() {
    const bytes = await resolveFileBytes(S().emapFile);
    const emap = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    if (emap.length !== 512 * 512) throw new Error('emap.bin has the wrong size. Re-upload the maintainer build.');
    return emap;
}

function requiredFiles() {
    const s = S();
    const files = [s.detFile, s.arcFile, s.swapFile, s.emapFile];
    if (s.enhance) files.push(s.enhanceFile);
    return [...new Set(files.filter(Boolean))];
}

// Install-time preload: downloads bytes to disk cache only, creates no sessions.
async function preloadModels(progress) {
    const files = requiredFiles();
    for (let i = 0; i < files.length; i++) {
        progress?.(`Downloading model ${i + 1}/${files.length}: ${files[i]}…`);
        await resolveFileBytes(files[i]);
    }
}

/* ------------------------------------------------------------------ */
/* Image + geometry helpers                                            */
/* ------------------------------------------------------------------ */

const toUrl = (p) => (/^(data:|https?:|blob:|\/)/.test(p) ? p : '/' + p);
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

function toCanvas(img) {
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    c.getContext('2d').putImageData(img, 0, 0);
    return c;
}

async function loadImage(src, maxSide) {
    const res = await fetch(toUrl(src));
    if (!res.ok) throw new Error(`Could not fetch image (${res.status})`);
    const bmp = await createImageBitmap(await res.blob());
    const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * k), h = Math.round(bmp.height * k);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingQuality = 'high';
    g.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    return g.getImageData(0, 0, w, h);
}

// 2D similarity transform (Umeyama) mapping src points -> dst points. Returns [a,b,tx,c,d,ty].
function similarity(src, dst) {
    const n = src.length;
    let sx = 0, sy = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
    sx /= n; sy /= n; dx /= n; dy /= n;
    let dot = 0, cross = 0, den = 0;
    for (let i = 0; i < n; i++) {
        const px = src[i][0] - sx, py = src[i][1] - sy, qx = dst[i][0] - dx, qy = dst[i][1] - dy;
        dot += px * qx + py * qy;
        cross += px * qy - py * qx;
        den += px * px + py * py;
    }
    const c = dot / den, s = cross / den;
    if (!Number.isFinite(c) || !Number.isFinite(s)) throw new Error('Bad face landmarks (degenerate alignment)');
    return [c, -s, dx - (c * sx - s * sy), s, c, dy - (s * sx + c * sy)];
}

function invert(M) {
    const [a, b, tx, c, d, ty] = M;
    const det = a * d - b * c;
    if (!det) throw new Error('Bad face landmarks (singular transform)');
    const ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
    return [ia, ib, -(ia * tx + ib * ty), ic, id, -(ic * tx + id * ty)];
}

const scalePts = (pts, k) => pts.map(([x, y]) => [x * k, y * k]);

function sampleBilinear(img, x, y, out) {
    const { data, width: w, height: h } = img;
    x = clamp(x, 0, w - 1); y = clamp(y, 0, h - 1);
    const x0 = x | 0, y0 = y | 0, x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
    const fx = x - x0, fy = y - y0;
    const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4, i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
    for (let c = 0; c < 3; c++) {
        out[c] = data[i00 + c] * (1 - fx) * (1 - fy) + data[i10 + c] * fx * (1 - fy)
            + data[i01 + c] * (1 - fx) * fy + data[i11 + c] * fx * fy;
    }
}

// Warp the region of `img` described by M (img -> crop coords) into a size x size RGBA crop.
function warpToCrop(img, M, size) {
    const inv = invert(M);
    const data = new Uint8ClampedArray(size * size * 4);
    const tmp = [0, 0, 0];
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            sampleBilinear(img, inv[0] * x + inv[1] * y + inv[2], inv[3] * x + inv[4] * y + inv[5], tmp);
            const o = (y * size + x) * 4;
            data[o] = tmp[0]; data[o + 1] = tmp[1]; data[o + 2] = tmp[2]; data[o + 3] = 255;
        }
    }
    return { data, width: size, height: size };
}

// Paste a crop back into dst through a soft mask.
function paste(dst, crop, size, M, maskFn, weight = 1) {
    const inv = invert(M);
    const xs = [], ys = [];
    for (const [cx, cy] of [[0, 0], [size, 0], [0, size], [size, size]]) {
        xs.push(inv[0] * cx + inv[1] * cy + inv[2]);
        ys.push(inv[3] * cx + inv[4] * cy + inv[5]);
    }
    const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(dst.width - 1, Math.ceil(Math.max(...xs)));
    const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(dst.height - 1, Math.ceil(Math.max(...ys)));
    const tmp = [0, 0, 0];
    const d = dst.data;
    for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
            const cx = M[0] * x + M[1] * y + M[2], cy = M[3] * x + M[4] * y + M[5];
            if (cx < 0 || cy < 0 || cx > size - 1 || cy > size - 1) continue;
            const a = maskFn(cx / (size - 1), cy / (size - 1)) * weight;
            if (a <= 0.002) continue;
            sampleBilinear(crop, cx, cy, tmp);
            const i = (y * dst.width + x) * 4;
            d[i] = d[i] * (1 - a) + tmp[0] * a;
            d[i + 1] = d[i + 1] * (1 - a) + tmp[1] * a;
            d[i + 2] = d[i + 2] * (1 - a) + tmp[2] * a;
        }
    }
}

const ellipseMask = (inner, outer) => (u, v) => 1 - smoothstep(inner, outer, Math.hypot((u - 0.5) * 2, (v - 0.5) * 2));

function toNCHW(img, mean, std) {
    const { data, width: w, height: h } = img;
    const n = w * h, f = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        f[i] = (data[i * 4] - mean) / std;
        f[n + i] = (data[i * 4 + 1] - mean) / std;
        f[2 * n + i] = (data[i * 4 + 2] - mean) / std;
    }
    return f;
}

function normalize(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    s = Math.sqrt(s) || 1;
    const o = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) o[i] = v[i] / s;
    return o;
}

/* ------------------------------------------------------------------ */
/* Colour matching (Reinhard, Lab)                                     */
/* ------------------------------------------------------------------ */

function rgb2lab(r, g, b) {
    const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const R = lin(r), G = lin(g), B = lin(b);
    const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
    const Y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
    const Z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / 1.08883;
    const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    const fx = f(X), fy = f(Y), fz = f(Z);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function lab2rgb(L, a, b) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const inv = (t) => { const t3 = t * t * t; return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787; };
    const X = inv(fx) * 0.95047, Y = inv(fy), Z = inv(fz) * 1.08883;
    const R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    const G = -0.969266 * X + 1.8760108 * Y + 0.041556 * Z;
    const B = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    const gam = (v) => { v = clamp(v, 0, 1); return 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055); };
    return [gam(R), gam(G), gam(B)];
}

// Shift `fake` (RGBA crop) toward the colour statistics of `ref` (RGBA crop) inside the face mask.
function reinhard(fake, ref, maskFn) {
    const n = fake.width * fake.width, size = fake.width;
    const fl = new Float32Array(n * 3), rl = new Float32Array(n * 3);
    const sf = [0, 0, 0], sr = [0, 0, 0], qf = [0, 0, 0], qr = [0, 0, 0];
    let cnt = 0;
    for (let i = 0; i < n; i++) {
        const x = i % size, y = (i / size) | 0;
        const lf = rgb2lab(fake.data[i * 4], fake.data[i * 4 + 1], fake.data[i * 4 + 2]);
        const lr = rgb2lab(ref.data[i * 4], ref.data[i * 4 + 1], ref.data[i * 4 + 2]);
        for (let c = 0; c < 3; c++) { fl[i * 3 + c] = lf[c]; rl[i * 3 + c] = lr[c]; }
        if (maskFn(x / (size - 1), y / (size - 1)) > 0.5) {
            cnt++;
            for (let c = 0; c < 3; c++) { sf[c] += lf[c]; sr[c] += lr[c]; qf[c] += lf[c] * lf[c]; qr[c] += lr[c] * lr[c]; }
        }
    }
    if (cnt < 100) return fake;
    const mf = sf.map((v) => v / cnt), mr = sr.map((v) => v / cnt);
    const stf = qf.map((v, c) => Math.sqrt(Math.max(v / cnt - mf[c] * mf[c], 1e-6)));
    const str = qr.map((v, c) => Math.sqrt(Math.max(v / cnt - mr[c] * mr[c], 1e-6)));
    const ratio = stf.map((v, c) => clamp(str[c] / v, 0.7, 1.3));
    const out = new Uint8ClampedArray(fake.data.length);
    for (let i = 0; i < n; i++) {
        const lab = [0, 1, 2].map((c) => (fl[i * 3 + c] - mf[c]) * ratio[c] + mr[c]);
        const rgb = lab2rgb(lab[0], lab[1], lab[2]);
        out[i * 4] = rgb[0]; out[i * 4 + 1] = rgb[1]; out[i * 4 + 2] = rgb[2]; out[i * 4 + 3] = 255;
    }
    return { data: out, width: size, height: size };
}

/* ------------------------------------------------------------------ */
/* Models                                                              */
/* ------------------------------------------------------------------ */

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / ua;
}

async function detect(session, ort, img, minFace) {
    const SZ = 640, scale = Math.min(SZ / img.width, SZ / img.height);
    const nw = Math.round(img.width * scale), nh = Math.round(img.height * scale);
    const c = document.createElement('canvas');
    c.width = c.height = SZ;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.fillStyle = '#000'; g.fillRect(0, 0, SZ, SZ);
    g.imageSmoothingQuality = 'high';
    g.drawImage(toCanvas(img), 0, 0, nw, nh);
    const px = g.getImageData(0, 0, SZ, SZ);

    const input = new ort.Tensor('float32', toNCHW(px, 127.5, 128), [1, 3, SZ, SZ]);
    const outs = await session.run({ [session.inputNames[0]]: input });
    const names = session.outputNames;
    if (names.length < 9) throw new Error(`Unexpected detector outputs (got ${names.length}, need 9). Use SCRFD det_2.5g.onnx from the buffalo_m pack.`);

    const dets = [];
    [8, 16, 32].forEach((stride, i) => {
        const sc = outs[names[i]].data, bb = outs[names[i + 3]].data, kp = outs[names[i + 6]].data;
        const fw = SZ / stride;
        for (let k = 0; k < sc.length; k++) {
            if (sc[k] < 0.5) continue;
            const cell = k >> 1;
            const cx = (cell % fw) * stride, cy = ((cell / fw) | 0) * stride;
            const kps = [];
            for (let j = 0; j < 5; j++) {
                kps.push([(cx + kp[k * 10 + 2 * j] * stride) / scale, (cy + kp[k * 10 + 2 * j + 1] * stride) / scale]);
            }
            dets.push({
                score: sc[k],
                x1: (cx - bb[k * 4] * stride) / scale, y1: (cy - bb[k * 4 + 1] * stride) / scale,
                x2: (cx + bb[k * 4 + 2] * stride) / scale, y2: (cy + bb[k * 4 + 3] * stride) / scale,
                kps,
            });
        }
    });

    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every((k) => iou(k, d) <= 0.4)) keep.push(d);
    const area = (f) => (f.x2 - f.x1) * (f.y2 - f.y1);
    return keep
        .filter((f) => Math.min(f.x2 - f.x1, f.y2 - f.y1) >= minFace)
        .sort((a, b) => area(b) - area(a));
}

async function embed(session, ort, crop112) {
    const t = new ort.Tensor('float32', toNCHW(crop112, 127.5, 127.5), [1, 3, 112, 112]);
    const out = await session.run({ [session.inputNames[0]]: t });
    return Float32Array.from(out[session.outputNames[0]].data);
}

function makeLatent(emb, emap) {
    const x = normalize(emb), out = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
        const xi = x[i], row = i * 512;
        for (let j = 0; j < 512; j++) out[j] += xi * emap[row + j];
    }
    return normalize(out);
}

function resizeCrop(crop, size) {
    const src = toCanvas(crop);
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingQuality = 'high';
    g.drawImage(src, 0, 0, size, size);
    return g.getImageData(0, 0, size, size);
}

/* ------------------------------------------------------------------ */
/* Pipeline: every call cold-starts each model and releases it after   */
/* ------------------------------------------------------------------ */

async function swapImage(targetSrc, refSrc, progress) {
    const s = S();
    const ort = await loadOrt();
    const target = await loadImage(targetSrc, s.maxSide);
    const ref = await loadImage(refSrc, 1024);

    // 1) detect (reference + target) -> release
    progress('Detecting faces…');
    let refFace, faces;
    {
        const det = await openSession(s.detFile);
        try {
            const r = await detect(det, ort, ref, 0);
            if (!r.length) throw new Error('No face found in the reference image');
            refFace = r[0];
            const all = await detect(det, ort, target, s.minFace);
            faces = s.allFaces ? all : all.slice(0, 1);
        } finally { await det.release(); }
    }
    if (!faces.length) return { result: null, reason: `No face >= ${s.minFace}px found` };

    // 2) source embedding, swap, identity check
    progress('Loading swap models…');
    const emap = await getEmap();

    const arc = await openSession(s.arcFile);
    let swapper = null;
    const jobs = [];
    const cosines = [];
    try {
        const refCrop = warpToCrop(ref, similarity(refFace.kps, scalePts(ARC_DST, 1)), 112);
        const srcEmb = await embed(arc, ort, refCrop);
        const latent = makeLatent(srcEmb, emap);
        const srcNorm = normalize(srcEmb);

        swapper = await openSession(s.swapFile);
        const nIn = swapper.inputNames;
        const tName = nIn.find((n) => /target/i.test(n)) ?? nIn[0];
        const sName = nIn.find((n) => /source/i.test(n)) ?? nIn[1];

        for (let i = 0; i < faces.length; i++) {
            progress(`Swapping face ${i + 1}/${faces.length}…`);
            const f = faces[i];
            const M = similarity(f.kps, scalePts(ARC_DST, 128 / 112));
            const orig = warpToCrop(target, M, 128);
            const feeds = {
                [tName]: new ort.Tensor('float32', toNCHW(orig, 0, 255), [1, 3, 128, 128]),
                [sName]: new ort.Tensor('float32', latent, [1, 512]),
            };
            const out = (await swapper.run(feeds))[swapper.outputNames[0]].data;
            const n = 128 * 128, fake = new Uint8ClampedArray(n * 4);
            for (let p = 0; p < n; p++) {
                fake[p * 4] = clamp(out[p] * 255, 0, 255);
                fake[p * 4 + 1] = clamp(out[n + p] * 255, 0, 255);
                fake[p * 4 + 2] = clamp(out[2 * n + p] * 255, 0, 255);
                fake[p * 4 + 3] = 255;
            }
            let crop = { data: fake, width: 128, height: 128 };
            const mask = ellipseMask(0.72, 0.97);
            if (s.colorMatch) crop = reinhard(crop, orig, mask);

            if (s.identityWarn) {
                const e = normalize(await embed(arc, ort, resizeCrop(crop, 112)));
                let cos = 0;
                for (let k = 0; k < 512; k++) cos += e[k] * srcNorm[k];
                cosines.push(cos);
            }
            jobs.push({ face: f, M, crop, mask });
        }
    } finally {
        if (swapper) await swapper.release();
        await arc.release();
    }

    // 3) composite onto the full-resolution image
    progress('Blending…');
    const result = new ImageData(new Uint8ClampedArray(target.data), target.width, target.height);
    for (const j of jobs) paste(result, j.crop, 128, j.M, j.mask, 1);

    // 4) optional light enhance on the composited result
    if (s.enhance && s.enhanceWeight > 0) {
        progress('Enhancing…');
        const gfp = await openSession(s.enhanceFile);
        try {
            for (const j of jobs) {
                const M = similarity(j.face.kps, FFHQ_512);
                const crop = warpToCrop(result, M, 512);
                const t = new ort.Tensor('float32', toNCHW(crop, 127.5, 127.5), [1, 3, 512, 512]);
                const o = (await gfp.run({ [gfp.inputNames[0]]: t }))[gfp.outputNames[0]].data;
                const n = 512 * 512, enh = new Uint8ClampedArray(n * 4);
                for (let p = 0; p < n; p++) {
                    enh[p * 4] = clamp((o[p] * 0.5 + 0.5) * 255, 0, 255);
                    enh[p * 4 + 1] = clamp((o[n + p] * 0.5 + 0.5) * 255, 0, 255);
                    enh[p * 4 + 2] = clamp((o[2 * n + p] * 0.5 + 0.5) * 255, 0, 255);
                    enh[p * 4 + 3] = 255;
                }
                paste(result, { data: enh, width: 512, height: 512 }, 512, M, ellipseMask(0.55, 0.85), s.enhanceWeight);
            }
        } finally { await gfp.release(); }
    }

    return { result, cosines };
}

/* ------------------------------------------------------------------ */
/* SillyTavern glue                                                    */
/* ------------------------------------------------------------------ */

let toast = null;
function progress(text) {
    if (toast) toastr.clear(toast);
    toast = toastr.info(text, 'Face swap', { timeOut: 0, extendedTimeOut: 0, tapToDismiss: false });
}
function endProgress() { if (toast) toastr.clear(toast); toast = null; }

// Generated images (SD extension, /sd, tool-call fix) carry generationType / title; user uploads are is_user.
function isGenerated(msg) {
    return !!msg?.extra?.image && !msg.is_user && !msg.extra.faceswapped
        && (msg.extra.generationType !== undefined || msg.extra.title !== undefined);
}

async function processMessage(id, force = false) {
    const ctx = getContext();
    const msg = ctx.chat[id];
    const s = S();
    if (!msg?.extra?.image) { toastr.warning('That message has no image.', 'Face swap'); return; }
    if (!s.reference) { toastr.warning('Set a reference face in the Face Swap settings.', 'Face swap'); return; }
    if (!force && msg.extra.faceswapped) return;

    const original = msg.extra.faceswap_original ?? msg.extra.image;
    try {
        progress('Starting…');
        const { result, cosines, reason } = await swapImage(original, s.reference, progress);
        if (!result) { endProgress(); toastr.warning(reason, 'Face swap'); return; }

        const dataUrl = toCanvas(result).toDataURL('image/jpeg', 0.92);
        const folder = ctx.groupId ? String(ctx.groupId) : (ctx.characters[ctx.characterId]?.name ?? 'faceswap');
        const path = await saveBase64AsFile(dataUrl.split(',')[1], folder, `faceswap_${Date.now()}`, 'jpg');

        const swipes = msg.extra.image_swipes;
        if (Array.isArray(swipes)) {
            const idx = swipes.indexOf(msg.extra.image);
            if (idx >= 0) swipes[idx] = path;
        }
        msg.extra.faceswap_original = original;
        msg.extra.image = path;
        msg.extra.faceswapped = true;

        const el = $(`#chat .mes[mesid="${id}"]`);
        try {
            if (typeof ST.appendMediaToMessage === 'function') ST.appendMediaToMessage(msg, el, false);
            else throw new Error('no appendMediaToMessage');
        } catch {
            el.find('.mes_img').attr('src', toUrl(path));
        }
        await ctx.saveChat?.();

        endProgress();
        const low = cosines?.filter((c) => c < 0.4).length ?? 0;
        if (low) toastr.warning(`Identity match is low on ${low} face(s) (cosine < 0.4).`, 'Face swap');
        else toastr.success('Face swapped.', 'Face swap', { timeOut: 2000 });
    } catch (e) {
        console.error('[FaceSwap]', e);
        endProgress();
        toastr.error(String(e.message ?? e), 'Face swap');
    }
}

// Serial queue so two swaps never hold models in memory at the same time.
let queue = Promise.resolve();
const enqueue = (id, force = false) => { queue = queue.then(() => processMessage(id, force)).catch(() => {}); return queue; };

function onMessageReceived(id) {
    const s = S();
    if (!s?.enabled || !s.reference) return;
    const msg = getContext().chat[id];
    if (isGenerated(msg)) enqueue(Number(id));
}

function lastImageMessageId() {
    const chat = getContext().chat;
    for (let i = chat.length - 1; i >= 0; i--) if (chat[i]?.extra?.image && !chat[i].is_user) return i;
    return -1;
}

/* ------------------------------ settings ------------------------------ */

const save = () => ST.saveSettingsDebounced();

function bindInput(sel, key, kind = 'text') {
    const el = $(sel);
    if (kind === 'check') el.prop('checked', !!S()[key]);
    else el.val(S()[key]);
    el.on(kind === 'check' ? 'change' : 'input change', function () {
        let v = kind === 'check' ? $(this).prop('checked') : $(this).val();
        if (kind === 'number') v = Number(v);
        S()[key] = v;
        save();
    });
}

function setPreview() {
    const img = $('#fs_ref_preview');
    if (S().reference) img.attr('src', S().reference).show(); else img.hide();
}

async function initSettings() {
    extension_settings[MODULE] = Object.assign({}, defaults, extension_settings[MODULE]);
    const html = await renderExtensionTemplateAsync(`third-party/${FOLDER}`, 'settings');
    $('#extensions_settings2').append(html);

    bindInput('#fs_enabled', 'enabled', 'check');
    bindInput('#fs_models_path', 'modelsPath');
    bindInput('#fs_remote_base', 'remoteBase');
    bindInput('#fs_det_file', 'detFile');
    bindInput('#fs_arc_file', 'arcFile');
    bindInput('#fs_swap_file', 'swapFile');
    bindInput('#fs_emap_file', 'emapFile');
    bindInput('#fs_enh_file', 'enhanceFile');
    bindInput('#fs_provider', 'provider');
    bindInput('#fs_color', 'colorMatch', 'check');
    bindInput('#fs_enhance', 'enhance', 'check');
    bindInput('#fs_all_faces', 'allFaces', 'check');
    bindInput('#fs_identity', 'identityWarn', 'check');
    bindInput('#fs_min_face', 'minFace', 'number');
    bindInput('#fs_max_side', 'maxSide', 'number');
    bindInput('#fs_enh_weight', 'enhanceWeight', 'number');
    $('#fs_enh_weight_val').text(S().enhanceWeight);
    $('#fs_enh_weight').on('input', function () { $('#fs_enh_weight_val').text($(this).val()); });

    setPreview();
    $('#fs_ref_file').on('change', async function () {
        const f = this.files?.[0];
        if (!f) return;
        try {
            const bmp = await createImageBitmap(f);
            const k = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
            const c = document.createElement('canvas');
            c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
            c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
            S().reference = c.toDataURL('image/jpeg', 0.92);
            save(); setPreview();
        } catch (e) { toastr.error('Could not read that image.', 'Face swap'); }
        this.value = '';
    });
    $('#fs_ref_clear').on('click', () => { S().reference = ''; save(); setPreview(); });
    $('#fs_test').on('click', () => {
        const id = lastImageMessageId();
        if (id < 0) toastr.info('No generated image in this chat yet.', 'Face swap');
        else enqueue(id, true);
    });

    const setDlStatus = (t) => $('#fs_dl_status').text(t);
    $('#fs_predownload').on('click', async () => {
        setDlStatus('Downloading…');
        try {
            await preloadModels((t) => setDlStatus(t));
            setDlStatus('Models ready (cached on disk, nothing in memory).');
            toastr.success('Face swap models downloaded.', 'Face swap', { timeOut: 2000 });
        } catch (e) {
            console.error('[FaceSwap]', e);
            setDlStatus(`Download failed: ${e.message}`);
            toastr.error(String(e.message ?? e), 'Face swap');
        }
    });
    // Install-time check: report cache status only, never bulk-download without
    // consent (~1GB). User starts the download via the button above.
    setTimeout(async () => {
        try {
            if (!S().remoteBase) { setDlStatus('Set a Remote base, then click Download models.'); return; }
            if (!('caches' in window)) { setDlStatus('Click Download models to fetch them.'); return; }
            const cache = await caches.open(CACHE_NAME);
            for (const f of requiredFiles()) {
                if (!await cache.match(remoteUrl(f))) { setDlStatus('Models not downloaded yet — click Download models.'); return; }
            }
            setDlStatus('Models ready (cached on disk, nothing in memory).');
        } catch (e) { setDlStatus(`Cache check failed: ${e.message} — click Download models.`); }
    }, 0);
}

jQuery(async () => {
    await initSettings();
    ST.eventSource.on(ST.event_types.MESSAGE_RECEIVED, onMessageReceived);

    try {
        const { SlashCommandParser } = await import('../../../slash-commands/SlashCommandParser.js');
        const { SlashCommand } = await import('../../../slash-commands/SlashCommand.js');
        SlashCommandParser.addCommandObject(SlashCommand.fromProps({
            name: 'faceswap',
            helpString: 'Run the face swap on a message with an image (default: the last one). Usage: /faceswap [message id]',
            returns: 'nothing',
            callback: async (_args, value) => {
                const id = value?.toString().trim() ? Number(value) : lastImageMessageId();
                if (Number.isNaN(id) || id < 0) { toastr.info('No image message found.', 'Face swap'); return ''; }
                await enqueue(id, true);
                return '';
            },
        }));
    } catch (e) { console.warn('[FaceSwap] slash command not registered', e); }
});
