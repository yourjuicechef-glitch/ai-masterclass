/**
 * LIVING NEURAL NETWORK — the 3D layer behind every page. "Living Bloom" interaction.
 *
 * One graph in four layers down the -Z axis; the camera flies through them as you scroll:
 *   news      a loose "brain" cloud; every bright node is a real story (cfg.news), newest fires
 *   tools     category hubs in a ring, tools around them; featured tools are bigger + brighter
 *   audience  a globe of followers/1000 nodes — one node per thousand followers, countable
 *   learn     a layered deep network (columns fully connected), the courses
 *
 * INTERACTION (tier 1): the cursor is a neuron — wires snap from it to the nearest dots; the nearest
 * story/tool within 140px BLOOMS (grows, neighbours scatter, a card unfolds on a leader line); every
 * dot near the cursor is pushed by a force field. Per layer: the globe counts the people under the
 * cursor and spins on drag; a category hub opens like a flower and names its tools; the deep network
 * fires a forward pass. Always-on tags explain the layers without any hovering. Tier 2: tap to bloom,
 * plus an idle auto-demo. The network assembles itself on load.
 *
 * THE FORCE FIELD lives in ONE GLSL function used by all four programs (node, edge, spark, halo), so
 * edges stay attached to displaced dots by construction. Offsets are computed in NDC and converted back
 * to view space at each point's own depth (Δx = Δndc·(−z)/P00), which keeps it correct under the
 * camera's view offset. Shared uniforms are shared OBJECTS: one write moves every material.
 */
import * as THREE from '../three/three.module.min.js';

const Z = { news: 0, tools: -80, audience: -160, learn: -240 };
const LAYER_OF = { hero: 0, news: 0, tools: 1, audience: 2, learn: 3 };      // which layer a camera stop "is"
const STOPS = {                         // camera stops: position, look-at, canvas opacity, sideways shift (x width), fog end
	hero:     { pos: [4, 5, 46],     look: [-46, 0, -91],  dim: 1, shift: .06, far: 330 },
	news:     { pos: [14, 3, 33],    look: [0, 0, -2],     dim: 1 },
	tools:    { pos: [14, 3, -50],   look: [0, 0, -82],    dim: 1 },
	audience: { pos: [16, 6, -128],  look: [0, 0, -160],   dim: 1 },
	learn:    { pos: [11, 4, -202],  look: [3, 0, -240],   dim: 1 },
	park:     { pos: [30, 14, 58],   look: [0, 0, -40],    dim: .38 },
};
const INNER_DIM = .3;                   // a layer's own inner page: that layer, dimmed under text, pushed right
const FOG = { near: 26, far: 150 };
const SNAP = 140, KEEP = 170, WIRES = 8, BUILD_S = 1.8;

/* seeded rng: same network every visit, so it reads as a place, not noise */
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ------------------------------------------------------------------ shaders */
const LIME = 'const vec3 LIME = vec3(0.784, 1.0, 0.180);';   // #C8FF2E, the logo's lit dots: moving light in the network is lime (fragment shaders only)
const COMMON = `
uniform float uTime; uniform float uNear; uniform float uFar; uniform float uForce; uniform float uBloom;
uniform float uAspect; uniform float uBuild; uniform float uWarp; uniform vec2 uMouse; uniform vec2 uBloomC;
float fogK(float d){ return 1.0 - smoothstep(uNear, uFar, d); }
// intro: each point flies in from a scatter derived from its own position (no extra attributes, and
// nodes / edges / sparks agree because they share positions). LOW-GAIN smooth functions only: a
// fract(sin(x)*43758) hash amplifies ulp differences between programs and detaches edges from nodes.
vec3 build(vec3 p){
	if (uBuild >= 1.0) return p;
	float e = clamp(uBuild * 1.7 - 0.7 * fract(dot(p.xy, vec2(.173, .311))), 0.0, 1.0);
	e = e * e * (3.0 - 2.0 * e);
	return p + vec3(sin(p.y * 1.3 + p.x * .4), cos(p.x * 1.1 - p.y * .3), sin(p.x * .9 + p.y * 1.7)) * 45.0 * (1.0 - e);
}
// aspect-corrected NDC distance vector from a field centre
vec2 dv(vec2 ndc, vec2 c){ return (ndc - c) * vec2(uAspect, 1.); }
// the force field, in NDC: magnitude ∝ distance × gaussian, so each centre itself never moves
vec2 field(vec2 ndc){
	vec2 a = dv(ndc, uMouse), b = dv(ndc, uBloomC);
	float gb = exp(-dot(b, b) / .09);
	vec2 o = a * exp(-dot(a, a) / .03) * uForce * .55 * max(0., 1. - exp(-dot(b, b) / .004) * uBloom)
	       + b * gb * uBloom * 1.1;
	return vec2(o.x / uAspect, o.y);
}
// move a view-space point by the field, at its own depth (nearby layer only)
vec4 fieldMV(vec4 mv){
	vec4 c = projectionMatrix * mv;
	if (c.w <= 0.) return mv;
	vec2 o = field(c.xy / c.w) * (1. - smoothstep(45., 90., c.w));
	mv.x += o.x * (-mv.z) / projectionMatrix[0][0];
	mv.y += o.y * (-mv.z) / projectionMatrix[1][1];
	return mv;
}
float near(vec4 mv, vec2 c, float r){ vec4 cl = projectionMatrix * mv; vec2 d = dv(cl.xy / cl.w, c); return exp(-dot(d, d) / r); }
`;

// Node spheres: instanced; the field moves each sphere RIGIDLY (offset computed at its centre), and
// the one under the bloom centre grows.
const nodeMat = U => new THREE.ShaderMaterial({
	uniforms: { ...U },
	vertexShader: `${COMMON}
		varying float vL; varying float vA;
		void main(){
			vec3 centre = (instanceMatrix * vec4(0., 0., 0., 1.)).xyz;
			vec3 local = (instanceMatrix * vec4(position, 1.)).xyz - centre;
			vec4 mvc = modelViewMatrix * vec4(build(centre), 1.);
			float grow = 1. + near(mvc, uBloomC, .0025) * uBloom * 1.6 + near(mvc, uMouse, .0015) * uForce * .6;
			vec4 mv = fieldMV(mvc) + vec4(mat3(modelViewMatrix) * local * grow, 0.);
			gl_Position = projectionMatrix * mv;
			#ifdef USE_INSTANCING_COLOR
				vL = instanceColor.r;
			#else
				vL = 1.;
			#endif
			vA = fogK(-mv.z);
		}`,
	fragmentShader: `varying float vL; varying float vA; void main(){ gl_FragColor = vec4(vec3(vL * vA), 1.); }`,
});

const edgeMat = (U, extra = {}) => new THREE.ShaderMaterial({
	uniforms: { ...U, uBurst: { value: 0 }, ...extra },
	vertexShader: `${COMMON}
		attribute float aT; attribute float aPhase; attribute float aSpeed; attribute float aBase;
		uniform float uBurst; varying float vT; varying float vHead; varying float vA; varying float vBase; varying float vLit;
		void main(){
			vT = aT; vBase = aBase * (1. + uBurst * 1.5);
			vHead = fract(uTime * aSpeed * (1. + uBurst) + aPhase) * 1.6 - 0.3;
			vec4 mv = fieldMV(modelViewMatrix * vec4(build(position), 1.0));
			vA = fogK(-mv.z) * smoothstep(.55, 1., uBuild);
			vLit = 1. + uBurst * 2.5 + near(mv, uBloomC, .02) * uBloom * 2.2 + near(mv, uMouse, .012) * uForce * .8;
			gl_Position = projectionMatrix * mv;
		}`,
	fragmentShader: `${LIME}
		varying float vT; varying float vHead; varying float vA; varying float vBase; varying float vLit;
		void main(){
			float d = (vT - vHead) / 0.07, h = exp(-d * d) * 0.85;
			gl_FragColor = vec4(mix(vec3(1.0), LIME, h / max(vBase + h, 1e-3)), (vBase + h) * vLit * vA);   // the wire stays white, its travelling head is lime
		}`,
	transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

const sparkMat = U => new THREE.ShaderMaterial({
	uniforms: { ...U, uBurst: { value: 0 } },
	vertexShader: `${COMMON}
		attribute vec3 aA; attribute vec3 aB; attribute float aPhase; attribute float aSpeed; attribute float aSize;
		uniform float uPR; uniform float uBurst; varying float vA;
		void main(){
			// the same head as the edge shader, so the spark rides exactly on its edge's glow
			float h = fract(uTime * aSpeed * (1. + uBurst) + aPhase) * 1.6 - 0.3, u = clamp(h, 0.0, 1.0);
			// displace BOTH ends, then mix: an edge is a straight line between its displaced ends
			vec4 mv = mix(fieldMV(modelViewMatrix * vec4(build(aA), 1.0)), fieldMV(modelViewMatrix * vec4(build(aB), 1.0)), u);
			gl_Position = projectionMatrix * mv;
			vA = sin(u * 3.14159) * step(0.0, h) * step(h, 1.0) * fogK(-mv.z) * (1. + uWarp) * smoothstep(.55, 1., uBuild);
			gl_PointSize = aSize * (1. + uWarp * 1.5 + uBurst) * uPR * (70.0 / -mv.z);
		}`,
	fragmentShader: `${LIME}
		varying float vA;
		void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.0, r); gl_FragColor = vec4(LIME, a * a * vA); }`,
	transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

const haloMat = U => new THREE.ShaderMaterial({
	uniforms: { ...U },
	vertexShader: `${COMMON}
		attribute float aSize; attribute float aPulse; uniform float uPR; varying float vA;
		void main(){
			vec4 mv = fieldMV(modelViewMatrix * vec4(build(position), 1.0));
			gl_Position = projectionMatrix * mv;
			float beat = aPulse * (0.5 + 0.5 * sin(uTime * 3.2));
			float hot = near(mv, uBloomC, .0025) * uBloom;
			vA = (0.35 + beat * 0.65 + hot) * fogK(-mv.z) * uBuild;
			gl_PointSize = aSize * (1.0 + beat * 0.9 + hot * 2.2) * uPR * (70.0 / -mv.z);
		}`,
	fragmentShader: `${LIME}
		varying float vA;
		void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.05, r); gl_FragColor = vec4(LIME, a * a * 0.5 * vA); }`,
	transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

/* -------------------------------------------------------------- graph build */
/** k nearest neighbours as undirected edges (i<j), deduped. O(n²) — fine up to a few thousand. */
function knn(pts, k) {
	const n = pts.length, seen = new Set(), out = [];
	for (let i = 0; i < n; i++) {
		const d = [];
		for (let j = 0; j < n; j++) if (j !== i) d.push([pts[i].distanceToSquared(pts[j]), j]);
		d.sort((a, b) => a[0] - b[0]);
		for (let m = 0; m < k && m < d.length; m++) {
			const j = d[m][1], key = i < j ? i * 100000 + j : j * 100000 + i;
			if (!seen.has(key)) { seen.add(key); out.push([Math.min(i, j), Math.max(i, j)]); }
		}
	}
	return out;
}

/** Hop distance from `src` over `edges` — the newest story's signal spreads outward in waves. */
function hops(n, edges, src) {
	const adj = Array.from({ length: n }, () => []);
	for (const [a, b] of edges) { adj[a].push(b); adj[b].push(a); }
	const h = new Array(n).fill(-1); h[src] = 0;
	const q = [src];
	while (q.length) { const v = q.shift(); for (const w of adj[v]) if (h[w] < 0) { h[w] = h[v] + 1; q.push(w); } }
	return h;
}

/**
 * One layer: nodes [{p, size, lum, halo, pulse, pick}], edges [[i,j]], per-edge timing.
 * Each layer gets its OWN edge/spark materials (same source → same program) so a burst is per layer.
 */
function buildLayer(L, U, rand, sparkBudget) {
	const g = new THREE.Group();
	g.position.z = L.z;
	const n = L.nodes.length;
	const mats = { node: nodeMat(U), edge: edgeMat(U), spark: sparkMat(U), halo: haloMat(U) };
	mats.spark.uniforms.uBurst = mats.edge.uniforms.uBurst;              // one burst value per layer

	const mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), mats.node, n);
	const m4 = new THREE.Matrix4(), c = new THREE.Color();
	L.nodes.forEach((nd, i) => {
		m4.makeScale(nd.size, nd.size, nd.size).setPosition(nd.p);
		mesh.setMatrixAt(i, m4);
		mesh.setColorAt(i, c.setScalar(nd.lum));
	});
	mesh.instanceMatrix.needsUpdate = true;
	mesh.frustumCulled = false;                                            // the intro scatters far outside the bounds
	g.add(mesh);

	const E = L.edges, h = L.src != null ? hops(n, E, L.src) : null;
	const pos = new Float32Array(E.length * 6), aT = new Float32Array(E.length * 2);
	const aPhase = new Float32Array(E.length * 2), aSpeed = new Float32Array(E.length * 2), aBase = new Float32Array(E.length * 2);
	const timing = E.map(([a, b]) => {
		const wave = h && h[a] >= 0;
		const hop = wave ? Math.min(h[a], h[b]) : 0;
		return { speed: wave ? .22 : .08 + rand() * .22, phase: wave ? -hop * .09 : rand(), base: L.base * (.6 + rand() * .8), flip: wave && h[b] < h[a] };
	});
	E.forEach(([a, b], e) => {
		const t = timing[e];
		const [p, q] = t.flip ? [L.nodes[b].p, L.nodes[a].p] : [L.nodes[a].p, L.nodes[b].p];
		pos.set([p.x, p.y, p.z, q.x, q.y, q.z], e * 6);
		aT.set([0, 1], e * 2); aPhase.set([t.phase, t.phase], e * 2); aSpeed.set([t.speed, t.speed], e * 2); aBase.set([t.base, t.base], e * 2);
	});
	const eg = new THREE.BufferGeometry();
	eg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
	eg.setAttribute('aT', new THREE.BufferAttribute(aT, 1));
	eg.setAttribute('aPhase', new THREE.BufferAttribute(aPhase, 1));
	eg.setAttribute('aSpeed', new THREE.BufferAttribute(aSpeed, 1));
	eg.setAttribute('aBase', new THREE.BufferAttribute(aBase, 1));
	const lines = new THREE.LineSegments(eg, mats.edge);
	lines.frustumCulled = false;
	g.add(lines);

	const S = Math.min(sparkBudget, E.length);
	const pick = [...E.keys()].sort(() => rand() - .5).slice(0, S);
	const sA = new Float32Array(S * 3), sB = new Float32Array(S * 3), sPh = new Float32Array(S), sSp = new Float32Array(S), sSz = new Float32Array(S);
	pick.forEach((e, k) => {
		const t = timing[e];
		const [a, b] = t.flip ? [E[e][1], E[e][0]] : E[e];
		const p = L.nodes[a].p, q = L.nodes[b].p;
		sA.set([p.x, p.y, p.z], k * 3); sB.set([q.x, q.y, q.z], k * 3);
		sSp[k] = t.speed; sPh[k] = t.phase; sSz[k] = L.spark * (.7 + rand() * .6);
	});
	const sg = new THREE.BufferGeometry();
	sg.setAttribute('position', new THREE.BufferAttribute(sA.slice(), 3));
	sg.setAttribute('aA', new THREE.BufferAttribute(sA, 3));
	sg.setAttribute('aB', new THREE.BufferAttribute(sB, 3));
	sg.setAttribute('aPhase', new THREE.BufferAttribute(sPh, 1));
	sg.setAttribute('aSpeed', new THREE.BufferAttribute(sSp, 1));
	sg.setAttribute('aSize', new THREE.BufferAttribute(sSz, 1));
	const sparks = new THREE.Points(sg, mats.spark);
	sparks.frustumCulled = false;
	g.add(sparks);

	const H = L.nodes.filter(nd => nd.halo);
	if (H.length) {
		const hp = new Float32Array(H.length * 3), hs = new Float32Array(H.length), hu = new Float32Array(H.length);
		H.forEach((nd, i) => { hp.set([nd.p.x, nd.p.y, nd.p.z], i * 3); hs[i] = nd.halo; hu[i] = nd.pulse ? 1 : 0; });
		const hg = new THREE.BufferGeometry();
		hg.setAttribute('position', new THREE.BufferAttribute(hp, 3));
		hg.setAttribute('aSize', new THREE.BufferAttribute(hs, 1));
		hg.setAttribute('aPulse', new THREE.BufferAttribute(hu, 1));
		const halos = new THREE.Points(hg, mats.halo);
		halos.frustumCulled = false;
		g.add(halos);
	}
	return { group: g, mesh, mats, picks: L.nodes.map((nd, i) => nd.pick ? { ...nd.pick, i, local: nd.p } : null).filter(Boolean) };
}

/* the four layers, as data */
function newsLayer(cfg, rand, tier) {
	const nodes = [], V = (x, y, z) => new THREE.Vector3(x, y, z);
	const news = cfg.news || [];
	news.forEach((s, i) => {
		const a = (i / Math.max(1, news.length)) * Math.PI * 2 + rand() * .4;
		const r = .55 + rand() * .45;
		nodes.push({ p: V(Math.cos(a) * 14 * r, Math.sin(a) * 8 * r, 3 + rand() * 6 - i * .18), size: i === 0 ? .3 : .19, lum: 1, halo: i === 0 ? 5 : 2.6, pulse: i === 0,
			pick: { kind: 'news', t: s.t, k: `${s.d} · ${s.c}`, u: s.u, img: s.i || '', live: i === 0 } });
	});
	const filler = tier === 1 ? 240 : 130;
	for (let i = 0; i < filler; i++) {
		const u = rand() * 2 - 1, th = rand() * Math.PI * 2, rr = Math.cbrt(rand());
		const s = Math.sqrt(1 - u * u);
		nodes.push({ p: V(s * Math.cos(th) * 16 * rr, s * Math.sin(th) * 9.5 * rr, u * 9 * rr), size: .07 + rand() * .11, lum: .3 + rand() * .35 });
	}
	return { z: Z.news, nodes, edges: knn(nodes.map(n => n.p), 3), src: news.length ? 0 : null, base: .1, spark: 2.2 };
}

function toolsLayer(cfg, rand, tier) {
	const nodes = [], V = (x, y, z) => new THREE.Vector3(x, y, z), edges = [];
	const cats = cfg.cats && cfg.cats.length ? cfg.cats : [{ n: 'Tools', u: null, c: 0 }];
	const hub = cats.map((cat, i) => {
		const a = (i / cats.length) * Math.PI * 2 + Math.PI / 2;
		nodes.push({ p: V(Math.cos(a) * 13, Math.sin(a) * 8.5, (rand() - .5) * 4), size: .46, lum: .9, halo: 3.4,
			pick: { kind: 'hub', t: cat.n, k: 'Category', u: cat.u, count: cat.c, cat: i } });
		return nodes.length - 1;
	});
	hub.forEach((h, i) => edges.push([h, hub[(i + 1) % hub.length]]));
	(cfg.tools || []).forEach(t => {
		const h = nodes[hub[t.c] ?? hub[0]].p, r = 2.4 + rand() * 2.6, a = rand() * Math.PI * 2, b = (rand() - .5) * Math.PI;
		nodes.push({ p: V(h.x + Math.cos(a) * Math.cos(b) * r, h.y + Math.sin(b) * r, h.z + Math.sin(a) * Math.cos(b) * r),
			size: t.f ? .4 : .18, lum: t.f ? 1 : .6, halo: t.f ? 4.6 : 0, pulse: !!t.f,
			pick: { kind: 'tool', t: t.t, k: (cats[t.c] || cats[0]).n, u: t.u, l: t.l || '', f: !!t.f, cat: t.c } });
		edges.push([hub[t.c] ?? hub[0], nodes.length - 1]);
	});
	const filler = tier === 1 ? 90 : 50, start = nodes.length;
	for (let i = 0; i < filler; i++) {
		const a = rand() * Math.PI * 2, r = 6 + rand() * 12;
		nodes.push({ p: V(Math.cos(a) * r, Math.sin(a) * r * .65, (rand() - .5) * 10), size: .08 + rand() * .1, lum: .25 + rand() * .3 });
	}
	for (const [a, b] of knn(nodes.map(n => n.p), 2)) if (a >= start || b >= start) edges.push([a, b]);
	return { z: Z.tools, nodes, edges, base: .09, spark: 2 };
}

function audienceLayer(cfg, rand) {
	// ONE NODE PER THOUSAND FOLLOWERS. The copy says so, so the count must be exact.
	const n = Math.max(1, Math.round((cfg.followers || 1300000) / 1000)), nodes = [], R = 14;
	const golden = Math.PI * (3 - Math.sqrt(5));
	for (let i = 0; i < n; i++) {
		const y = 1 - (i / (n - 1)) * 2, r = Math.sqrt(1 - y * y), th = golden * i;
		nodes.push({ p: new THREE.Vector3(Math.cos(th) * r * R, y * R, Math.sin(th) * r * R), size: .13 + rand() * .07, lum: .45 + rand() * .45 });
	}
	return { z: Z.audience, nodes, edges: knn(nodes.map(n => n.p), 2), base: .07, spark: 1.8, R };
}

function learnLayer(cfg, rand) {
	const cols = [6, 10, 14, 10, 5], nodes = [], edges = [], idx = [];
	cols.forEach((count, ci) => {
		const x = -15 + (30 / (cols.length - 1)) * ci, ids = [];
		for (let k = 0; k < count; k++) {
			const y = (k - (count - 1) / 2) * (17 / 13);
			nodes.push({ p: new THREE.Vector3(x, y, (rand() - .5) * 3), size: .26, lum: .85, halo: ci === cols.length - 1 ? 4 : 0 });
			ids.push(nodes.length - 1);
		}
		idx.push(ids);
	});
	for (let c = 0; c < idx.length - 1; c++) for (const a of idx[c]) for (const b of idx[c + 1]) edges.push([a, b]);
	return { z: Z.learn, nodes, edges, src: idx[0][0], base: .045, spark: 1.8, outputs: idx[idx.length - 1] };
}

/* ------------------------------------------------------------------ DOM bits */
const el = (tag, cls, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (parent) parent.appendChild(e); return e; };

/* ------------------------------------------------------------------ boot */
export async function boot(cfg) {
	const canvas = document.getElementById('net');
	if (!canvas) return;
	const doc = document.documentElement, tier = cfg.tier || 1;
	const pr = Math.min(devicePixelRatio || 1, tier === 1 ? 1.25 : 1);
	const renderer = new THREE.WebGLRenderer({ canvas, antialias: tier === 1, powerPreference: 'high-performance' });
	renderer.setPixelRatio(pr);
	renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // flat white-on-black: skip the sRGB pass
	renderer.setClearColor(0x000000, 1);

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(50, 1, .1, 400);

	// SHARED uniform objects — spread into every material, so one write updates all of them
	const U = {
		uTime: { value: 0 }, uNear: { value: FOG.near }, uFar: { value: FOG.far }, uPR: { value: pr },
		uMouse: { value: new THREE.Vector2(-9, -9) }, uForce: { value: 0 },
		uBloomC: { value: new THREE.Vector2(-9, -9) }, uBloom: { value: 0 },
		uAspect: { value: 1 }, uBuild: { value: 0 }, uWarp: { value: 0 },
	};
	const rand = rng(1300);
	const spark = tier === 1 ? 260 : 140;
	const layers = [newsLayer(cfg, rand, tier), toolsLayer(cfg, rand, tier), audienceLayer(cfg, rand), learnLayer(cfg, rand)]
		.map(L => ({ L, ...buildLayer(L, U, rand, spark) }));
	layers.forEach(({ group }) => scene.add(group));

	// the spine: long edges from each layer's far side to the next layer's near side
	{
		const pairs = [];
		for (let i = 0; i < layers.length - 1; i++) {
			const A = layers[i], B = layers[i + 1];
			for (let k = 0; k < 7; k++) {
				const a = A.L.nodes[Math.floor(rand() * A.L.nodes.length)].p.clone(); a.z += A.L.z;
				const b = B.L.nodes[Math.floor(rand() * B.L.nodes.length)].p.clone(); b.z += B.L.z;
				pairs.push([a, b]);
			}
		}
		const nodes = pairs.flat().map(p => ({ p, size: .01, lum: 0 })), edges = pairs.map((_, i) => [i * 2, i * 2 + 1]);
		const spine = buildLayer({ z: 0, nodes, edges, base: .05, spark: 2.4 }, U, rand, edges.length);
		spine.mesh.visible = false;
		scene.add(spine.group);
	}

	/* the cursor neuron: WIRES segments from the cursor to the nearest dots. World-space positions,
	   displaced by the SAME shader field — the cursor end sits where the field is zero and the node end
	   shares the node's exact position, so both ends land true without any JS displacement. */
	const wireMat = edgeMat(U);
	const wirePos = new Float32Array(WIRES * 6);
	const wg = new THREE.BufferGeometry();
	wg.setAttribute('position', new THREE.BufferAttribute(wirePos, 3).setUsage(THREE.DynamicDrawUsage));
	const wAttr = (vals) => new THREE.BufferAttribute(new Float32Array(WIRES * 2).map((_, i) => vals(i >> 1, i & 1)), 1);
	wg.setAttribute('aT', wAttr((w, end) => end));
	wg.setAttribute('aPhase', wAttr(w => w / WIRES));
	wg.setAttribute('aSpeed', wAttr(() => .9));
	wg.setAttribute('aBase', wAttr(() => .32));
	wg.setDrawRange(0, 0);
	const wires = new THREE.LineSegments(wg, wireMat);
	wires.frustumCulled = false;
	scene.add(wires);
	const cursorGlow = new THREE.Points(new THREE.BufferGeometry()
		.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
		.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array([7]), 1))
		.setAttribute('aPulse', new THREE.BufferAttribute(new Float32Array([1]), 1)), haloMat(U));
	cursorGlow.frustumCulled = false; cursorGlow.visible = false;
	scene.add(cursorGlow);

	/* camera: scroll through [data-layer] sections, or park on this page's layer */
	const sections = [...document.querySelectorAll('section[data-layer]')];
	const stopOf = key => {
		const s = STOPS[key] || STOPS.park;
		const dim = sections.length ? s.dim : (key === 'park' || key === 'hero' ? STOPS.park.dim : INNER_DIM);
		return { key, pos: new THREE.Vector3(...s.pos), look: new THREE.Vector3(...s.look), dim, shift: s.shift ?? .2, far: s.far ?? FOG.far };
	};
	const pageStop = stopOf(cfg.layer === 'hero' && !sections.length ? 'park' : cfg.layer);
	const stops = sections.map(s => stopOf(s.dataset.layer));
	const ease = x => x * x * (3 - 2 * x);
	const phoneMQ = matchMedia('(max-width: 900px)');     // = the CSS breakpoint: acts become window + solid copy, the card a bottom sheet
	const target = { pos: new THREE.Vector3(), look: new THREE.Vector3(), dim: 1, shift: .2, far: FOG.far, key: 'park' };
	// the footer pushes the net up (screen px, set in frame()) once it reaches the lowest dot in view: nothing draws under footer text
	const foot = sections.length ? document.querySelector('.foot') : null;
	let footTop = Infinity, lift = 0;
	const footLift = () => { footTop = foot ? foot.getBoundingClientRect().top : Infinity; };
	function aim() {
		footLift();
		if (!sections.length) { Object.assign(target, { dim: pageStop.dim, shift: .28, far: pageStop.far, key: pageStop.key }); target.pos.copy(pageStop.pos); target.look.copy(pageStop.look); return; }
		let k = 0, e = 0;
		if (phoneMQ.matches) {
			// phones: every act opens with a window onto the network (CSS --win). A layer flies in while ITS window rises
			// from the bottom edge to mid-screen, so a window never shows the empty sky between two layers.
			for (let i = 1; i < sections.length; i++) {
				const top = sections[i].getBoundingClientRect().top;
				if (top <= innerHeight * .55) { k = i; continue; }
				e = ease(Math.min(1, Math.max(0, (innerHeight - top) / (innerHeight * .45)))); break;
			}
		} else {
			const anchor = innerHeight * .45;
			let f = 0;
			for (let i = 0; i < sections.length; i++) {
				const r = sections[i].getBoundingClientRect();
				if (r.top <= anchor) { k = i; f = Math.min(1, (anchor - r.top) / r.height); }
			}
			e = k < stops.length - 1 ? ease(Math.max(0, (f - .5) / .5)) : 0;       // hold, then fly
		}
		target.k = k; target.e = e;
		const A = stops[k], B = stops[Math.min(k + 1, stops.length - 1)];
		target.pos.lerpVectors(A.pos, B.pos, e);
		target.look.lerpVectors(A.look, B.look, e);
		target.dim = A.dim + (B.dim - A.dim) * e;
		target.shift = A.shift + (B.shift - A.shift) * e;
		target.far = A.far + (B.far - A.far) * e;
		target.key = e < .5 ? A.key : B.key;
	}
	aim();
	const cam = { pos: target.pos.clone(), look: target.look.clone(), dim: target.dim, shift: target.shift, far: target.far };
	addEventListener('scroll', aim, { passive: true });

	/* ---------------------------------------------------------- the DOM overlay */
	const world = canvas.parentElement;
	const tagBox = el('div', 'tags', world);
	const bloomEl = document.getElementById('bloom');
	const lead = bloomEl && bloomEl.querySelector('.bloom__lead');
	const dot = bloomEl && bloomEl.querySelector('.bloom__dot');
	const card = bloomEl && bloomEl.querySelector('.bloom__card');

	// always-on tags: [layer index or 'hero', world-point fn, text, class]
	const tags = [];
	const tag = (layer, at, text, cls = '') => { const e = el('span', 'tag ' + cls, tagBox); e.textContent = text; tags.push({ layer, at, e, o: 0 }); return tags[tags.length - 1]; };
	const worldOf = (li, local) => () => local.clone().applyMatrix4(layers[li].group.matrixWorld);
	// hero: one label per layer. The layers recede toward one vanishing point, so the anchors fan out
	// (and placeTags() hides any hero label that would still overlap one already placed)
	[[0, 12], [-17, 11], [17, 16], [-4, -14]].forEach(([x, y], i) =>
		tag('hero', () => new THREE.Vector3(x, y, Object.values(Z)[i]), ['01 · News', '02 · Tools', '03 · Audience', '04 · Learn'][i], 'tag--layer'));
	const live = layers[0].picks.find(p => p.live);
	if (live) tag(0, worldOf(0, live.local), 'LIVE · ' + (live.t.length > 42 ? live.t.slice(0, 40).trim() + '…' : live.t), 'tag--live');
	layers[1].picks.filter(p => p.kind === 'hub').forEach(p => tag(1, worldOf(1, p.local), p.t, 'tag--cat'));
	tag(2, () => new THREE.Vector3(layers[2].L.R * .78, layers[2].L.R * .86, Z.audience), '1 dot = 1,000 followers', 'tag--note');
	const outs = layers[3].L.outputs, top = outs.length - 2;               // skip the very top and bottom nodes
	(cfg.courses || []).slice(0, top).forEach((name, i) =>
		tag(3, worldOf(3, layers[3].L.nodes[outs[Math.max(0, top - i)]].p), name, 'tag--course'));
	// dynamic tags: the tool names of an opened category (a pool), and the people-under-cursor count
	const flower = Array.from({ length: 10 }, () => { const e = el('span', 'tag tag--tool', tagBox); e.style.opacity = 0; return e; });
	const countTag = el('span', 'tag tag--count', tagBox); countTag.style.opacity = 0;

	/* ---------------------------------------------------------- pointer state */
	const P = { x: -1, y: -1, nx: 0, ny: 0, inside: false, onCard: false, last: 0, drag: false, dragX: 0 };
	let spin = 0, spinV = 0;
	const busy = e => e && e.closest && e.closest('a, button, input, select, textarea, label, .act__copy, .page__head, .tools, .stories, .offer, .bloom, .bar, .ribbon, .foot');
	addEventListener('pointermove', e => {
		P.x = e.clientX; P.y = e.clientY; P.nx = P.x / innerWidth * 2 - 1; P.ny = P.y / innerHeight * 2 - 1;
		P.onCard = !!(e.target && e.target.closest && e.target.closest('.bloom'));
		P.inside = !P.onCard && !busy(e.target); P.last = performance.now();
		if (P.drag) { spinV = (e.clientX - P.dragX) * .006; spin += spinV; P.dragX = e.clientX; }
	}, { passive: true });
	addEventListener('pointerdown', e => { if (!busy(e.target) && active() === 2) { P.drag = true; P.dragX = e.clientX; } });
	addEventListener('pointerup', () => { P.drag = false; });
	document.addEventListener('pointerleave', () => { P.inside = false; });
	addEventListener('blur', () => { P.inside = false; });

	/* ---------------------------------------------------------- projection helpers */
	const tmp = new THREE.Vector3();
	const toScreen = (v, out = {}) => {
		tmp.copy(v).project(camera);
		out.x = (tmp.x * .5 + .5) * innerWidth; out.y = (-tmp.y * .5 + .5) * innerHeight;
		out.ndcX = tmp.x; out.ndcY = tmp.y; out.front = tmp.z < 1 && tmp.z > -1;
		return out;
	};
	const active = () => LAYER_OF[target.key] ?? LAYER_OF[pageStop.key] ?? 0;
	const allPicks = layers.flatMap(({ group, picks }, li) => picks.map(p => ({ ...p, group, li })));
	const pickWorld = p => p.local.clone().applyMatrix4(p.group.matrixWorld);
	const screenOfPick = p => { const w = pickWorld(p); const s = toScreen(w); s.dist = w.distanceTo(camera.position); return s; };

	/* ---------------------------------------------------------- the bloom */
	let snapped = null, demo = null, demoEnd = 0, bloomTo = 0, autoAt = 0, openY = 0, openAt = 0;
	const demoTag = el('span', 'tag tag--live tag--demo', tagBox); demoTag.style.opacity = 0;
	function nearestPick(x, y, radius) {
		let best = null, bd = radius * radius;
		for (const p of allPicks) {
			if (p.li !== active()) continue;
			const s = screenOfPick(p);
			if (!s.front || s.dist > 80) continue;
			const d = (s.x - x) ** 2 + (s.y - y) ** 2;
			if (d < bd) { bd = d; best = p; }
		}
		return best;
	}
	function fillCard(p) {
		if (!card) return;
		card.replaceChildren();
		card.className = 'bloom__card bloom__card--' + p.kind;
		if (p.u) card.setAttribute('href', p.u); else card.removeAttribute('href');
		if (p.kind === 'news' && p.img) { const i = el('img', 'bloom__img', card); i.alt = ''; i.src = p.img; }
		if (p.kind === 'tool') { const m = el('span', 'bloom__mono', card); m.textContent = p.t.split(/[\s.\-]+/).slice(0, 2).map(w => w[0]).join('').toUpperCase().slice(0, 2).padEnd(2, p.t[1] || ''); }
		const k = el('span', 'bloom__k', card);
		k.textContent = p.kind === 'hub' ? 'Category' : p.kind === 'tool' ? (p.f ? 'Featured · ' : '') + p.k : (p.live ? 'Live · ' : '') + p.k;
		el('span', 'bloom__t', card).textContent = p.t;
		if (p.kind === 'tool' && p.l) el('span', 'bloom__l', card).textContent = p.l;
		if (p.kind === 'hub') el('span', 'bloom__l', card).textContent = `${p.count} tools · the ring opens to show them`;
		if (p.u) el('span', 'bloom__go', card).textContent = p.kind === 'news' ? 'Read the story →' : p.kind === 'hub' ? 'Browse the category →' : 'See the tool →';
	}
	function setSnap(p) {
		if (p === snapped) return;
		snapped = p;
		bloomTo = p ? 1 : 0;
		if (!bloomEl) return;
		if (!p) { bloomEl.classList.remove('is-open'); return; }
		openY = scrollY; openAt = performance.now();
		bloomEl.classList.remove('is-open');
		fillCard(p);
		void bloomEl.offsetWidth;                          // restart the unfold
		bloomEl.hidden = false;
		bloomEl.classList.add('is-open');
	}
	function placeCard() {
		if (demo) { const s = screenOfPick(demo); U.uBloomC.value.set(s.ndcX, s.ndcY); placeTag(demoTag, s.x, s.y); demoTag.style.opacity = behindCopy(s.x, s.y) ? 0 : 1; }
		if (!snapped || !bloomEl || bloomEl.hidden) return;
		const s = screenOfPick(snapped);
		U.uBloomC.value.set(s.ndcX, s.ndcY);
		if (phoneMQ.matches) { dot.style.transform = `translate(${Math.round(s.x)}px, ${Math.round(s.y)}px)`; card.style.transform = ''; return; }
		const w = card.offsetWidth || 300, h = card.offsetHeight || 160;
		const right = s.x + 70 + w < innerWidth - 16;
		const cx = Math.min(Math.max(right ? s.x + 70 : s.x - 70 - w, copyEdge + 16), innerWidth - w - 16);
		const cy = Math.min(Math.max(s.y - h * .45, 110), innerHeight - h - 16);
		card.style.transform = `translate(${Math.round(cx)}px, ${Math.round(cy)}px)`;
		card.style.transformOrigin = right ? '0 50%' : '100% 50%';
		dot.style.transform = `translate(${Math.round(s.x)}px, ${Math.round(s.y)}px)`;
		const ax = right ? cx : cx + w, ay = cy + Math.min(h * .45, 60), dx = ax - s.x, dy = ay - s.y;
		lead.style.width = Math.hypot(dx, dy) + 'px';
		lead.style.transform = `translate(${s.x}px, ${s.y}px) rotate(${Math.atan2(dy, dx)}rad)`;
		doc.style.cursor = P.inside && snapped.u && tier === 1 ? 'pointer' : '';
	}
	// Touch (tier 2): a tap on the graph opens the nearest dot or switches to another; a tap anywhere else (text,
	// header, cards) or on the close button closes the card. Taps come from pointer events: iOS sends no click for a
	// tap on a plain page area. The card itself is the link.
	let tapAt = null, tappedAt = 0;
	function tap(e) {
		const t = e.target && e.target.closest ? e.target : document.body;
		if (t.closest('.bloom__x, .bloom__card')) return;   // the X closes on its own click; the card is the link
		if (busy(t)) { if (snapped) setSnap(null); return; }
		demo = null; demoTag.style.opacity = 0;
		setSnap(nearestPick(e.clientX, e.clientY, 90)); autoAt = performance.now() + 9000;
	}
	// ghost-click guards: the X closes on its OWN click (so no click falls through to what lies under it), and the tap
	// that opened the sheet cannot also follow its link
	const xBtn = bloomEl && bloomEl.querySelector('.bloom__x');
	if (xBtn) xBtn.addEventListener('click', e => { e.preventDefault(); setSnap(null); });
	if (card) card.addEventListener('click', e => { if (phoneMQ.matches && performance.now() - openAt < 450) e.preventDefault(); });
	addEventListener('pointerdown', e => { tapAt = e.pointerType === 'mouse' ? null : { x: e.clientX, y: e.clientY, t: performance.now() }; }, { passive: true });
	addEventListener('pointercancel', () => { tapAt = null; });
	addEventListener('pointerup', e => {
		const a = tapAt; tapAt = null;
		if (!a || tier === 1 || Math.hypot(e.clientX - a.x, e.clientY - a.y) > 12 || performance.now() - a.t > 600) return;
		tappedAt = performance.now(); tap(e);
	});
	addEventListener('click', e => {
		if (tier === 1) {                                  // only a click ON the bloomed dot navigates; the card is a link too
			if (busy(e.target)) return;
			if (snapped && snapped.u) { const s = screenOfPick(snapped); if (Math.hypot(s.x - e.clientX, s.y - e.clientY) < 60) location.href = snapped.u; }
			return;
		}
		if (performance.now() - tappedAt > 700) tap(e);   // a mouse in a narrow window, or a click no tap produced
	});
	addEventListener('scroll', () => { if (tier !== 1 && snapped && Math.abs(scrollY - openY) > 60) setSnap(null); }, { passive: true });

	/* ---------------------------------------------------------- per-frame helpers */
	const wireTo = [], near = [];
	function updateWires(li) {
		// the nearest WIRES dots of the active layer to the cursor, by screen distance
		const L = layers[li], nodes = L.L.nodes, m = L.group.matrixWorld;
		near.length = 0;
		let count = 0, sx = {}, depth = 0;
		const front = li === 2 ? L.group.getWorldPosition(new THREE.Vector3()).distanceTo(camera.position) : Infinity;
		for (let i = 0; i < nodes.length; i++) {
			const w = nodes[i].p.clone().applyMatrix4(m);
			toScreen(w, sx);
			if (!sx.front) continue;
			const d2 = (sx.x - P.x) ** 2 + (sx.y - P.y) ** 2;
			const dist = w.distanceTo(camera.position);
			if (li === 2 && d2 < 90 * 90 && dist < front) count++;               // front hemisphere only
			if (d2 > 260 * 260) continue;
			if (near.length < WIRES) near.push([d2, w, dist]);
			else { let wi = 0; for (let j = 1; j < WIRES; j++) if (near[j][0] > near[wi][0]) wi = j; if (d2 < near[wi][0]) near[wi] = [d2, w, dist]; }
		}
		near.sort((a, b) => a[0] - b[0]);
		depth = near.length ? near[0][2] : 30;
		// the cursor's world point: unproject the mouse at the nearest dot's depth
		const c = new THREE.Vector3(P.nx, -P.ny, .5).unproject(camera).sub(camera.position).normalize().multiplyScalar(depth * .92).add(camera.position);
		near.forEach(([, w], i) => wirePos.set([c.x, c.y, c.z, w.x, w.y, w.z], i * 6));
		wg.attributes.position.needsUpdate = true;
		wg.setDrawRange(0, near.length * 2);
		cursorGlow.geometry.attributes.position.array.set([c.x, c.y, c.z]);
		cursorGlow.geometry.attributes.position.needsUpdate = true;
		return count;
	}
	function flowerFor(hub) {
		const tools = hub ? allPicks.filter(p => p.kind === 'tool' && p.cat === hub.cat) : [];
		const r = hub && card ? card.getBoundingClientRect() : null, placed = [];
		flower.forEach((e, i) => {
			const p = tools[i];
			if (!p) { e.style.opacity = 0; return; }
			const s = screenOfPick(p);
			// the bloom pushes the tools outward; place the label on the pushed position
			const dx = s.x - (U.uBloomC.value.x * .5 + .5) * innerWidth, dy = s.y - (-U.uBloomC.value.y * .5 + .5) * innerHeight;
			const push = 1 + .55 * U.uBloom.value;
			if (e.textContent !== p.t) e.textContent = p.t;
			const x = s.x + dx * (push - 1), y = s.y + dy * (push - 1);
			const box = placeTag(e, x, y), bx = e.classList.contains('is-flip') ? x - 12 - box.w : x + 12;
			// a name that would sit under the card, or on another name, is left out rather than half-hidden
			const hit = (ax, aw, ay, q) => ax < q.x + q.w + 6 && q.x < ax + aw + 6 && Math.abs(ay - q.y) < 22;
			const blocked = (r && hit(bx, box.w, y, { x: r.left, w: r.width, y: Math.min(Math.max(y, r.top), r.bottom) }))
				|| placed.some(q => hit(bx, box.w, y, q));
			if (!blocked && s.front) placed.push({ x: bx, w: box.w, y });
			e.style.opacity = s.front && !blocked ? U.uBloom.value : 0;
		});
	}
	/** Position a tag at a screen point without leaving the viewport: side tags flip to the left of
	    their anchor near the right edge, centred tags clamp. Widths are measured once per text. */
	const widths = new WeakMap();
	function placeTag(e, x, y) {
		let w = widths.get(e);
		if (!w || w.text !== e.textContent) { w = { text: e.textContent, px: e.offsetWidth }; widths.set(e, w); }
		const centred = e.classList.contains('tag--layer') || e.classList.contains('tag--cat');
		if (centred) x = Math.min(Math.max(x, w.px / 2 + 10), innerWidth - w.px / 2 - 10);
		else {
			let flip = x + 12 + w.px > innerWidth - 10;
			// room on neither side (narrow phones): pin the label inside the viewport instead
			if (flip && x - 12 - w.px < 10) { flip = false; x = Math.max(-2, innerWidth - 22 - w.px); }
			e.classList.toggle('is-flip', flip);
		}
		e.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
		return { x: centred ? x - w.px / 2 : x, w: w.px };
	}
	let copyRects = [];
	const behindCopy = (x, y) => copyRects.some(r => x > r.left - 16 && x < r.right + 16 && y > r.top - 16 && y < r.bottom + 16);
	function updateTags(heroW) {
		copyRects = sections.length ? [...document.querySelectorAll('.act__copy')].map(e => e.getBoundingClientRect()).filter(r => r.bottom > 0 && r.top < innerHeight) : [];
		const li = active(), placed = [];
		for (const t of tags) {
			// tags narrate the home-page tour only; inner pages keep the network as quiet scenery
			const want = sections.length ? (t.layer === 'hero' ? heroW : (t.layer === li ? 1 - heroW : 0)) * U.uBuild.value : 0;
			const w = t.at();
			const s = toScreen(w);
			const d = w.distanceTo(camera.position);
			let o = want * (s.front && !behindCopy(s.x, s.y) ? 1 : 0) * (1 - Math.min(1, Math.max(0, (d - 70) / 160)));
			if (o > .01 || t.o > .01) {
				const r = placeTag(t.e, s.x, s.y);
				if (t.layer === 'hero') {
					if (placed.some(q => Math.abs(q.y - s.y) < 30 && r.x < q.x + q.w + 8 && q.x < r.x + r.w + 8)) o = 0;
					else if (o > .01) placed.push({ x: r.x, y: s.y, w: r.w });
				}
			}
			t.o += (o - t.o) * .15;
			t.e.style.opacity = t.o.toFixed(3);
		}
	}

	/* size — the graph sits RIGHT of the copy on desktop (by the stop's shift), ABOVE it on a phone */
	let portrait = false, lastShift = -1, lastLift = -1, fovKick = 0, winY = 0, lastWinY = 0, winPad = [];
	const PHONE_ZOOM = .8;                                     // ponytail: one zoom for every phone; tune here if a layer crops
	// phones: an act's window = its padding-top (CSS --win / --win-hero; the hero's padding also holds the header).
	let headH = 0;
	const winCentre = i => { const top = sections[i].getBoundingClientRect().top; return top + winPad[i] - (winPad[i] - (i ? 0 : headH)) / 2; };
	const phoneWin = () => { const a = winCentre(target.k); return a + (winCentre(Math.min(target.k + 1, sections.length - 1)) - a) * target.e; };
	function frameView() {
		const w = innerWidth, h = innerHeight, phone = phoneMQ.matches && sections.length;
		camera.fov = (portrait ? 62 : 50) + fovKick;
		camera.zoom = phone ? PHONE_ZOOM : 1;
		camera.setViewOffset(w, h, portrait || phone ? 0 : -w * cam.shift, (phone ? h / 2 - winY : portrait ? h * .2 : 0) + lift, w, h);
		camera.updateProjectionMatrix();
		lastShift = cam.shift; lastLift = lift; lastWinY = winY;
	}
	let copyEdge = 0;                                          // right edge of the home copy panels: the card stays clear of it
	function resize() {
		const w = innerWidth, h = innerHeight;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		U.uAspect.value = w / h;
		portrait = h > w * 1.1;
		copyEdge = portrait ? 0 : Math.max(0, ...[...document.querySelectorAll('.act__copy')].map(e => e.getBoundingClientRect().right));
		footLift();
		winPad = sections.map(s => parseFloat(getComputedStyle(s).paddingTop) || 0);
		headH = (document.querySelector('.bar') || { getBoundingClientRect: () => ({ bottom: 0 }) }).getBoundingClientRect().bottom;
		if (phoneMQ.matches && sections.length) { aim(); winY = phoneWin(); }
		frameView();
	}
	resize();
	addEventListener('resize', resize);

	/* ---------------------------------------------------------- loop */
	const clock = new THREE.Clock();
	const audience = layers[2].group;
	const prevPos = new THREE.Vector3().copy(cam.pos);
	const lowV = new THREE.Vector3(), lowS = {}, FOOT_GAP = 40;
	let raf = 0, ready = false, t0 = -1, count = 0, burstAt = -9;
	function frame() {
		raf = requestAnimationFrame(frame);
		const dt = Math.min(clock.getDelta(), .05), t = clock.elapsedTime, now = performance.now();
		const k = 1 - Math.exp(-dt * 3.2);
		cam.pos.lerp(target.pos, k); cam.look.lerp(target.look, k); cam.dim += (target.dim - cam.dim) * k;
		cam.shift += (target.shift - cam.shift) * k;
		cam.far += (target.far - cam.far) * k;

		// warp: the faster the camera flies between layers, the wider the lens and the brighter the sparks.
		// ponytail: an FOV kick + spark boost instead of streak geometry; add streaks if it ever reads weak.
		const speed = cam.pos.distanceTo(prevPos) / Math.max(dt, 1e-3);
		prevPos.copy(cam.pos);
		const warp = Math.min(1, speed / 45);
		U.uWarp.value += (warp - U.uWarp.value) * .12;
		const kick = U.uWarp.value * 9;
		if (Math.abs(kick - fovKick) > .05 || Math.abs(cam.shift - lastShift) > .0005) { fovKick = kick; frameView(); }

		const par = tier === 1 && P.inside ? 1 : 0;
		camera.position.set(cam.pos.x + P.nx * 1.6 * par + Math.sin(t * .11) * 1.2, cam.pos.y - P.ny * 1.1 * par + Math.sin(t * .07) * .6, cam.pos.z);
		camera.lookAt(cam.look);
		spinV *= P.drag ? 1 : .96; if (!P.drag) spin += spinV;
		audience.rotation.y = t * .05 + spin;
		camera.updateMatrixWorld();
		scene.updateMatrixWorld();

		if (t0 < 0) t0 = t;
		U.uBuild.value = Math.min(1, (t - t0) / BUILD_S);
		U.uTime.value = t; U.uFar.value = cam.far;

		const li = active();
		if (footTop < innerHeight) {                                 // lowest dot of this layer as it sits unlifted, kept FOOT_GAP above the footer
			const m = layers[li].group.matrixWorld; let low = 0;
			for (const nd of layers[li].L.nodes) if (toScreen(lowV.copy(nd.p).applyMatrix4(m), lowS).front) low = Math.max(low, lowS.y + lastLift);
			lift = Math.max(0, low + FOOT_GAP - footTop);
		} else lift = 0;
		if (phoneMQ.matches && sections.length) winY = phoneWin();
		if (lift !== lastLift || Math.abs(winY - lastWinY) > .5) frameView();
		const heroW = sections.length && target.key === 'hero' ? 1 : 0;
		if (snapped && snapped.li !== li) setSnap(null);           // scrolled to another layer: the old bloom goes, card or not
		const hovering = tier === 1 && P.inside && now - P.last < 4000 && U.uBuild.value >= 1;
		U.uForce.value += ((hovering ? 1 : 0) - U.uForce.value) * .12;
		U.uMouse.value.set(P.nx, -P.ny);

		if (hovering) {
			count = updateWires(li);
			wires.visible = cursorGlow.visible = true;
			// magnetic snap with hysteresis: keep the current bloom until the cursor leaves KEEP px
			const keep = snapped && snapped.li === li && (() => { const s = screenOfPick(snapped); return s.front && Math.hypot(s.x - P.x, s.y - P.y) < KEEP; })();
			if (!keep) setSnap(nearestPick(P.x, P.y, SNAP));
			if (li === 3 && t - burstAt > 2.5) burstAt = t;                         // the forward pass
		} else if (tier === 1) {
			wires.visible = cursorGlow.visible = false;
			if (snapped && !P.onCard) setSnap(null);            // reaching for the card keeps it open
			count = 0;
		} else if (U.uBuild.value >= 1 && sections.length && !snapped) {           // tier 2 idle auto-demo
			if (demo && (now > demoEnd || demo.li !== li)) { demo = null; bloomTo = 0; demoTag.style.opacity = 0; }
			if (!demo && now > autoAt) {
				const pool = allPicks.filter(p => { if (p.li !== li || p.kind === 'hub') return false; const s = screenOfPick(p); return s.front && s.y > 100 && s.x > 0 && s.x < innerWidth && !behindCopy(s.x, s.y); });
				if (pool.length) {
					demo = pool[Math.floor(Math.random() * pool.length)];
					demoTag.textContent = demo.t.length > 44 ? demo.t.slice(0, 42).trim() + '…' : demo.t;
					demoTag.style.opacity = 1; bloomTo = 1; demoEnd = now + 2600;
				}
				if (li === 3) burstAt = t;
				autoAt = now + 4200;
			}
		}
		layers.forEach((L, i) => {
			const b = i === 3 ? Math.max(0, 1 - (t - burstAt) / 1.6) : (snapped && snapped.li === i ? U.uBloom.value * .35 : 0);
			L.mats.edge.uniforms.uBurst.value = b;
		});
		U.uBloom.value += (bloomTo - U.uBloom.value) * .14;
		if (!snapped && !demo && U.uBloom.value < .01 && bloomEl && !bloomEl.hidden) {
			bloomEl.hidden = true;
			// a mouse resting where the card was is now over the graph: re-read what is under it
			const under = P.x >= 0 && document.elementFromPoint(P.x, P.y);
			P.onCard = false; P.inside = !!under && !busy(under);
		}

		placeCard();
		flowerFor(snapped && snapped.kind === 'hub' ? snapped : null);
		countTag.style.opacity = hovering && li === 2 && count ? 1 : 0;
		if (hovering && li === 2) {
			countTag.textContent = `≈ ${(count * 1000).toLocaleString('en-US')} people under your cursor`;
			placeTag(countTag, P.x + 22, P.y + 18);
		}
		updateTags(heroW);

		canvas.style.opacity = ready ? cam.dim.toFixed(3) : '';
		renderer.render(scene, camera);
		if (!ready) { ready = true; doc.classList.add('is-ready'); }
	}
	document.addEventListener('visibilitychange', () => {
		if (document.hidden) cancelAnimationFrame(raf);
		else { clock.getDelta(); frame(); }
	});
	frame();

	// for the QA script
	window.__theai = {
		scene, camera, renderer, layers, U, counts: layers.map(l => l.L.nodes.length),
		screenOf: (li, i) => { const p = allPicks.filter(q => q.li === li)[i]; return p ? screenOfPick(p) : null; },
		bloom: () => ({ t: snapped ? snapped.t : null, demo: demo ? demo.t : null, v: U.uBloom.value }),
		wires: () => (wires.visible ? wg.drawRange.count / 2 : 0),
		state: () => ({ inside: P.inside, onCard: P.onCard, x: P.x, y: P.y, count, force: +U.uForce.value.toFixed(2), snapped: snapped ? snapped.t : null, hidden: bloomEl ? bloomEl.hidden : null }),
		active,
	};
}
