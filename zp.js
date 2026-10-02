/* Zingerpaint engine: a photograph -> white lines on black, as gcode that rides on top of an ordinary slicer's output.

A port of the pipeline that printed Child of the Future (~/projects/tetons/child_of_the_future/print/: engrave.target,
spacing_lines.fmap/pieces/block, flow_lines.order/build, iron.heightmap + iron2.passes), kept move-for-move so the two can
be compared. Pure functions, no DOM: the page and the node test (test_zp.js) both run THIS file.

The method (tone_wedge CAIRN, v7): at 0.04 mm layers the bead is ~0.47 mm whatever width is asked for, so tone comes from
SPACING. Ten white layers over a black base; layer k keeps a line wherever the picture's level f (0..10, the photo pushed
through the measured v7 curve) clears k plus that line's golden-ratio threshold. Then a dry ironing pass.

Iris (Opus 5.5) with Ivy, Sep 30 2026.
*/
(function (root) {
  'use strict';
  const BASE = 0.28, H_L = 0.04, N = 10;
  const PITCH = 0.8, STEP = 0.5, MIN_PIECE = 1.5, MIN_GAP = 1.0, GOLD = 0.618034;
  const FIL_AREA = Math.PI * 0.875 * 0.875, FLOW = 0.98;
  const F_LINE = 1800, F_HOP = 12000, RETRACT = 0.4, MARGIN = 0.6, LONG_GAP = 2.0;
  const PPM = 10;                                  // working resolution: 0.1 mm pixels
  const L_BLACK = 3.8, LIFT = 0.75;
  const BEAD = 0.47, IRON_SPACING = 0.2, IRON_SPEED = 30, IRON_STEP = 0.5, R_NOZ = 0.4;

  // ---------- tone ----------
  function curve() {
    const C = root.ZP_CURVE;
    const f = [0].concat(C.f), L = [L_BLACK].concat(C.L);
    for (let i = 1; i < L.length; i++) L[i] = Math.max(L[i], L[i - 1]);   // a level never reads darker than the one below
    return { f, L };
  }
  const Yof = L => (L > 8 ? Math.pow((L + 16) / 116, 3) : L / 903.3);

  function interp(x, xp, fp) {                    // numpy.interp
    const n = xp.length;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
    if (xp[hi] === xp[lo]) return fp[hi];
    return fp[lo] + (x - xp[lo]) * (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]);
  }

  function percentile(sorted, p) {               // numpy default (linear)
    const idx = p / 100 * (sorted.length - 1), lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  // Lanczos-3 resample of an 8-bit grey image, horizontal pass then vertical, each rounded to 8 bits (as PIL does).
  function lanczos(x) {
    if (x === 0) return 1;
    if (x <= -3 || x >= 3) return 0;
    const px = Math.PI * x;
    return 3 * Math.sin(px) * Math.sin(px / 3) / (px * px);
  }
  function resample1d(src, sw, sh, dw, horizontal) {
    const inLen = horizontal ? sw : sh, outLen = dw;
    const scale = inLen / outLen, ss = Math.max(scale, 1), support = 3 * ss;
    const w = horizontal ? dw : sw, h = horizontal ? sh : dw;
    const out = new Uint8Array(w * h);
    for (let o = 0; o < outLen; o++) {
      const center = (o + 0.5) * scale;
      let x0 = Math.max(0, Math.floor(center - support + 0.5)), x1 = Math.min(inLen, Math.floor(center + support + 0.5));
      const ws = []; let tot = 0;
      for (let j = x0; j < x1; j++) { const v = lanczos((j - center + 0.5) / ss); ws.push(v); tot += v; }
      for (let j = 0; j < ws.length; j++) ws[j] /= tot;
      if (horizontal) {
        for (let r = 0; r < sh; r++) {
          let acc = 0; const base = r * sw;
          for (let j = 0; j < ws.length; j++) acc += ws[j] * src[base + x0 + j];
          out[r * dw + o] = Math.min(255, Math.max(0, Math.round(acc)));
        }
      } else {
        for (let c = 0; c < sw; c++) {
          let acc = 0;
          for (let j = 0; j < ws.length; j++) acc += ws[j] * src[(x0 + j) * sw + c];
          out[o * sw + c] = Math.min(255, Math.max(0, Math.round(acc)));
        }
      }
    }
    return out;
  }
  function resize(grey, w, h, dw, dh) {
    const a = resample1d(grey, w, h, dw, true);
    return resample1d(a, dw, h, dh, false);
  }

  function gaussian(arr, w, h, sigma) {          // scipy.ndimage.gaussian_filter, mode 'reflect', truncate 4
    const r = Math.floor(4 * sigma + 0.5), k = [];
    let s = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * i * i / (sigma * sigma)); k.push(v); s += v; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    const refl = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };
    const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
    for (let c = 0; c < w; c++)                   // axis 0 first, as scipy
      for (let y = 0; y < h; y++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) acc += k[i + r] * arr[refl(y + i, h) * w + c];
        tmp[y * w + c] = acc;
      }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) acc += k[i + r] * tmp[y * w + refl(x + i, w)];
        out[y * w + x] = acc;
      }
    return out;
  }

  /* grey: Uint8Array of 8-bit luma (ITU-R 601: 0.299 R + 0.587 G + 0.114 B, as PIL's convert('L')), iw x ih.
     Returns the level map f (0..10) at 0.1 mm pixels, y down, plus the plate size in mm. */
  function fmap(grey, iw, ih, widthMM, crop) {
    crop = crop || 0;
    let g = grey, w = iw, h = ih;
    if (crop) {
      w = iw - 2 * crop; h = ih - 2 * crop; g = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) g.set(grey.subarray((y + crop) * iw + crop, (y + crop) * iw + crop + w), y * w);
    }
    const dw = Math.round(widthMM * PPM), dh = Math.round(h / w * widthMM * PPM);
    const im = resize(g, w, h, dw, dh);
    const n = dw * dh, L = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = im[i] / 255, Y = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
      L[i] = Y > 0.008856 ? 116 * Math.cbrt(Y) - 16 : 903.3 * Y;
    }
    const sorted = Float64Array.from(L).sort();
    const lo = percentile(sorted, 1), hi = percentile(sorted, 99);
    const C = curve(), Lmax = C.L[C.L.length - 1];
    for (let i = 0; i < n; i++) {
      const x = (Math.min(Math.max(L[i], lo), hi) - lo) / (hi - lo || 1);
      L[i] = L_BLACK + Math.pow(x, LIFT) * (Lmax - L_BLACK);
    }
    const Lt = gaussian(L, dw, dh, 0.3 * PPM);
    const Yc = C.L.map(Yof), f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = interp(Yof(Lt[i]), Yc, C.f);
    return { f, w: dw, h: dh, W: dw / PPM, H: dh / PPM };
  }

  /* The plate: f map placed on the bed with its bottom-left corner at (OX, OY). */
  function plate(fm, OX, OY) {
    return {
      f: fm.f, fw: fm.w, fh: fm.h, W: fm.W, H: fm.H, OX, OY,
      level(x, y) {
        let r = Math.trunc((OY + this.H - y) * PPM), c = Math.trunc((x - OX) * PPM);
        r = Math.min(Math.max(r, 0), this.fh - 1); c = Math.min(Math.max(c, 0), this.fw - 1);
        return this.f[r * this.fw + c];
      }
    };
  }

  // ---------- lines ----------
  // numpy.arange exactly: it steps by (a + s) - a, not by s. Lines sit on 0.1 mm pixel edges (81.6 = OX + 3.6), so the
  // last bit decides which column a line reads: a naive a + i*s put 81.60000000000001 one column right of numpy's 81.6
  // and changed a few pieces per layer.
  const arange = (a, b, s) => {
    const n = Math.max(0, Math.ceil((b - a) / s)), d = (a + s) - a, o = [];
    for (let i = 0; i < n; i++) o.push(a + i * d);
    return o;
  };

  function families(method, P) {
    const { OX, OY, W, H } = P;
    const rows = off => arange(OY + off, OY + H - 0.2, PITCH).map(y => [[OX + 0.3, y], [OX + W - 0.3, y]]);
    const cols = off => arange(OX + off, OX + W - 0.2, PITCH).map(x => [[x, OY + 0.3], [x, OY + H - 0.3]]);
    const diag = off => {
      const out = [], s = Math.SQRT1_2, n = [-s, s], u = [s, s];
      const ds = [[OX, OY], [OX + W, OY], [OX, OY + H], [OX + W, OY + H]].map(p => p[0] * n[0] + p[1] * n[1]);
      for (const d of arange(Math.min(...ds) + off, Math.max(...ds), PITCH)) {
        const base = [d * n[0], d * n[1]];
        const tx = [(OX + 0.3 - base[0]) / u[0], (OX + W - 0.3 - base[0]) / u[0]].sort((a, b) => a - b);
        const ty = [(OY + 0.3 - base[1]) / u[1], (OY + H - 0.3 - base[1]) / u[1]].sort((a, b) => a - b);
        const t0 = Math.max(tx[0], ty[0]), t1 = Math.min(tx[1], ty[1]);
        if (t1 - t0 > 2) out.push([[base[0] + t0 * u[0], base[1] + t0 * u[1]], [base[0] + t1 * u[0], base[1] + t1 * u[1]]]);
      }
      return out;
    };
    const M = { horizontal: [rows(0.4), rows(0.8)], vertical: [cols(0.4), cols(0.8)],
                cross: [rows(0.4), cols(0.8)], diagonal: [diag(0.4), diag(0.8)] }[method];
    if (!M) throw new Error('unknown line direction ' + method);
    return [order(M[0]), order(M[1])];
  }

  function order(paths) {                         // greedy: nearest path end next, reversed if that end is closer
    const left = paths.map((_, i) => i), out = [];
    let cx = 0, cy = 0;
    while (left.length) {
      let best = 0, bestD = Infinity, rev = false;
      for (let j = 0; j < left.length; j++) {
        const p = paths[left[j]], a = p[0], b = p[p.length - 1];
        const d0 = Math.hypot(a[0] - cx, a[1] - cy), d1 = Math.hypot(b[0] - cx, b[1] - cy);
        const d = Math.min(d0, d1);
        if (d < bestD) { bestD = d; best = j; rev = d0 > d1; }
      }
      const k = left.splice(best, 1)[0];
      const p = rev ? paths[k].slice().reverse() : paths[k];
      out.push(p); cx = p[p.length - 1][0]; cy = p[p.length - 1][1];
    }
    return out;
  }

  function resample(p, step) {
    const s = [0];
    for (let i = 1; i < p.length; i++) s.push(s[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
    const end = s[s.length - 1];
    if (end < step) return p;
    const xs = p.map(q => q[0]), ys = p.map(q => q[1]);
    return arange(0, end, step).map(t => [interp(t, s, xs), interp(t, s, ys)]);
  }

  function pieces(P, k, fam) {
    const out = [];
    for (let i = 0; i < fam.length; i++) {
      const p = resample(fam[i], STEP);
      if (p.length < 2) continue;
      const T = ((i + 1) * GOLD + k * 0.381966) % 1.0;
      const idx = [];
      for (let j = 0; j < p.length; j++) {
        const c = Math.min(Math.max(P.level(p[j][0], p[j][1]) - k, 0), 1);
        if (c > T) idx.push(j);
      }
      if (!idx.length) continue;
      const runs = []; let start = idx[0];
      for (let j = 1; j < idx.length; j++)
        if ((idx[j] - idx[j - 1]) * STEP > MIN_GAP) { runs.push([start, idx[j - 1]]); start = idx[j]; }
      runs.push([start, idx[idx.length - 1]]);
      for (const [a, b] of runs) if ((b - a) * STEP >= MIN_PIECE) out.push(p.slice(a, b + 1));
    }
    return out;
  }

  const f3 = v => v.toFixed(3), f5 = v => v.toFixed(5), f2 = v => v.toFixed(2);
  const eMM = w => ((w - H_L) * H_L + Math.PI * (H_L / 2) * (H_L / 2)) / FIL_AREA * FLOW;

  function block(ps, k, z) {
    const e = eMM(PITCH);
    const out = [`; SPACING layer ${k}: ${ps.length} pieces`, '; FEATURE: Top surface', `; LINE_WIDTH: ${f3(PITCH)}`];
    let pos = null;
    for (const p of ps) {
      const a = p[0];
      const gap = pos ? Math.hypot(a[0] - pos[0], a[1] - pos[1]) : 99;
      if (!pos || gap > LONG_GAP) {
        if (pos) out.push(`G1 E${f2(-RETRACT)} F1800`);
        out.push(`G1 Z${f3(z + 0.2)} F${F_HOP}`, `G1 X${f3(a[0])} Y${f3(a[1])} F${F_HOP}`, `G1 Z${f3(z)}`);
        if (pos) out.push(`G1 E${f2(RETRACT)} F1800`);
      } else {
        out.push(`G1 Z${f3(z + 0.1)} F${F_HOP}`, `G1 X${f3(a[0])} Y${f3(a[1])}`, `G1 Z${f3(z)}`);
      }
      for (let j = 1; j < p.length; j++) {
        const u = p[j - 1], v = p[j];
        out.push(`G1 X${f3(v[0])} Y${f3(v[1])} E${f5(e * Math.hypot(v[0] - u[0], v[1] - u[1]))} F${F_LINE}`);
      }
      pos = p[p.length - 1];
    }
    out.push(`G1 Z${f3(z + 0.2)} F${F_HOP}`);
    return out;
  }

  /* Every layer's kept pieces, in print order. */
  function layers(P, method) {
    const fams = families(method, P), out = [];
    for (let k = 0; k < N; k++) out.push(order(pieces(P, k, fams[k % 2])));
    return out;
  }

  // ---------- rasters: what lands where ----------
  /* Draw polylines into a Uint8 mask (px per mm, bed -> image y down) as capsules of the given width. */
  function stroke(mask, mw, mh, P, polys, widthMM, px) {
    const r = widthMM / 2 * px, r2 = r * r;
    for (const p of polys)
      for (let j = 1; j < p.length; j++) {
        const ax = (p[j - 1][0] - P.OX) * px, ay = (P.OY + P.H - p[j - 1][1]) * px;
        const bx = (p[j][0] - P.OX) * px, by = (P.OY + P.H - p[j][1]) * px;
        const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy || 1e-12;
        const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - r)), x1 = Math.min(mw - 1, Math.ceil(Math.max(ax, bx) + r));
        const y0 = Math.max(0, Math.floor(Math.min(ay, by) - r)), y1 = Math.min(mh - 1, Math.ceil(Math.max(ay, by) + r));
        for (let y = y0; y <= y1; y++)
          for (let x = x0; x <= x1; x++) {
            const cx = x + 0.5, cy = y + 0.5;
            let t = ((cx - ax) * dx + (cy - ay) * dy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t;
            const ex = ax + t * dx - cx, ey = ay + t * dy - cy;
            if (ex * ex + ey * ey <= r2) mask[y * mw + x] = 1;
          }
      }
  }

  /* Per pixel (0.1 mm): how many white layers cover it, and the top height. */
  function stack(P, L) {
    const w = Math.round(P.W * PPM), h = Math.round(P.H * PPM);
    const count = new Uint8Array(w * h), top = new Float64Array(w * h).fill(BASE);
    const m = new Uint8Array(w * h);
    for (let k = 0; k < L.length; k++) {
      m.fill(0);
      stroke(m, w, h, P, L[k], BEAD, PPM);
      const z = +(BASE + (k + 1) * H_L).toFixed(2);
      for (let i = 0; i < m.length; i++) if (m[i]) { count[i]++; top[i] = z; }
    }
    return { w, h, count, top };
  }

  /* The step-back look (methods.py's top row, the sheet Ivy chose Child of the Future's lines from): each layer's kept
     pieces drawn at the lattice width (0.8), averaged over 1 mm, summed, and read through the measured v7 curve.
     Returns 8-bit sRGB grey, one value per 0.1 mm pixel. */
  function stepBack(P, L) {
    const w = Math.round(P.W * PPM), h = Math.round(P.H * PPM), n = w * h;
    const feff = new Float64Array(n), m = new Uint8Array(n), row = new Float64Array(n), C = curve();
    const half = PPM >> 1;                         // uniform_filter(size 10): window [i-5, i+4], edges reflected
    const refl = (i, len) => { while (i < 0 || i >= len) i = i < 0 ? -i - 1 : 2 * len - i - 1; return i; };
    for (let k = 0; k < L.length; k++) {
      m.fill(0);
      stroke(m, w, h, P, L[k], PITCH, PPM);
      for (let y = 0; y < h; y++) {
        let s = 0;
        for (let i = -half; i < PPM - half; i++) s += m[y * w + refl(i, w)];
        for (let x = 0; x < w; x++) {
          row[y * w + x] = s / PPM;
          s += m[y * w + refl(x + PPM - half, w)] - m[y * w + refl(x - half, w)];
        }
      }
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let i = -half; i < PPM - half; i++) s += row[refl(i, h) * w + x];
        for (let y = 0; y < h; y++) {
          feff[y * w + x] += s / PPM;
          s += row[refl(y + PPM - half, h) * w + x] - row[refl(y - half, h) * w + x];
        }
      }
    }
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const Y = Yof(interp(feff[i], C.f, C.L));
      out[i] = Math.round(255 * Math.min(1, Math.max(0, Y <= 0.0031308 ? 12.92 * Y : 1.055 * Math.pow(Y, 1 / 2.4) - 0.055)));
    }
    return { w, h, grey: out };
  }

  /* The line-level look (frame/mockup_lines.py, made when Ivy said "The layer lines need to be clearer"): every line
     drawn at the bead's real 0.47 mm at R px/mm, a spot's lightness set by how many layers stack there (the v6 treads),
     then each 1 mm neighbourhood scaled to what the MEASURED v7 curve says it reads as -- the drawn gaps are pure black,
     but real white plastic spreads light into them, so without the gain it reads too dark. Returns 8-bit sRGB. */
  function lineView(P, L, R) {
    const w = Math.round(P.W * R), h = Math.round(P.H * R), n = w * h, C = curve();
    const cnt = new Uint8Array(n), cov = new Uint8Array(n), m = new Uint8Array(n);
    for (let k = 0; k < L.length; k++) {
      m.fill(0); stroke(m, w, h, P, L[k], BEAD, R);
      for (let i = 0; i < n; i++) cnt[i] += m[i];
      m.fill(0); stroke(m, w, h, P, L[k], PITCH, R);
      for (let i = 0; i < n; i++) cov[i] += m[i];
    }
    const size = Math.max(1, Math.round(R));       // 1 mm box; linear, so one filter over the summed layers
    const feff = boxFilter(cov, w, h, size), Y8 = new Float32Array(n);
    const TY = [4, 13, 24, 33, 42, 49, 55, 61, 67, 71, 72].map(Yof);
    for (let i = 0; i < n; i++) Y8[i] = TY[Math.min(cnt[i], 10)];
    const mean = boxFilter(Y8, w, h, size), out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const gain = Math.min(4, Math.max(0.5, Yof(interp(feff[i], C.f, C.L)) / Math.max(mean[i], 1e-4)));
      const Y = Math.min(1, Y8[i] * gain);
      out[i] = Math.round(255 * (Y <= 0.0031308 ? 12.92 * Y : 1.055 * Math.pow(Y, 1 / 2.4) - 0.055));
    }
    return { w, h, grey: out, R };
  }

  function boxFilter(a, w, h, size) {             // uniform_filter: window [i - size/2, i + size - size/2), reflected
    const half = size >> 1, row = new Float32Array(w * h), out = new Float32Array(w * h);
    const refl = (i, len) => { while (i < 0 || i >= len) i = i < 0 ? -i - 1 : 2 * len - i - 1; return i; };
    for (let y = 0; y < h; y++) {
      const o = y * w; let s = 0;
      for (let i = -half; i < size - half; i++) s += a[o + refl(i, w)];
      for (let x = 0; x < w; x++) { row[o + x] = s / size; s += a[o + refl(x + size - half, w)] - a[o + refl(x - half, w)]; }
    }
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -half; i < size - half; i++) s += row[refl(i, h) * w + x];
      for (let y = 0; y < h; y++) { out[y * w + x] = s / size; s += row[refl(y + size - half, h) * w + x] - row[refl(y - half, h) * w + x]; }
    }
    return out;
  }

  // ---------- ironing (iron.py + iron2.py) ----------
  function ironing(P, S, whiteDir) {
    const { w: W, h: H, top } = S;
    const r = Math.round(R_NOZ * PPM), offs = [];
    for (let yy = -r; yy <= r; yy++) for (let xx = -r; xx <= r; xx++) if (xx * xx + yy * yy <= r * r) offs.push([xx, yy]);
    const ride = new Float64Array(W * H);
    for (let y = 0; y < H; y++)                   // the nozzle rides the MAX height within its flat
      for (let x = 0; x < W; x++) {
        let m = 0;
        for (const [ox, oy] of offs) {
          const yy = y + oy, xx = x + ox;
          if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;   // scipy pads with the nearest; the max is unchanged
          const v = top[yy * W + xx]; if (v > m) m = v;
        }
        ride[y * W + x] = m;
      }
    let white = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) white[i] = top[i] > BASE + 0.001 ? 1 : 0;
    for (let it = 0; it < Math.round(0.5 * PPM); it++) {   // binary_dilation, cross, 5 iterations
      const nw = white.slice();
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          if (white[y * W + x]) continue;
          if ((x > 0 && white[y * W + x - 1]) || (x < W - 1 && white[y * W + x + 1]) ||
              (y > 0 && white[(y - 1) * W + x]) || (y < H - 1 && white[(y + 1) * W + x])) nw[y * W + x] = 1;
        }
      white = nw;
    }
    let zmax = 0; for (let i = 0; i < ride.length; i++) if (ride[i] > zmax) zmax = ride[i];
    const zlift = zmax + 0.3, F = (IRON_SPEED * 60).toFixed(0);

    function passes(inRegion, vertical) {
      const { OX, OY } = P, out = [];
      let length = 0, nlift = 0;
      const lines = vertical ? arange(OX + 0.3, OX + P.W - 0.3, IRON_SPACING) : arange(OY + 0.3, OY + P.H - 0.3, IRON_SPACING);
      const along = vertical ? arange(OY + 0.3, OY + P.H - 0.3, IRON_STEP) : arange(OX + 0.3, OX + P.W - 0.3, IRON_STEP);
      const clampR = v => Math.min(Math.max(Math.trunc(v), 0), H - 1), clampC = v => Math.min(Math.max(Math.trunc(v), 0), W - 1);
      const pix = vertical ? (ln, a) => clampR((OY + P.H - a) * PPM) * W + clampC((ln - OX) * PPM)
                           : (ln, a) => clampR((OY + P.H - ln) * PPM) * W + clampC((a - OX) * PPM);
      const xy = vertical ? (ln, a) => [ln, a] : (ln, a) => [a, ln];
      lines.forEach((ln, i) => {
        const ids = along.map(a => pix(ln, a));
        const inside = ids.map(q => inRegion(q)), zs = ids.map(q => ride[q]);
        let sp = [];                                // spans of at least 1 mm
        for (let a = 0; a < inside.length;) {
          if (!inside[a]) { a++; continue; }
          let b = a; while (b < inside.length && inside[b]) b++;
          if (b - a >= Math.trunc(1.0 / IRON_STEP)) sp.push([a, b]);
          a = b;
        }
        if (i % 2) sp = sp.reverse().map(([a, b]) => [b, a]);   // serpentine
        for (const [a, b] of sp) {
          const idx = [];
          if (b > a) for (let j = a; j < b; j++) idx.push(j); else for (let j = a - 1; j > b - 1; j--) idx.push(j);
          const [x0, y0] = xy(ln, along[idx[0]]);
          out.push(`G1 Z${f3(zlift)} F1200`, `G1 X${f3(x0)} Y${f3(y0)} F12000`, `G1 Z${f3(zs[idx[0]])} F1200`);
          nlift++;
          let cur = zs[idx[0]];
          for (let j = 1; j < idx.length; j++) {
            const q = idx[j];
            if (zs[q] !== cur || j === idx.length - 1) {
              const [x, y] = xy(ln, along[q]);
              out.push(`G1 X${f3(x)} Y${f3(y)} Z${f3(zs[q])} F${F}`); cur = zs[q];
            }
          }
          length += Math.abs(along[idx[idx.length - 1]] - along[idx[0]]);
        }
      });
      out.push(`G1 Z${f3(zlift)} F1200`);
      return { out, length, nlift };
    }
    const vWhite = whiteDir === 'vertical';
    let parts;
    if (vWhite) parts = [passes(q => white[q] === 1, true), passes(q => white[q] === 0, false)];
    else parts = [passes(() => true, false)];      // white and black both horizontal: one pass over the whole face
    const blk = ['; IRONING (zingerpaint): no flow, riding each spot\'s own height', '; FEATURE: Ironing', 'G1 E-0.8 F1800'];
    for (const p of parts) blk.push(...p.out);     // no unretract after: it would drop a blob on the face
    const length = parts.reduce((s, p) => s + p.length, 0), lifts = parts.reduce((s, p) => s + p.nlift, 0);
    return { block: blk, length, lifts, seconds: length / IRON_SPEED + lifts * 0.3 };
  }

  // ---------- the slicer's gcode ----------
  const NUM = '(-?\\d*\\.?\\d+)';
  const RX = { X: new RegExp('\\bX' + NUM), Y: new RegExp('\\bY' + NUM), E: new RegExp('\\bE' + NUM) };
  const getv = (s, k) => { const m = RX[k].exec(s); return m ? parseFloat(m[1]) : null; };
  const layerZ = raw => raw.startsWith('; Z_HEIGHT:') ? +parseFloat(raw.slice(11)).toFixed(2)
                      : raw.startsWith(';Z:') ? +parseFloat(raw.slice(3)).toFixed(2) : null;
  const isWhiteZ = z => z !== null && z >= BASE + H_L - 0.005 && z <= BASE + N * H_L + 0.005;

  /* Read what the slicer made: white layers present, the white box, extrusion mode, the slicer's time outside the white. */
  function inspect(text) {
    const lines = text.split('\n');
    let z = null, x = null, y = null, rel = null;
    const zs = new Set();
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity, nWhite = 0;
    let r0 = null, rWhiteStart = null, rWhiteEnd = null, lastR = null, inWhite = false;
    let tool = 0;                                  // the filament in force: last bare T<n>, n < 16 (T255/T1000 are not filaments)
    const baseTools = new Set(), whiteTools = new Set();
    for (const raw of lines) {
      const tm = /^\s*T(\d+)\s*(;|$)/.exec(raw);
      if (tm && +tm[1] < 16) tool = +tm[1];
      const lz = layerZ(raw);
      if (lz !== null) {
        z = lz; zs.add(z);
        if (isWhiteZ(z) && !inWhite) { inWhite = true; rWhiteStart = lastR; }
        if (!isWhiteZ(z) && inWhite && rWhiteEnd === null) rWhiteEnd = lastR;
      }
      if (raw.startsWith('M83')) rel = true; else if (raw.startsWith('M82')) rel = false;
      if (raw.startsWith('M73')) { const m = /R(\d+)/.exec(raw); if (m) { lastR = +m[1]; if (r0 === null) r0 = lastR; } }
      const s = raw.split(';')[0].trim();
      if (!/^G[0-3]\b/.test(s)) continue;
      const nx = getv(s, 'X'), ny = getv(s, 'Y'), e = getv(s, 'E');
      const tx = nx !== null ? nx : x, ty = ny !== null ? ny : y;
      if (isWhiteZ(z) && e !== null && e > 0 && x !== null && (nx !== null || ny !== null)) {
        nWhite++; whiteTools.add(tool);
        bx0 = Math.min(bx0, x, tx); bx1 = Math.max(bx1, x, tx); by0 = Math.min(by0, y, ty); by1 = Math.max(by1, y, ty);
      }
      if (z !== null && z <= BASE + 0.005 && e !== null && e > 0 && (nx !== null || ny !== null)) baseTools.add(tool);
      x = tx; y = ty;
    }
    if (inWhite && rWhiteEnd === null) rWhiteEnd = lastR;
    const white = [...zs].filter(isWhiteZ).sort((a, b) => a - b);
    const otherMin = (r0 !== null && rWhiteStart !== null) ? (r0 - rWhiteStart) + (rWhiteEnd || 0) : null;
    return { white, relative: rel, nWhite, box: nWhite ? [bx0, by0, bx1, by1] : null, hasBase: zs.has(BASE), otherMin,
             baseTools: [...baseTools], whiteTools: [...whiteTools] };
  }

  /* Check the gcode against the blank this picture needs; returns plain-words problems (empty = fine). */
  function check(info, W, H) {
    const bad = [];
    const want = []; for (let k = 1; k <= N; k++) want.push(+(BASE + k * H_L).toFixed(2));
    if (info.relative === false) bad.push('This gcode uses absolute extrusion (M82). Turn on relative E distances in the slicer.');
    if (!info.hasBase || want.some(z => !info.white.includes(z)))
      bad.push(`The white has to be ten layers of 0.04 mm on top of a 0.28 mm base. Found white layers at ${info.white.join(', ') || 'none'}. Set the layer height to 0.04 and the first layer to 0.2.`);
    if (!info.box) bad.push('No white printing found in the gcode. Is the white part on filament 2?');
    else if (info.baseTools && info.baseTools.some(t => info.whiteTools.includes(t)))
      bad.push(`The black base and the white both print on filament ${info.whiteTools.map(t => t + 1).join(', ')}, so there would be no picture. ` +
               'In Orca\u2019s object list, open the blank, set the part \u201cblack base\u201d to filament 1 and \u201cwhite\u201d to filament 2, then slice again.');
    else {
      const bw = info.box[2] - info.box[0], bh = info.box[3] - info.box[1];
      if (Math.abs(bw - W) > 1.5 || Math.abs(bh - H) > 1.5)
        bad.push(`The white in this gcode is ${bw.toFixed(1)} x ${bh.toFixed(1)} mm, but this picture is ${W.toFixed(1)} x ${H.toFixed(1)} mm. Slice the blank made for this picture, at this size.`);
    }
    return bad;
  }

  /* Delete the slicer's white inside the box, write our layers there (and the ironing after the last). */
  function rewrite(text, P, L, iron) {
    const src = text.split('\n'), out = [];
    let z = null, x = null, y = null, removed = 0;
    const done = new Set();
    const ins = (u, v) => P.OX - MARGIN <= u && u <= P.OX + P.W + MARGIN && P.OY - MARGIN <= v && v <= P.OY + P.H + MARGIN;
    for (const raw of src) {
      const lz = layerZ(raw);
      if (lz !== null) z = lz;
      const s = raw.split(';')[0].trim();
      if (!/^G[0-3]\b/.test(s)) { out.push(raw); continue; }
      const nx = getv(s, 'X'), ny = getv(s, 'Y'), e = getv(s, 'E');
      const tx = nx !== null ? nx : x, ty = ny !== null ? ny : y;
      const hit = isWhiteZ(z) && x !== null && (nx !== null || ny !== null) && ins(x, y) && ins(tx, ty);
      if (hit && e !== null && e > 0) {
        const k = Math.round((z - BASE) / H_L) - 1;
        if (!done.has(k)) {
          out.push(...block(L[k], k, z)); done.add(k);
          if (k === N - 1 && iron) out.push(...iron.block);
        }
        out.push('; ZINGERPAINT removed: ' + raw.trim()); removed++;
      } else if (hit && e !== null && e < 0) {
        out.push(`G1 E${f5(e)} F1800 ; ZINGERPAINT kept wipe retraction`);
      } else out.push(raw);
      x = tx; y = ty;
    }
    return { text: out.join('\n'), layers: [...done].sort((a, b) => a - b), removed };
  }

  /* Seconds our moves take: lines at their feed, hops, retracts (the estimator validated on the prints within ~2-9 min). */
  function ourSeconds(L) {
    let s = 0;
    for (const ps of L) {
      let pos = null;
      for (const p of ps) {
        if (pos) s += 0.13;
        for (let j = 1; j < p.length; j++) s += Math.hypot(p[j][0] - p[j - 1][0], p[j][1] - p[j - 1][1]) / (F_LINE / 60);
        pos = p[p.length - 1];
      }
    }
    return s;
  }

  // ---------- the blank: a two-part 3mf to slice ----------
  function crc32(buf) {
    let c, crc = 0xFFFFFFFF;
    for (let n = 0; n < buf.length; n++) {
      c = (crc ^ buf[n]) & 0xFF;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      crc = (crc >>> 8) ^ c;
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }
  function zip(files) {                           // STORE-only zip; files: [[name, string]]
    const enc = new TextEncoder(), parts = [], central = [];
    let off = 0;
    for (const [name, str] of files) {
      const nb = enc.encode(name), db = enc.encode(str), crc = crc32(db);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(8, 0, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, db.length, true); lh.setUint32(22, db.length, true);
      lh.setUint16(26, nb.length, true);
      parts.push(new Uint8Array(lh.buffer), nb, db);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true);
      ch.setUint32(16, crc, true); ch.setUint32(20, db.length, true); ch.setUint32(24, db.length, true);
      ch.setUint16(28, nb.length, true); ch.setUint32(42, off, true);
      central.push(new Uint8Array(ch.buffer), nb);
      off += 30 + nb.length + db.length;
    }
    const csize = central.reduce((s, a) => s + a.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, csize, true); end.setUint32(16, off, true);
    const all = parts.concat(central, [new Uint8Array(end.buffer)]);
    const out = new Uint8Array(all.reduce((s, a) => s + a.length, 0));
    let p = 0; for (const a of all) { out.set(a, p); p += a.length; }
    return out;
  }
  function boxMesh(id, name, x0, y0, z0, x1, y1, z1) {
    const v = [[x0, y0, z1], [x0, y1, z1], [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z1], [x1, y0, z1], [x1, y1, z0]];
    const t = [[0, 1, 2], [3, 0, 2], [2, 1, 4], [4, 3, 2], [0, 5, 1], [6, 0, 3], [6, 5, 0], [1, 5, 4], [7, 3, 4], [4, 5, 7], [7, 6, 3], [5, 6, 7]];
    return `<object id="${id}" name="${name}" type="model"><mesh><vertices>` +
      v.map(p => `<vertex x="${p[0]}" y="${p[1]}" z="${p[2]}"/>`).join('') + '</vertices><triangles>' +
      t.map(q => `<triangle v1="${q[0]}" v2="${q[1]}" v3="${q[2]}"/>`).join('') + '</triangles></mesh></object>';
  }
  /* One object, two parts: black base on filament 1, white slab on filament 2, centred at (cx, cy). */
  function blank3mf(W, H, cx, cy) {
    cx = cx === undefined ? 128 : cx; cy = cy === undefined ? 128 : cy;
    const x0 = +(cx - W / 2).toFixed(3), y0 = +(cy - H / 2).toFixed(3), x1 = +(cx + W / 2).toFixed(3), y1 = +(cy + H / 2).toFixed(3);
    const model = '<?xml version="1.0" encoding="UTF-8"?>\n<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" unit="millimeter"><resources>' +
      boxMesh(1, 'black base', x0, y0, 0, x1, y1, BASE) + boxMesh(2, 'white', x0, y0, BASE, x1, y1, +(BASE + N * H_L).toFixed(2)) +
      '<object id="3" name="zingerpaint blank" type="model"><components><component objectid="1"/><component objectid="2"/></components></object>' +
      '</resources><build><item objectid="3"/></build></model>\n';
    const settings = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="3">\n' +
      '    <metadata key="name" value="zingerpaint blank"/>\n' +
      '    <metadata key="layer_height" value="0.04"/>\n' +
      '    <metadata key="solid_infill_direction" value="0"/>\n' +
      '    <metadata key="extruder" value="1"/>\n' +
      '    <part id="1" subtype="normal_part"><metadata key="name" value="black base"/><metadata key="extruder" value="1"/></part>\n' +
      '    <part id="2" subtype="normal_part"><metadata key="name" value="white"/><metadata key="extruder" value="2"/></part>\n' +
      '  </object>\n</config>\n';
    const types = '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>' +
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
      '<Default Extension="config" ContentType="text/xml"/></Types>\n';
    const rels = '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
      '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>\n';
    return zip([['[Content_Types].xml', types], ['_rels/.rels', rels], ['3D/3dmodel.model', model],
                ['Metadata/model_settings.config', settings]]);
  }

  // ---------- the slicer's own package: .gcode.3mf in, .gcode.3mf out ----------
  /* A sliced .gcode.3mf (Orca's / Bambu's "export plate sliced file") is a zip whose Metadata/plate_N.gcode is the job,
     next to plate_N.gcode.md5 (uppercase hex) and slice_info.config (the printer's predicted seconds). Swapping only
     those three keeps everything a slicer or a Bambu printer needs to open and send it. */
  async function inflate(bytes) {
    const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  async function deflate(bytes) {
    const s = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  async function unzip(buf) {
    const b = buf instanceof Uint8Array ? buf : new Uint8Array(buf), dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    let e = b.length - 22;
    while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
    if (e < 0) throw new Error('That file is not a readable 3mf (no zip directory).');
    const count = dv.getUint16(e + 10, true);
    let p = dv.getUint32(e + 16, true);
    const out = [], dec = new TextDecoder();
    for (let i = 0; i < count; i++) {
      if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('That 3mf has a damaged directory.');
      const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true);
      const nlen = dv.getUint16(p + 28, true), xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
      const lho = dv.getUint32(p + 42, true), name = dec.decode(b.subarray(p + 46, p + 46 + nlen));
      const start = lho + 30 + dv.getUint16(lho + 26, true) + dv.getUint16(lho + 28, true);
      const raw = b.subarray(start, start + csize);
      if (!name.endsWith('/')) out.push({ name, data: method === 0 ? raw.slice() : method === 8 ? await inflate(raw) : null, method });
      p += 46 + nlen + xlen + clen;
    }
    if (out.some(f => !f.data)) throw new Error('That 3mf uses a compression this page cannot read.');
    return out;
  }
  async function zipDeflated(files) {           // files: [{name, data}] -> deflated zip
    const enc = new TextEncoder(), parts = [], central = [];
    let off = 0;
    for (const f of files) {
      const nb = enc.encode(f.name), crc = crc32(f.data), comp = await deflate(f.data);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(8, 8, true); lh.setUint16(12, 0x5D3E, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, comp.length, true); lh.setUint32(22, f.data.length, true);
      lh.setUint16(26, nb.length, true);
      parts.push(new Uint8Array(lh.buffer), nb, comp);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(10, 8, true);
      ch.setUint16(14, 0x5D3E, true); ch.setUint32(16, crc, true); ch.setUint32(20, comp.length, true);
      ch.setUint32(24, f.data.length, true); ch.setUint16(28, nb.length, true); ch.setUint32(42, off, true);
      central.push(new Uint8Array(ch.buffer), nb);
      off += 30 + nb.length + comp.length;
    }
    const csize = central.reduce((s, a) => s + a.length, 0), end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, csize, true); end.setUint32(16, off, true);
    const all = parts.concat(central, [new Uint8Array(end.buffer)]);
    const out = new Uint8Array(all.reduce((s, a) => s + a.length, 0));
    let q = 0; for (const a of all) { out.set(a, q); q += a.length; }
    return out;
  }

  function md5(bytes) {                         // RFC 1321; returns uppercase hex, the way Bambu writes plate_N.gcode.md5
    const K = new Int32Array(64), S = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    for (let i = 0; i < 64; i++) K[i] = Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) | 0;
    const n = bytes.length, words = ((n + 8) >>> 6) + 1, M = new Int32Array(words * 16);
    for (let i = 0; i < n; i++) M[i >> 2] |= bytes[i] << ((i % 4) * 8);
    M[n >> 2] |= 0x80 << ((n % 4) * 8);
    M[words * 16 - 2] = (n * 8) | 0; M[words * 16 - 1] = Math.floor(n / 536870912) | 0;
    let a0 = 0x67452301, b0 = 0xefcdab89 | 0, c0 = 0x98badcfe | 0, d0 = 0x10325476;
    for (let o = 0; o < M.length; o += 16) {
      let A = a0, B = b0, C = c0, D = d0;
      for (let i = 0; i < 64; i++) {
        let F, g;
        if (i < 16) { F = (B & C) | (~B & D); g = i; }
        else if (i < 32) { F = (D & B) | (~D & C); g = (5 * i + 1) % 16; }
        else if (i < 48) { F = B ^ C ^ D; g = (3 * i + 5) % 16; }
        else { F = C ^ (B | ~D); g = (7 * i) % 16; }
        const s = S[(i >> 4) * 4 + (i % 4)], t = (A + F + K[i] + M[o + g]) | 0;
        A = D; D = C; C = B; B = (B + ((t << s) | (t >>> (32 - s)))) | 0;
      }
      a0 = (a0 + A) | 0; b0 = (b0 + B) | 0; c0 = (c0 + C) | 0; d0 = (d0 + D) | 0;
    }
    let hex = '';
    for (const v of [a0, b0, c0, d0]) for (let i = 0; i < 4; i++) hex += ((v >>> (i * 8)) & 0xff).toString(16).padStart(2, '0');
    return hex.toUpperCase();
  }

  /* Open a sliced package: the first plate's gcode as text, and everything needed to put a new one back. */
  async function openPackage(buf) {
    const files = await unzip(buf);
    const g = files.find(f => /^Metadata\/plate_\d+\.gcode$/.test(f.name));
    if (!g) throw new Error('No sliced plate in that 3mf. Slice it first, then export the plate as a sliced file (.gcode.3mf).');
    if (files.filter(f => /^Metadata\/plate_\d+\.gcode$/.test(f.name)).length > 1)
      throw new Error('That 3mf has more than one sliced plate. Put the blank on its own plate.');
    return { files, gname: g.name, text: new TextDecoder().decode(g.data) };
  }
  async function repack(pkg, text, seconds) {
    const enc = new TextEncoder(), gbytes = enc.encode(text);
    const files = pkg.files.map(f => {
      if (f.name === pkg.gname) return { name: f.name, data: gbytes };
      if (f.name === pkg.gname + '.md5') return { name: f.name, data: enc.encode(md5(gbytes)) };
      if (f.name === 'Metadata/slice_info.config' && seconds) {
        const s = new TextDecoder().decode(f.data).replace(/(<metadata key="prediction" value=")\d+("\/>)/, `$1${Math.round(seconds)}$2`);
        return { name: f.name, data: enc.encode(s) };
      }
      return f;
    });
    return zipDeflated(files);
  }

  const ZP = { md5, unzip, openPackage, repack, BASE, H_L, N, PITCH, PPM, BEAD, curve, Yof, interp, fmap, plate, families, order, resample, pieces, block,
               layers, stack, stepBack, lineView, boxFilter, ironing, inspect, check, rewrite, ourSeconds, blank3mf, zip, crc32, resize, gaussian };
  if (typeof module !== 'undefined' && module.exports) module.exports = ZP; else root.ZP = ZP;
})(typeof window !== 'undefined' ? window : globalThis);
