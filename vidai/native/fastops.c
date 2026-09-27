/* VidAI hot loops in C. Built on first use (see __init__.py) and called through ctypes.
 * Everything is plain C99 + optional OpenMP; no Python headers needed. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* RMS level in dBFS per hop of mono float samples. out has n/hop entries. */
void rms_db(const float *pcm, int64_t n, int32_t hop, float *out) {
    int64_t m = n / hop;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; i++) {
        const float *p = pcm + i * hop;
        double acc = 0.0;
        for (int32_t j = 0; j < hop; j++) acc += (double)p[j] * p[j];
        double v = 20.0 * log10(sqrt(acc / hop + 1e-12) + 1e-9);
        out[i] = (float)(v < -90.0 ? -90.0 : (v > 0.0 ? 0.0 : v));
    }
}

/* Runs where x < thr (below=1) or x >= thr (below=0) lasting >= min_len samples.
 * Writes [start, end) index pairs; returns the number of runs (capped at max_runs). */
int64_t find_runs(const float *x, int64_t n, float thr, int32_t below, int64_t min_len,
                  int64_t *starts, int64_t *ends, int64_t max_runs) {
    int64_t k = 0, i = 0;
    while (i < n && k < max_runs) {
        int in = below ? (x[i] < thr) : (x[i] >= thr);
        if (!in) { i++; continue; }
        int64_t j = i;
        while (j < n && (below ? (x[j] < thr) : (x[j] >= thr))) j++;
        if (j - i >= min_len) { starts[k] = i; ends[k] = j; k++; }
        i = j;
    }
    return k;
}

/* Mean absolute difference between consecutive uint8 frames (0..1). out[0] = 0. */
void frame_mad(const uint8_t *frames, int64_t nframes, int64_t size, float *out) {
    if (nframes > 0) out[0] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int64_t f = 1; f < nframes; f++) {
        const uint8_t *a = frames + (f - 1) * size, *b = frames + f * size;
        uint64_t acc = 0;
        for (int64_t i = 0; i < size; i++) acc += (uint64_t)abs((int)a[i] - (int)b[i]);
        out[f] = (float)((double)acc / (double)size / 255.0);
    }
}

static inline uint8_t clamp8(float v) { return v <= 0.f ? 0 : (v >= 255.f ? 255 : (uint8_t)(v + 0.5f)); }

/* Per-pixel color model (ColorMatch): features [r g b (r^2 g^2 b^2 if degree 2) 1] @ W (nfeat x 3),
 * on RGB uint8 pixels scaled to 0..1, blended with the input by `strength`.
 * Every feature depends on one input channel, so out_c = T[c][0][r] + T[c][1][g] + T[c][2][b]:
 * the 3x3x256 tables are built once and each pixel costs 9 lookups. */
void affine_color(const uint8_t *in, uint8_t *out, int64_t npix, const float *W, int32_t degree, float strength) {
    const int sq = degree >= 2;
    const int bias_row = sq ? 6 : 3;
    float T[3][3][256];
    for (int c = 0; c < 3; c++)
        for (int ch = 0; ch < 3; ch++)
            for (int v = 0; v < 256; v++) {
                float x = v / 255.f;
                float y = x * W[ch * 3 + c] + (sq ? x * x * W[(3 + ch) * 3 + c] : 0.f);
                if (ch == 0) y += W[bias_row * 3 + c];
                T[c][ch][v] = y * 255.f * strength + (ch == c ? v * (1.f - strength) : 0.f);
            }
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < npix; p++) {
        const uint8_t r = in[p * 3], g = in[p * 3 + 1], b = in[p * 3 + 2];
        out[p * 3] = clamp8(T[0][0][r] + T[0][1][g] + T[0][2][b]);
        out[p * 3 + 1] = clamp8(T[1][0][r] + T[1][1][g] + T[1][2][b]);
        out[p * 3 + 2] = clamp8(T[2][0][r] + T[2][1][g] + T[2][2][b]);
    }
}

/* Alpha-blend an RGBA overlay (ow x oh) onto an RGB frame (fw x fh) at (x, y), in place.
 * `opacity` (0..1) scales the overlay alpha. Parts outside the frame are clipped. */
void alpha_blend(uint8_t *frame, int32_t fw, int32_t fh, const uint8_t *ov, int32_t ow, int32_t oh,
                 int32_t x, int32_t y, float opacity) {
    int32_t x0 = x < 0 ? -x : 0, y0 = y < 0 ? -y : 0;
    int32_t x1 = (x + ow > fw) ? fw - x : ow, y1 = (y + oh > fh) ? fh - y : oh;
    if (x0 >= x1 || y0 >= y1 || opacity <= 0.f) return;
    const int32_t op = (int32_t)(opacity * 256.f + 0.5f);
    #pragma omp parallel for schedule(static) if ((y1 - y0) * (x1 - x0) > 200000)
    for (int32_t j = y0; j < y1; j++) {
        const uint8_t *s = ov + ((int64_t)j * ow + x0) * 4;
        uint8_t *d = frame + ((int64_t)(y + j) * fw + x + x0) * 3;
        for (int32_t i = x0; i < x1; i++, s += 4, d += 3) {
            int32_t a = (s[3] * op) >> 8;
            if (a == 0) continue;
            if (a >= 255) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; continue; }
            d[0] = (uint8_t)((s[0] * a + d[0] * (255 - a) + 127) / 255);
            d[1] = (uint8_t)((s[1] * a + d[1] * (255 - a) + 127) / 255);
            d[2] = (uint8_t)((s[2] * a + d[2] * (255 - a) + 127) / 255);
        }
    }
}
