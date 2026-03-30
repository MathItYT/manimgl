#version 330

uniform float opacity;
uniform float scale_factor;

in vec3 xyz_coords;
out vec4 frag_color;

#INSERT finalize_color.glsl
#INSERT complex_functions.glsl

// ── HSV → RGB ──
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ── Complex exponential ──
vec2 complex_exp(vec2 z) {
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

// ── Complex sine ──
vec2 complex_sin(vec2 z) {
    // sin(z) = (e^(iz) - e^(-iz)) / (2i)
    vec2 iz = vec2(-z.y, z.x);
    vec2 niz = vec2(z.y, -z.x);
    vec2 e1 = complex_exp(iz);
    vec2 e2 = complex_exp(niz);
    vec2 diff = e1 - e2;
    // divide by 2i: (a+bi)/(2i) = (b - ai)/2 → (b/2, -a/2)
    return vec2(diff.y, -diff.x) / 2.0;
}

// ── The complex function to visualize ──
vec2 f(vec2 z) {
    // f(z) = sin(z) — visually rich with periodic structure
    return complex_sin(z);
}

// ── Smooth magnitude mapping ──
float magnitude_shade(float r) {
    // Creates contour lines at powers of 2
    float logr = log2(max(r, 1e-10));
    float frac_part = fract(logr);
    // Smooth shading between contours
    return 0.45 + 0.55 * (0.5 + 0.5 * cos(6.28318 * frac_part));
}

// ── Grid lines (darken near integer real/imag parts) ──
float grid_factor(vec2 w) {
    vec2 grid = abs(fract(w + 0.5) - 0.5);
    float line = min(grid.x, grid.y);
    return smoothstep(0.0, 0.05, line);
}

void main() {
    vec2 z = xyz_coords.xy;
    vec2 w = f(z);

    // Argument → Hue (normalized to [0,1])
    float arg = atan(w.y, w.x); // range [-pi, pi]
    float hue = arg / 6.28318 + 0.5; // normalize to [0,1]

    // Magnitude → Value (brightness)
    float mag = length(w);
    float value = magnitude_shade(mag);

    // Grid lines for structure
    float grid = grid_factor(w);
    value *= mix(0.6, 1.0, grid);

    // Saturation: full for most, reduce near zero/infinity
    float sat = 0.75 + 0.25 * (1.0 - 1.0 / (1.0 + mag));

    vec3 rgb = hsv2rgb(vec3(hue, sat, value));

    frag_color = finalize_color(
        vec4(rgb, opacity),
        xyz_coords,
        vec3(0.0, 0.0, 1.0)
    );
}
