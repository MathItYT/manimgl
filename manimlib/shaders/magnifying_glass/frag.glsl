#version 330

in vec3 v_point;

uniform sampler2D Scene;
uniform sampler2D Zoomed;

// Magnifier parameters (world/frame coords depending on is_fixed_in_frame)
uniform vec3 lens_center;
uniform float lens_radius;
uniform float magnification;
uniform float rasterize;

// Camera uniforms (see inserts/emit_gl_Position.glsl)
uniform float is_fixed_in_frame;
uniform mat4 view;
uniform vec3 frame_rescale_factors;
uniform float pixel_size;
uniform float frame_scale;

out vec4 frag_color;

vec2 point_to_uv(vec3 point){
    vec4 result = vec4(point, 1.0);
    result = mix(view * result, result, is_fixed_in_frame);
    result.xyz *= frame_rescale_factors;
    float w = 1.0 - result.z;
    vec2 ndc = result.xy / max(w, 1e-8);
    return ndc * 0.5 + 0.5;
}

void main(){
    vec2 uv = v_point.xy * 0.5 + 0.5;
    vec2 center_uv = point_to_uv(lens_center);

    // Convert uv delta to frame-space delta so radius is measured in scene units.
    vec2 uv_to_frame = vec2(
        2.0 / frame_rescale_factors.x,
        2.0 / frame_rescale_factors.y
    );
    vec2 delta_frame = (uv - center_uv) * uv_to_frame;
    float dist = length(delta_frame);

    // Anti-aliased circular edge.
    // `pixel_size` grows when the camera zooms out, but fixed-in-frame objects
    // are rendered in frame coordinates; use a base pixel size in that case.
    float base_pixel_size = pixel_size / max(frame_scale, 1e-8);
    float effective_pixel_size = mix(pixel_size, base_pixel_size, is_fixed_in_frame);
    float aa = max(1.5 * effective_pixel_size, 1e-8);
    float edge_alpha = 1.0 - smoothstep(lens_radius - aa, lens_radius, dist);

    // Only draw inside the circular lens (with anti-aliased edge).
    if(edge_alpha <= 0.0) discard;

    // Rasterized mode (legacy): sample from a captured scene texture.
    if(rasterize > 0.5){
        float zoom = max(magnification, 1e-6);
        vec2 magnified_uv = center_uv + (uv - center_uv) / zoom;
        frag_color = texture(Scene, magnified_uv);
        frag_color.a *= edge_alpha;
        return;
    }

    // Non-rasterized mode: sample from an offscreen re-render of the scene
    // with modified camera uniforms (preserves vector detail).
    frag_color = texture(Zoomed, uv);
    frag_color.a *= edge_alpha;
}
