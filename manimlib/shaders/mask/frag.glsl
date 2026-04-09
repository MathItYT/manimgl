#version 330

in vec3 v_point;

uniform sampler2D Source;
uniform sampler2D Mask;

out vec4 frag_color;

void main() {
    vec4 source_color = texture(Source, v_point.xy * 0.5 + 0.5);
    vec4 mask_color = texture(Mask, v_point.xy * 0.5 + 0.5);
    // The source texture is rendered onto a transparent framebuffer with
    // blending enabled, which leaves it in premultiplied-alpha form.
    // If we output that directly while using the default (straight-alpha)
    // blend function, alpha gets applied twice and transparent sources
    // look too dark. Convert back to straight alpha here.
    float a = source_color.a;
    vec3 rgb = (a > 0.0) ? (source_color.rgb / a) : vec3(0.0);
    float out_a = a * mask_color.a;
    frag_color = vec4(rgb, out_a);
}