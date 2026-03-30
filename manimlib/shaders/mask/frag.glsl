#version 330

in vec3 v_point;

uniform sampler2D Source;
uniform sampler2D Mask;

out vec4 frag_color;

void main() {
    vec4 source_color = texture(Source, v_point.xy * 0.5 + 0.5);
    vec4 mask_color = texture(Mask, v_point.xy * 0.5 + 0.5);
    frag_color = vec4(source_color.rgb, source_color.a * mask_color.a);
}