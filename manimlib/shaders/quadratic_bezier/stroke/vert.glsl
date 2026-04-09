#version 330

uniform float frame_scale;
uniform float is_fixed_in_frame;
uniform float scale_stroke_with_zoom;

in vec3 point;
in vec4 stroke_rgba;
in float stroke_width;
in float joint_angle;
in vec3 unit_normal;

// Bezier control point
out vec3 verts;

out vec4 v_color;
out float v_stroke_width;
out float v_joint_angle;
out vec3 v_unit_normal;

const float STROKE_WIDTH_CONVERSION = 0.01;

void main(){
    verts = point;
    v_color = stroke_rgba;
    float zoom_factor = mix(frame_scale, 1.0, scale_stroke_with_zoom);
    // Fixed-in-frame objects are already in frame coordinates (no view scaling),
    // so they must not scale stroke width with camera zoom.
    zoom_factor = mix(zoom_factor, 1.0, is_fixed_in_frame);
    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width * zoom_factor;
    v_joint_angle = joint_angle;
    v_unit_normal = unit_normal;
}