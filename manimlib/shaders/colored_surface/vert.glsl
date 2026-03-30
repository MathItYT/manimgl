#version 330
in vec3 point;
in vec3 d_normal_point;
in vec4 rgba;

out vec3 v_point;
out vec3 v_unit_normal;
out vec4 v_color;

#INSERT emit_gl_Position.glsl
#INSERT get_unit_normal.glsl

void main(){
    v_point = point;
    v_unit_normal = normalize(d_normal_point - point);
    v_color = rgba;
    emit_gl_Position(point);
}