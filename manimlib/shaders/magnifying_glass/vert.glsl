#version 330

in vec3 point;

out vec3 v_point;

#INSERT emit_gl_Position.glsl

void main(){
    emit_gl_Position(point);
    v_point = gl_Position.xyz;
}
