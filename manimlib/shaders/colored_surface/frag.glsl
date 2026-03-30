#version 330

uniform vec4 u_color_light;
uniform vec4 u_color_dark;

// NOTA: No declaramos light_position aquí porque viene dentro del INSERT
// uniform vec3 light_position; 

in vec3 v_point;
in vec3 v_unit_normal;
in vec4 v_color; // Color/Opacidad base del objeto

out vec4 frag_color;

#INSERT finalize_color.glsl

const float dark_shift = 0.2;

void main() {
    // light_position está disponible gracias al INSERT de arriba
    float dp = dot(
        normalize(light_position - v_point),
        v_unit_normal
    );
    
    float alpha = smoothstep(-dark_shift, dark_shift, dp);
    
    // Mezcla de colores para el efecto toon
    vec4 mixed_color = mix(u_color_dark, u_color_light, alpha);
    
    // Si la opacidad calculada es 0, descartamos
    if (mixed_color.a == 0) discard;

    // IMPORTANTE: finalize_color aplica shading estándar.
    // Como calculamos nuestra propia luz, en Python haremos set_shading(0,0,0)
    // para que esta función solo procese la salida sin alterar la luz.
    frag_color = finalize_color(
        mixed_color,
        v_point,
        v_unit_normal
    );
    
    // Combinar con la opacidad del objeto (v_color.a)
    frag_color.a *= v_color.a;
}