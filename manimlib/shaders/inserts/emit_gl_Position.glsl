uniform float is_fixed_in_frame;
uniform mat4 view;
uniform float focal_distance;
uniform vec3 frame_rescale_factors;
// Optional magnifier (camera-uniform driven) post-projection transform
uniform float magnify_active;
uniform vec2 magnify_center;
uniform float magnify_zoom;
uniform vec4 clip_plane0;
uniform vec4 clip_plane1;
uniform vec4 clip_plane2;
uniform vec4 clip_plane3;

void emit_gl_Position(vec3 point){
    vec4 result = vec4(point, 1.0);
    // This allows for smooth transitions between objects fixed and unfixed from frame
    result = mix(view * result, result, is_fixed_in_frame);
    // Essentially a projection matrix
    result.xyz *= frame_rescale_factors;
    result.w = 1.0 - result.z;
    // Flip and scale to prevent premature clipping
    result.z *= -0.1;
    gl_Position = result;

    // Apply optional magnifier transform in normalized device coordinates.
    // This is used to re-render the scene with a "zoomed" camera without
    // rasterizing a texture, preserving vector detail.
    if(magnify_active > 0.5){
        float w = max(gl_Position.w, 1e-8);
        vec2 ndc = gl_Position.xy / w;
        float zoom = max(magnify_zoom, 1e-6);
        ndc = magnify_center + (ndc - magnify_center) * zoom;
        gl_Position.xy = ndc * w;
    }
    
    // Set clip planes
    if(clip_plane0.xyz != vec3(0.0, 0.0, 0.0)){
        gl_ClipDistance[0] = dot(vec4(point, 1.0), clip_plane0);
    }
    if(clip_plane1.xyz != vec3(0.0, 0.0, 0.0)){
        gl_ClipDistance[1] = dot(vec4(point, 1.0), clip_plane1);
    }
    if(clip_plane2.xyz != vec3(0.0, 0.0, 0.0)){
        gl_ClipDistance[2] = dot(vec4(point, 1.0), clip_plane2);
    }
    if(clip_plane3.xyz != vec3(0.0, 0.0, 0.0)){
        gl_ClipDistance[3] = dot(vec4(point, 1.0), clip_plane3);
    }
}
