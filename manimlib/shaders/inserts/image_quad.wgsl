/*
The vertex stage shared by everything drawn as four corners with a texture stretched over
them: it does no shaping at all, reading the corners and handing on where in the image each
of them sits. What differs between such shaders is only how the fragment stage reads.
*/
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clip_distances: vec4f,
    @location(1) im_coords: vec2f,
    @location(2) opacity: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    if (index >= VERTS_PER_QUAD) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let corner = quad_corner(index);

    let projection = project_point(read_vec3(corner, DATA_OFFSET_point));
    out.position = projection.position;
    out.clip_distances = projection.clip_distances;
    out.im_coords = read_vec2(corner, DATA_OFFSET_im_coords);
    out.opacity = read_float(corner, DATA_OFFSET_opacity);
    return out;
}
