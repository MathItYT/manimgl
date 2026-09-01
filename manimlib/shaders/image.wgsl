/*
An image is four corners with a texture stretched over them, so unlike everything else here
the shader does no shaping at all: it reads the corners, and reads the image at each point.
*/
#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT quad_corners.wgsl
#INSERT clip_test.wgsl

// TEXTURES

#INSERT image_quad.wgsl

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    clip_test(in.clip_distances);
    var color = textureSample(Texture, image_sampler, in.im_coords);
    color.a *= in.opacity;
    return color;
}
