/*
Browser video uses a single 2D GPU texture updated directly from the HTMLVideoElement.
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
