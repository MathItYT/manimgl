/*
A video is drawn exactly as an image is, four corners with a texture stretched over them,
except that the texture holds every frame of the clip at once and the mobject says which of
them to read, see mobject/types/video_mobject.py.
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
    // A preloaded clip has a layer per frame; one read a frame at a time has just the one,
    // whatever frame the mobject says it is showing. A blend landing between two frames
    // shows the nearer, and one past either end comes round again, which is how a clip set
    // to loop reads its first frame back, all as VideoMobject.frame_index reads it too
    let count = f32(textureNumLayers(Texture));
    var index = round(mob.frame) % count;
    if (index < 0.0) {
        index = index + count;
    }
    let layer = u32(index);
    var color = textureSample(Texture, image_sampler, in.im_coords, layer);
    color.a *= in.opacity;
    return color;
}
