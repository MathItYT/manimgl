/*
Depth-only version of fill.wgsl.

This shader deliberately keeps the exact same geometric construction and
fragment clipping as fill.wgsl.

The important difference is not in the fragment output: the pipeline using
this shader has color_write=false.

Therefore successful fragments update ONLY the depth/stencil attachment.
*/

#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT fill_color.wgsl
#INSERT finalize_color.wgsl
#INSERT clip_test.wgsl

const VERTS_PER_CURVE: u32 = 6u;
const RECORD_STEP: u32 = 2u;

const SIMPLE_QUADRATIC = array<vec2f, 3>(
    vec2f(0.0, 0.0),
    vec2f(0.5, 0.0),
    vec2f(1.0, 1.0),
);

struct VertexOutput {
    @builtin(position) position: vec4f,

    @location(0)
    clip_distances: vec4f,

    @location(1)
    color: vec4f,

    @location(2)
    fill_all: f32,

    @location(3)
    uv_coords: vec2f,
}


@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let curve = index / VERTS_PER_CURVE;
    let corner = index % VERTS_PER_CURVE;

    let record = RECORD_STEP * curve;

    let controls = array<vec3f, 3>(
        read_vec3(
            record,
            DATA_OFFSET_point,
        ),

        read_vec3(
            record + 1u,
            DATA_OFFSET_point,
        ),

        read_vec3(
            record + 2u,
            DATA_OFFSET_point,
        ),
    );

    /*
    Degenerate/empty curves must produce no rasterized fragments.
    */
    if (
        all(
            controls[0] == controls[1]
        )
        ||
        max(
            mob.fill_rgba.a,
            mob.fill_rgba_end.a,
        ) == 0.0
    ) {
        out.position = vec4f(
            0.0,
            0.0,
            0.0,
            1.0,
        );

        return out;
    }

    let subpath = read_vec2(
        record,
        DATA_OFFSET_subpath_range,
    );

    let base_point = read_vec3(
        u32(
            i32(record)
            - i32(subpath.x)
        ),
        DATA_OFFSET_point,
    );

    let corner_index = corner % 3u;

    var point: vec3f;

    var fan = array<vec3f, 3>(
        base_point,
        controls[0],
        controls[2],
    );

    if (corner < 3u) {
        /*
        Interior fan triangle.
        */
        out.fill_all = 1.0;

        point = fan[corner_index];
    } else {
        /*
        Triangle hugging the bezier.
        */
        out.fill_all = 0.0;

        point = controls[corner_index];
    }

    let uv = SIMPLE_QUADRATIC;

    out.uv_coords = uv[corner_index];

    /*
    We still need a valid color calculation because finalize_color can
    contain transformations needed by the shader's normal/color path.
    The pipeline will discard the color output anyway.
    */
    out.color = finalize_color(
        fill_color_at(point),
        point,
        mob.unit_normal,
    );

    let projection = project_point(point);

    out.position = projection.position;
    out.clip_distances = projection.clip_distances;

    return out;
}


@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4f {
    clip_test(
        in.clip_distances
    );

    /*
    Invisible fill must not reserve depth.
    */
    if (
        in.color.a == 0.0
    ) {
        discard;
    }

    /*
    The second triangle around every bezier is larger than the actual
    curve region. Keep exactly the same quadratic clipping used by the
    visible fill shader.
    */
    if (
        in.fill_all == 0.0
        &&
        in.uv_coords.y
            < in.uv_coords.x
            * in.uv_coords.x
    ) {
        discard;
    }

    /*
    The actual color is irrelevant because color writes are disabled.
    Returning it is still required by the fragment-stage interface.
    */
    return in.color;
}
