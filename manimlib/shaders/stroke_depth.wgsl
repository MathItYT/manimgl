/*
Depth-only stroke shader.

This is the depth counterpart of stroke.wgsl.

The generated geometry is identical. The fragment stage only keeps the
portion that belongs to the actual stroke.

The render pipeline using this shader has:

    color_write = false
    depth_write = true

so the returned color is never stored.
*/

const IS_FILL_BORDER: bool = false;

#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT fill_color.wgsl
#INSERT finalize_color.wgsl
#INSERT clip_test.wgsl

const ALIGNMENT_THRESHOLD: f32 = 0.99;
const POLYLINE_FACTOR: f32 = 100.0;
const MAX_STEPS: i32 = 32;
const NONE: i32 = -1;
const EPSILON: f32 = 1e-7;

const ROUND_COS_SHARP: f32 = -0.5;
const ROUND_COS_ROUND: f32 = -0.95;

const STROKE_WIDTH_CONVERSION: f32 = 0.01;

const RECORD_STEP: u32 = 2u;

const VERTS_PER_CURVE: u32 =
    u32(6 * (MAX_STEPS - 1));

const JOINT_SEGMENTS: i32 = 3;

const POLYLINE_SEGMENTS: i32 =
    MAX_STEPS - 1 - JOINT_SEGMENTS;

const FAN_TRIANGLES: i32 =
    2 * JOINT_SEGMENTS;


const CORNERS = array<vec2f, 6>(
    vec2f(0.0, -1.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, -1.0),

    vec2f(1.0, -1.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
);


struct VertexOutput {
    @builtin(position) position: vec4f,

    @location(0)
    clip_distances: vec4f,

    @location(1)
    color: vec4f,

    @location(2)
    dist_to_aaw: f32,

    @location(3)
    half_width_to_aaw: f32,
}


fn point_on_quadratic(
    t: f32,
    c0: vec3f,
    c1: vec3f,
    c2: vec3f,
) -> vec3f {
    return c0
        + c1 * t
        + c2 * t * t;
}


fn tangent_on_quadratic(
    t: f32,
    c1: vec3f,
    c2: vec3f,
) -> vec3f {
    return c1
        + 2.0 * c2 * t;
}


fn project(
    vect: vec3f,
    normal: vec3f,
) -> vec3f {
    return vect
        - dot(vect, normal) * normal;
}


fn rotate_vector(
    vect: vec3f,
    normal: vec3f,
    turn: vec2f,
) -> vec3f {
    return turn.x * vect
        + turn.y * cross(normal, vect);
}


fn neighbor_tangent(
    record: i32,
    subpath: vec2f,
    at_start: bool,
    anchor: vec3f,
) -> vec3f {
    let first =
        record - i32(subpath.x);

    let last =
        record + i32(subpath.y);

    let closed = all(
        read_vec3(
            u32(first),
            DATA_OFFSET_point,
        )
        ==
        read_vec3(
            u32(last),
            DATA_OFFSET_point,
        )
    );

    if (at_start) {
        var previous = NONE;

        if (record > first) {
            previous = record - 1;
        }
        else if (closed) {
            previous = last - 1;
        }
        else if (IS_FILL_BORDER) {
            previous = last;
        }

        if (previous == NONE) {
            return vec3f(0.0);
        }

        return anchor
            - read_vec3(
                u32(previous),
                DATA_OFFSET_point,
            );
    }

    var next = NONE;

    if (record + 2 < last) {
        next = record + 3;
    }
    else if (closed || record == last) {
        next = first + 1;
    }
    else if (IS_FILL_BORDER) {
        next = first;
    }

    if (next == NONE) {
        return vec3f(0.0);
    }

    return read_vec3(
        u32(next),
        DATA_OFFSET_point,
    ) - anchor;
}


fn vector_length_squared(v: vec3f) -> f32 {
    return dot(v, v);
}


fn is_degenerate(v: vec3f) -> bool {
    return vector_length_squared(v) <= EPSILON * EPSILON;
}


fn safe_normalize(v: vec3f, fallback: vec3f) -> vec3f {
    let len2 = dot(v, v);
    if (len2 <= EPSILON * EPSILON) {
        let fallback_len2 = dot(fallback, fallback);
        if (fallback_len2 <= EPSILON * EPSILON) {
            return vec3f(0.0, 0.0, 0.0);
        }
        return fallback * inverseSqrt(fallback_len2);
    }
    return v * inverseSqrt(len2);
}


fn flat_tangent(
    tangent: vec3f,
    facing_normal: vec3f,
) -> vec3f {
    let flattened = project(tangent, facing_normal);

    if (is_degenerate(flattened)) {
        return vec3f(0.0);
    }

    return safe_normalize(flattened, vec3f(0.0));
}


fn safe_normalize_2d(v: vec2f, fallback: vec2f) -> vec2f {
    let len2 = dot(v, v);
    if (len2 <= EPSILON * EPSILON) {
        let fallback_len2 = dot(fallback, fallback);
        if (fallback_len2 <= EPSILON * EPSILON) {
            return vec2f(0.0, 0.0);
        }
        return fallback * inverseSqrt(fallback_len2);
    }
    return v * inverseSqrt(len2);
}


fn get_camera_basis() -> array<vec3f, 2> {
    var R = vec3f(frame.view[0].x, frame.view[1].x, frame.view[2].x);
    var U = vec3f(frame.view[0].y, frame.view[1].y, frame.view[2].y);
    if (mob.is_fixed_in_frame != 0.0) {
        R = vec3f(1.0, 0.0, 0.0);
        U = vec3f(0.0, 1.0, 0.0);
    } else {
        R = safe_normalize(R, vec3f(1.0, 0.0, 0.0));
        U = safe_normalize(U, vec3f(0.0, 1.0, 0.0));
    }
    return array<vec3f, 2>(R, U);
}


fn get_camera_normal() -> vec3f {
    if (mob.is_fixed_in_frame != 0.0) {
        return vec3f(0.0, 0.0, 1.0);
    }
    let N = vec3f(frame.view[0].z, frame.view[1].z, frame.view[2].z);
    return safe_normalize(N, vec3f(0.0, 0.0, 1.0));
}


fn get_screen_tangent(point: vec3f, tangent: vec3f) -> vec2f {
    var p_view = (frame.view * vec4f(point, 1.0)).xyz;
    var t_view = (frame.view * vec4f(tangent, 0.0)).xyz;
    if (mob.is_fixed_in_frame != 0.0) {
        p_view = point;
        t_view = tangent;
    }

    let w = 1.0 - p_view.z * frame.frame_rescale_factors.z;
    let screen_tan = t_view.xy * w + p_view.xy * (t_view.z * frame.frame_rescale_factors.z);

    if (dot(screen_tan, screen_tan) > EPSILON * EPSILON) {
        return safe_normalize_2d(screen_tan, vec2f(1.0, 0.0));
    }
    if (dot(t_view.xy, t_view.xy) > EPSILON * EPSILON) {
        return safe_normalize_2d(t_view.xy, vec2f(1.0, 0.0));
    }
    return vec2f(1.0, 0.0);
}


fn joint_shift(
    point: vec3f,
    tan_in: vec3f,
    tan_out: vec3f,
    facing_normal: vec3f,
    draw_flat: bool,
) -> f32 {
    if (!draw_flat) {
        let a = get_screen_tangent(point, tan_in);
        let b = get_screen_tangent(point, tan_out);

        let sin_angle = a.x * b.y - a.y * b.x;

        if (abs(sin_angle) < 1e-6) {
            return 0.0;
        }

        let cos_angle = clamp(
            dot(a, b),
            -1.0,
            1.0,
        );

        let keep = smoothstep(
            ROUND_COS_ROUND,
            ROUND_COS_SHARP,
            cos_angle,
        ) * (1.0 - mob.joint_roundness);

        return keep * (cos_angle - 1.0) / sin_angle;
    }

    let a = flat_tangent(
        tan_in,
        facing_normal,
    );

    let b = flat_tangent(
        tan_out,
        facing_normal,
    );

    if (
        all(a == vec3f(0.0))
        ||
        all(b == vec3f(0.0))
    ) {
        return 0.0;
    }

    let sin_angle =
        dot(
            cross(a, b),
            facing_normal,
        );

    if (
        abs(sin_angle)
        < 1e-6
    ) {
        return 0.0;
    }

    let cos_angle =
        dot(a, b);

    let keep =
        smoothstep(
            ROUND_COS_ROUND,
            ROUND_COS_SHARP,
            cos_angle,
        )
        * (
            1.0
            - mob.joint_roundness
        );

    return keep
        * (cos_angle - 1.0)
        / sin_angle;
}


fn step_to_corner(
    point: vec3f,
    tangent: vec3f,
    facing_normal: vec3f,
    shift: f32,
    draw_flat: bool,
) -> vec3f {
    if (!draw_flat) {
        let basis = get_camera_basis();
        let R = basis[0];
        let U = basis[1];
        let tan_2d = get_screen_tangent(point, tangent);
        let step_2d = vec2f(-tan_2d.y, tan_2d.x);
        return (step_2d.x + shift * tan_2d.x) * R + (step_2d.y + shift * tan_2d.y) * U;
    }

    var unflattened = tangent;

    let unit_tan =
        normalize(
            unflattened
        );

    var step =
        normalize(
            cross(
                facing_normal,
                unit_tan,
            )
        );

    let alignment =
        abs(
            dot(
                normalize(tangent),
                facing_normal,
            )
        );

    if (
        alignment
        > ALIGNMENT_THRESHOLD
    ) {
        let perp =
            normalize(
                cross(
                    mob.unit_normal,
                    tangent,
                )
            );

        step = mix(
            step,
            project(
                step,
                perp,
            ),
            smoothstep(
                ALIGNMENT_THRESHOLD,
                1.0,
                alignment,
            ),
        );
    }

    return step
        + shift * unit_tan;
}


@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let blank =
        vec4f(
            0.0,
            0.0,
            0.0,
            1.0,
        );

    let curve =
        vertex_index
        / VERTS_PER_CURVE;

    let within =
        vertex_index
        % VERTS_PER_CURVE;

    let segment =
        i32(within / 6u);

    let tri_vert =
        i32(within % 6u);

    let corner =
        CORNERS[tri_vert];

    let record =
        RECORD_STEP * curve;

    let joint_fan =
        segment
        >= POLYLINE_SEGMENTS;

    let subpath =
        read_vec2(
            record,
            DATA_OFFSET_subpath_range,
        );

    var controls =
        array<vec3f, 3>(
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

    if (
        IS_FILL_BORDER
        &&
        subpath.y == 0.0
    ) {
        controls[2] =
            read_vec3(
                u32(
                    i32(record)
                    - i32(subpath.x)
                ),
                DATA_OFFSET_point,
            );

        controls[1] =
            0.5
            * (
                controls[0]
                + controls[2]
            );
    }

    let widths =
        array<f32, 3>(
            read_float(
                record,
                DATA_OFFSET_stroke_width,
            ),

            read_float(
                record + 1u,
                DATA_OFFSET_stroke_width,
            ),

            read_float(
                record + 2u,
                DATA_OFFSET_stroke_width,
            ),
        );

    let colors =
        array<vec4f, 3>(
            read_vec4(
                record,
                DATA_OFFSET_stroke_rgba,
            ),

            read_vec4(
                record + 1u,
                DATA_OFFSET_stroke_rgba,
            ),

            read_vec4(
                record + 2u,
                DATA_OFFSET_stroke_rgba,
            ),
        );

    let c0 =
        controls[0];

    let c1 =
        2.0
        * (
            controls[1]
            - controls[0]
        );

    let c2 =
        controls[0]
        - 2.0 * controls[1]
        + controls[2];

    let area =
        0.5
        * length(
            cross(
                controls[1]
                - controls[0],

                controls[2]
                - controls[0],
            )
        );

    let count =
        i32(
            round(
                POLYLINE_FACTOR
                * sqrt(area)
                / get_frame_unit_size()
            )
        );

    let n_steps =
        min(
            2 + count,
            POLYLINE_SEGMENTS + 1,
        );

    var nothing_to_draw =
        all(
            controls[0]
            == controls[1]
        );

    nothing_to_draw =
        nothing_to_draw
        ||
        (
            !joint_fan
            &&
            segment >= n_steps - 1
        );

    if (IS_FILL_BORDER) {
        nothing_to_draw =
            nothing_to_draw
            ||
            min(
                mob.fill_rgba.a,
                mob.fill_rgba_end.a,
            ) < 0.999;
    }
    else {
        nothing_to_draw =
            nothing_to_draw
            ||
            all(
                vec3f(
                    widths[0],
                    widths[1],
                    widths[2],
                )
                == vec3f(0.0)
            )
            ||
            any(
                vec3f(
                    colors[0].a,
                    colors[1].a,
                    colors[2].a,
                )
                < vec3f(0.999)
            );
    }

    if (nothing_to_draw) {
        out.position = blank;
        return out;
    }

    let index =
        segment
        + i32(corner.x);

    var t = 1.0;

    if (!joint_fan) {
        t =
            f32(index)
            / f32(n_steps - 1);
    }

    var point =
        controls[2];

    if (!joint_fan) {
        point =
            point_on_quadratic(
                t,
                c0,
                c1,
                c2,
            );
    }

    let tangent =
        tangent_on_quadratic(
            t,
            c1,
            c2,
        );

    var own_width =
        mix(
            widths[0],
            widths[2],
            t,
        );

    if (IS_FILL_BORDER) {
        own_width =
            mob.fill_border_width;
    }

    let width =
        STROKE_WIDTH_CONVERSION
        * mix(
            get_frame_unit_size(),
            1.0,
            mob.stroke_width_in_scene_units,
        )
        * own_width;

    let draw_flat =
        mob.flat_stroke != 0.0
        ||
        mob.is_fixed_in_frame != 0.0;

    var facing_normal = get_camera_normal();

    if (draw_flat) {
        facing_normal = safe_normalize(
            mob.unit_normal,
            vec3f(0.0, 0.0, 1.0),
        );
    }

    var own_color =
        mix(
            colors[0],
            colors[2],
            t,
        );

    if (IS_FILL_BORDER) {
        own_color =
            fill_color_at(point);
    }

    out.color =
        finalize_color(
            own_color,
            point,
            facing_normal,
        );

    let aaw =
        max(
            mob.anti_alias_width
            * get_pixel_unit_size(),
            1e-8,
        );

    let half_width =
        0.5
        * (
            width
            + aaw
        );

    let at_start =
        !joint_fan
        &&
        index == 0;

    let at_joint =
        joint_fan
        ||
        at_start
        ||
        index == n_steps - 1;

    var neighbor =
        vec3f(0.0);

    if (at_joint) {
        neighbor =
            neighbor_tangent(
                i32(record),
                subpath,
                at_start,
                point,
            );
    }

    var has_joint =
        at_joint
        && !is_degenerate(neighbor);

    var shift = 0.0;

    if (has_joint) {
        if (at_start) {
            shift =
                -joint_shift(
                    point,
                    neighbor,
                    tangent,
                    facing_normal,
                    draw_flat,
                );
        }
        else {
            shift =
                joint_shift(
                    point,
                    tangent,
                    neighbor,
                    facing_normal,
                    draw_flat,
                );
        }
    }

    var step: vec3f;
    var dist_to_curve: f32;
    var edge_dist: f32;

    if (joint_fan) {
        if (!has_joint) {
            out.position = blank;
            return out;
        }

        if (!draw_flat) {
            let basis = get_camera_basis();
            let R = basis[0];
            let U = basis[1];

            let a = get_screen_tangent(point, tangent);
            let b = get_screen_tangent(point, neighbor);

            let perp_a = vec2f(-a.y, a.x);
            let perp_b = vec2f(-b.y, b.x);

            var outward = -1.0;
            if (a.x * b.y - a.y * b.x < 0.0) {
                outward = 1.0;
            }

            let edge_in_2d = outward * safe_normalize_2d(perp_a + shift * a, perp_a);
            let edge_out_2d = outward * safe_normalize_2d(perp_b - shift * b, perp_b);

            let sweep = atan2(
                edge_in_2d.x * edge_out_2d.y - edge_in_2d.y * edge_out_2d.x,
                dot(edge_in_2d, edge_out_2d),
            );

            if (abs(sweep) < 1e-5) {
                out.position = blank;
                return out;
            }

            let fan_tri =
                2 * (segment - POLYLINE_SEGMENTS) + tri_vert / 3;

            let fan_vert = tri_vert % 3;

            let along =
                f32(fan_tri + fan_vert - 1) / f32(FAN_TRIANGLES);

            let angle = along * sweep;
            let c = cos(angle);
            let s = sin(angle);
            let rotated_2d = vec2f(
                edge_in_2d.x * c - edge_in_2d.y * s,
                edge_in_2d.x * s + edge_in_2d.y * c,
            );

            step = rotated_2d.x * R + rotated_2d.y * U;

            edge_dist =
                sqrt(1.0 + shift * shift) * 0.5 * width;

            dist_to_curve = 0.0;

            if (fan_vert != 0) {
                dist_to_curve =
                    edge_dist + 0.5 * aaw;
            }
        }
        else {
            /*
            The fan normally joins the incoming and outgoing strips.

            If one of the tangents is degenerate, do NOT discard the whole fan.
            Instead construct a local round cap using the valid tangent.

            This is the important change for complex VMobjects/text.
            */
            var tan_in = flat_tangent(
                tangent,
                facing_normal,
            );

            var tan_out = flat_tangent(
                neighbor,
                facing_normal,
            );

            /*
            If the neighboring tangent is unavailable, use the current tangent.
            This turns the fan into a round cap instead of creating a hole.
            */
            if (is_degenerate(tan_out)) {
                tan_out = tan_in;
            }

            if (is_degenerate(tan_in)) {
                tan_in = tan_out;
            }

            /*
            If both are still unavailable, fall back to a stable direction derived from
            the object's normal. This prevents NaNs and, importantly, prevents an entire
            reserved fan from disappearing.
            */
            if (is_degenerate(tan_in) && is_degenerate(tan_out)) {
                let fallback_axis = safe_normalize(
                    cross(
                        facing_normal,
                        mob.unit_normal,
                    ),
                    vec3f(1.0, 0.0, 0.0),
                );

                tan_in = fallback_axis;
                tan_out = fallback_axis;

                has_joint = false;
            }

            var outward = -1.0;

            if (
                dot(
                    cross(
                        tan_in,
                        tan_out,
                    ),
                    facing_normal,
                ) < 0.0
            ) {
                outward = 1.0;
            }

            /*
            Construct the incoming and outgoing edge directions.

            Unlike the previous version, normalize() is never applied to a vector that
            has not first been checked for degeneracy.
            */
            var edge_in_raw =
                cross(
                    facing_normal,
                    tan_in,
                )
                + shift * tan_in;

            var edge_out_raw =
                cross(
                    facing_normal,
                    tan_out,
                )
                - shift * tan_out;

            var edge_in = safe_normalize(
                edge_in_raw,
                cross(
                    facing_normal,
                    tan_in,
                ),
            );

            var edge_out = safe_normalize(
                edge_out_raw,
                cross(
                    facing_normal,
                    tan_out,
                ),
            );

            /*
            A completely degenerate camera/normal configuration should still produce
            finite geometry.
            */
            if (is_degenerate(edge_in)) {
                edge_in = safe_normalize(
                    cross(
                        mob.unit_normal,
                        tan_in,
                    ),
                    vec3f(1.0, 0.0, 0.0),
                );
            }

            if (is_degenerate(edge_out)) {
                edge_out = safe_normalize(
                    cross(
                        mob.unit_normal,
                        tan_out,
                    ),
                    edge_in,
                );
            }

            edge_in *= outward;
            edge_out *= outward;

            let sweep = atan2(
                dot(
                    cross(
                        edge_in,
                        edge_out,
                    ),
                    facing_normal,
                ),
                dot(
                    edge_in,
                    edge_out,
                ),
            );

            /*
            Of each triangle's three vertices, one sits at the joint and two on the arc.
            */
            let fan_tri =
                2 * (
                    segment - POLYLINE_SEGMENTS
                )
                + tri_vert / 3;

            let fan_vert = tri_vert % 3;

            let along =
                f32(
                    fan_tri
                    + fan_vert
                    - 1
                )
                / f32(FAN_TRIANGLES);

            step = rotate_vector(
                edge_in,
                facing_normal,
                vec2f(
                    cos(along * sweep),
                    sin(along * sweep),
                ),
            );

            /*
            The corners reach out by this much, being a step out plus a shift along.
            */
            edge_dist = sqrt(
                max(
                    1.0 + shift * shift,
                    0.0,
                )
            ) * 0.5 * width;

            dist_to_curve = 0.0;

            if (fan_vert != 0) {
                dist_to_curve =
                    edge_dist
                    + 0.5 * aaw;
            }
        }
    }
    else {
        step =
            step_to_corner(
                point,
                tangent,
                facing_normal,
                shift,
                draw_flat,
            );

        edge_dist =
            0.5 * width;

        dist_to_curve =
            corner.y
            * half_width;
    }

    out.half_width_to_aaw =
        edge_dist / aaw;

    out.dist_to_aaw =
        dist_to_curve / aaw;

    let projection =
        project_point(
            point
            + dist_to_curve * step,
        );

    out.position =
        projection.position;

    out.clip_distances =
        projection.clip_distances;

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
    Reconstruct the exact signed distance used by the visible stroke.
    */
    let signed_dist_to_region =
        abs(in.dist_to_aaw)
        - in.half_width_to_aaw;

    /*
    Unlike the normal stroke shader, the anti-alias fringe must NOT receive
    depth. Otherwise the fringe itself would occlude geometry behind it.

    Keep the interior and the geometric edge.
    */
    if (
        signed_dist_to_region > 0.0
        ||
        in.color.a < 0.999
    ) {
        discard;
    }

    /*
    For the depth-only pipeline the color value is ignored.
    */
    return in.color;
}
