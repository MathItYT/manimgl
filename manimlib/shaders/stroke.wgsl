/*
A stroke along a path, drawn as a strip of quads following each bezier, with a fan of
triangles rounding off the joint at each end.

The border around a fill is a stroke like any other, except that it takes its color and
width from the fill's fields rather than the stroke's, and that a width of zero still draws
a band just wide enough to anti-alias the fill's edge. That is settled when the shader is
compiled rather than asked per draw, so the two are two modules from this one source, see
VShaderWrapper.init_program.
*/
const IS_FILL_BORDER: bool = false;

#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT fill_color.wgsl
#INSERT finalize_color.wgsl
#INSERT clip_test.wgsl

// Beyond this much alignment between the tangent and the view direction, the step to the
// side of the curve gets adjusted to avoid glitches.
const ALIGNMENT_THRESHOLD: f32 = 0.99;

// Used to determine how many lines to break the curve into.
const POLYLINE_FACTOR: f32 = 100.0;

// This must agree with VDrawing.stroke_verts_per_curve:
//     6 * (MAX_STEPS - 1)
const MAX_STEPS: i32 = 32;

// Stands in for a record index where there is no neighboring curve to read.
const NONE: i32 = -1;

// Small tolerance used for degenerate handles/tangents.
const EPSILON: f32 = 1e-7;

// Over this range of turn cosines, a joint eases from a sharp miter to a round end.
const ROUND_COS_SHARP: f32 = -0.5;
const ROUND_COS_ROUND: f32 = -0.95;

// Number of units spanned by a stroke_width of 1 in a default scale frame,
// so for instance a stroke_width of 100 comes out one unit thick.
const STROKE_WIDTH_CONVERSION: f32 = 0.01;

/*
A bezier is three consecutive records of the buffer, sharing its last with the next curve's
first, so curve n begins at record 2n. It's drawn as one quad for each of the polyline
segments it gets broken into, and that count per curve has to match what VShaderWrapper
draws.

The last few of those segments go instead to a fan of triangles rounding off the joint at
the curve's end. A curve rarely needs anywhere near its full allowance of polyline steps, so
this costs nothing that was being used.
*/
const RECORD_STEP: u32 = 2u;
const VERTS_PER_CURVE: u32 = u32(6 * (MAX_STEPS - 1));

const JOINT_SEGMENTS: i32 = 3;
const POLYLINE_SEGMENTS: i32 = MAX_STEPS - 1 - JOINT_SEGMENTS;
const FAN_TRIANGLES: i32 = 2 * JOINT_SEGMENTS;

// The two triangles of one segment's quad, as (which end of it, which side).
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

    @location(0) clip_distances: vec4f,

    @location(1) color: vec4f,

    // Distance to the curve, and half the curve's width, both as a ratio of the
    // antialias width.
    @location(2) dist_to_aaw: f32,

    @location(3) half_width_to_aaw: f32,
};


// -----------------------------------------------------------------------------
// Basic curve helpers
// -----------------------------------------------------------------------------

fn point_on_quadratic(t: f32, c0: vec3f, c1: vec3f, c2: vec3f) -> vec3f {
    return c0 + c1 * t + c2 * t * t;
}


fn tangent_on_quadratic(t: f32, c1: vec3f, c2: vec3f) -> vec3f {
    return c1 + 2.0 * c2 * t;
}


// The vector as it appears in the plane perpendicular to a given unit normal.
fn project(vect: vec3f, normal: vec3f) -> vec3f {
    return vect - dot(vect, normal) * normal;
}


fn rotate_vector(vect: vec3f, normal: vec3f, turn: vec2f) -> vec3f {
    return turn.x * vect + turn.y * cross(normal, vect);
}


// -----------------------------------------------------------------------------
// Safe vector helpers
// -----------------------------------------------------------------------------

fn vector_length_squared(v: vec3f) -> f32 {
    return dot(v, v);
}


fn is_degenerate(v: vec3f) -> bool {
    return vector_length_squared(v) <= EPSILON * EPSILON;
}


/*
Normalize a vector while avoiding NaNs for zero-length vectors.

This is particularly important for text VMobjects because their paths can contain
coincident anchors and zero-length handles.
*/
fn safe_normalize(v: vec3f, fallback: vec3f) -> vec3f {
    let len2 = vector_length_squared(v);

    if (len2 <= EPSILON * EPSILON) {
        let fallback_len2 = vector_length_squared(fallback);

        if (fallback_len2 <= EPSILON * EPSILON) {
            return vec3f(0.0, 0.0, 0.0);
        }

        return fallback * inverseSqrt(fallback_len2);
    }

    return v * inverseSqrt(len2);
}


// -----------------------------------------------------------------------------
// Neighbor detection
// -----------------------------------------------------------------------------

/*
The tangent of the curve neighbouring this one at the given end, pointing the same way along
the path. Where the subpath ends, and so has no neighbour to make a joint with, this comes
back as zero.

A border's closing chord is a neighbour like any other curve: it runs straight, so either of
its endpoints stands in for the handle a curve would have had at the other.
*/
fn neighbor_tangent(
    record: i32,
    subpath: vec2f,
    at_start: bool,
    anchor: vec3f,
) -> vec3f {
    // How far the subpath reaches either side of the record, see VMobject.set_subpath_range.
    let first = record - i32(subpath.x);
    let last = record + i32(subpath.y);

    // Protect against malformed ranges.
    if (first < 0 || last < first) {
        return vec3f(0.0);
    }

    let first_point = read_vec3(u32(first), DATA_OFFSET_point);
    let last_point = read_vec3(u32(last), DATA_OFFSET_point);

    let closed = all(first_point == last_point);

    if (at_start) {
        var previous = NONE;

        if (record > first) {
            previous = record - 1;
        } else if (closed && last > first) {
            previous = last - 1;
        } else if (IS_FILL_BORDER) {
            previous = last;
        }

        if (previous == NONE || previous < 0) {
            return vec3f(0.0);
        }

        let tangent = anchor - read_vec3(
            u32(previous),
            DATA_OFFSET_point,
        );

        if (is_degenerate(tangent)) {
            return vec3f(0.0);
        }

        return tangent;
    }

    var next = NONE;

    if (record + 2 < last) {
        next = record + 3;
    } else if (closed || record == last) {
        next = first + 1;
    } else if (IS_FILL_BORDER) {
        next = first;
    }

    if (next == NONE || next < 0) {
        return vec3f(0.0);
    }

    let tangent = read_vec3(
        u32(next),
        DATA_OFFSET_point,
    ) - anchor;

    if (is_degenerate(tangent)) {
        return vec3f(0.0);
    }

    return tangent;
}


// -----------------------------------------------------------------------------
// Tangent / joint calculations
// -----------------------------------------------------------------------------

/*
The tangent as it appears in the plane the stroke is drawn in, or zero for anything
degenerate, such as the tangent at a repeated point.
*/
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


/*
How far along its tangent the incoming strip must run to reach where the outgoing strip's
edge meets it, the exact miter.

Degenerate tangents are deliberately treated as having no miter shift. The neighboring
geometry will still be rendered, rather than being discarded.
*/
fn joint_shift(
    tan_in: vec3f,
    tan_out: vec3f,
    facing_normal: vec3f,
) -> f32 {
    let a = flat_tangent(tan_in, facing_normal);
    let b = flat_tangent(tan_out, facing_normal);

    if (is_degenerate(a) || is_degenerate(b)) {
        return 0.0;
    }

    let sin_angle = dot(
        cross(a, b),
        facing_normal,
    );

    // Both a straight joint and a full reversal want no shift.
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


/*
Step perpendicular to the curve, out to the edge of the stroke, then along the curve by
however far the joint at this end reaches.
*/
fn step_to_corner(
    tangent: vec3f,
    facing_normal: vec3f,
    shift: f32,
    draw_flat: bool,
) -> vec3f {
    var unflattened = tangent;

    if (!draw_flat) {
        unflattened = project(
            tangent,
            facing_normal,
        );
    }

    /*
    A degenerate tangent can occur in text paths. In that case use the object's normal
    as a fallback instead of allowing normalize() to generate NaNs.
    */
    var unit_tan = safe_normalize(
        unflattened,
        vec3f(0.0),
    );

    if (is_degenerate(unit_tan)) {
        unit_tan = safe_normalize(
            mob.unit_normal,
            vec3f(0.0, 0.0, 1.0),
        );
    }

    var step = cross(
        facing_normal,
        unit_tan,
    );

    step = safe_normalize(
        step,
        vec3f(0.0),
    );

    /*
    For non-flat stroke, there can be glitches when the tangent direction lines up very
    closely with the direction to the camera, treated here as the unit normal. To avoid
    those, this smoothly transitions to a step direction perpendicular to the true curve
    normal.
    */
    let tangent_normalized = safe_normalize(
        tangent,
        unit_tan,
    );

    let alignment = abs(
        dot(
            tangent_normalized,
            facing_normal,
        ),
    );

    if (alignment > ALIGNMENT_THRESHOLD) {
        let perp_raw = cross(
            mob.unit_normal,
            tangent_normalized,
        );

        if (!is_degenerate(perp_raw)) {
            let perp = safe_normalize(
                perp_raw,
                step,
            );

            let projected_step = project(
                step,
                perp,
            );

            if (!is_degenerate(projected_step)) {
                step = mix(
                    step,
                    safe_normalize(
                        projected_step,
                        step,
                    ),
                    smoothstep(
                        ALIGNMENT_THRESHOLD,
                        1.0,
                        alignment,
                    ),
                );
            }
        }
    }

    return step + shift * unit_tan;
}


// -----------------------------------------------------------------------------
// Vertex shader
// -----------------------------------------------------------------------------

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let blank = vec4f(
        0.0,
        0.0,
        0.0,
        1.0,
    );

    // Which quadratic curve this vertex belongs to.
    let curve = vertex_index / VERTS_PER_CURVE;

    // Vertex inside the curve's reserved geometry.
    let within = vertex_index % VERTS_PER_CURVE;

    // Which 6-vertex quad/fan element.
    let segment = i32(within / 6u);

    // Vertex inside that triangle pair.
    let tri_vert = i32(within % 6u);

    let corner = CORNERS[tri_vert];

    // Each quadratic begins at every second record.
    let record = RECORD_STEP * curve;

    // The final JOINT_SEGMENTS are reserved for the joint fan.
    let joint_fan = segment >= POLYLINE_SEGMENTS;

    let subpath = read_vec2(
        record,
        DATA_OFFSET_subpath_range,
    );

    var controls = array<vec3f, 3>(
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
    A fill counts its winding as though every subpath ran back to its own first point,
    see fill.wgsl, so the border traces that chord too, in the slot the curve marking
    the subpath's end sits in.
    */
    if (IS_FILL_BORDER && subpath.y == 0.0) {
        let first_record = i32(record) - i32(subpath.x);

        if (first_record >= 0) {
            controls[2] = read_vec3(
                u32(first_record),
                DATA_OFFSET_point,
            );

            controls[1] = 0.5 * (
                controls[0] + controls[2]
            );
        }
    }

    let widths = array<f32, 3>(
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

    let colors = array<vec4f, 3>(
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

    // Coefficients for:
    //
    //     c0 + c1*t + c2*t^2
    //
    let c0 = controls[0];

    let c1 = 2.0 * (
        controls[1] - controls[0]
    );

    let c2 = controls[0]
        - 2.0 * controls[1]
        + controls[2];

    /*
    Determine how many polyline segments are actually necessary for this quadratic.
    */
    let area = 0.5 * length(
        cross(
            controls[1] - controls[0],
            controls[2] - controls[0],
        ),
    );

    let frame_unit = max(
        get_frame_unit_size(),
        EPSILON,
    );

    let count = i32(
        round(
            POLYLINE_FACTOR
            * sqrt(max(area, 0.0))
            / frame_unit,
        ),
    );

    let n_steps = clamp(
        2 + count,
        2,
        POLYLINE_SEGMENTS + 1,
    );

    /*
    A curve is degenerate when its first anchor and first handle coincide.

    Keep the original behavior here for genuinely empty curves, but make the comparison
    tolerant to tiny floating-point differences.
    */
    let initial_handle_delta = controls[1] - controls[0];

    var nothing_to_draw = vector_length_squared(
        initial_handle_delta
    ) <= EPSILON * EPSILON;

    /*
    Segments after n_steps belong to the unused portion of the reserved geometry.
    */
    nothing_to_draw = nothing_to_draw
        || (
            !joint_fan
            && segment >= n_steps - 1
        );

    if (IS_FILL_BORDER) {
        nothing_to_draw = nothing_to_draw
            || max(
                mob.fill_rgba.a,
                mob.fill_rgba_end.a,
            ) == 0.0;
    } else {
        nothing_to_draw = nothing_to_draw
            || all(
                vec3f(
                    widths[0],
                    widths[1],
                    widths[2],
                ) == vec3f(0.0)
            )
            || all(
                vec3f(
                    colors[0].a,
                    colors[1].a,
                    colors[2].a,
                ) == vec3f(0.0)
            );
    }

    if (nothing_to_draw) {
        out.position = blank;
        out.clip_distances = vec4f(0.0);
        out.color = vec4f(0.0);
        out.dist_to_aaw = 0.0;
        out.half_width_to_aaw = 0.0;
        return out;
    }

    /*
    The fan sits at the curve's end, where the polyline's last point also lands.
    */
    let index = segment + i32(corner.x);

    var t = 1.0;

    if (!joint_fan) {
        t = f32(index)
            / f32(max(n_steps - 1, 1));
    }

    var point = controls[2];

    if (!joint_fan) {
        point = point_on_quadratic(
            t,
            c0,
            c1,
            c2,
        );
    }

    /*
    Tangent at the current point.
    */
    var tangent = tangent_on_quadratic(
        t,
        c1,
        c2,
    );

    /*
    A quadratic with coincident handles can have a zero tangent. The geometric position
    remains valid; use the chord as a fallback direction.
    */
    if (is_degenerate(tangent)) {
        tangent = controls[2] - controls[0];
    }

    /*
    Stroke width is measured relative to the frame unless it is explicitly in scene units.
    */
    var own_width = mix(
        widths[0],
        widths[2],
        t,
    );

    if (IS_FILL_BORDER) {
        own_width = mob.fill_border_width;
    }

    let width = STROKE_WIDTH_CONVERSION
        * mix(
            get_frame_unit_size(),
            1.0,
            mob.stroke_width_in_scene_units,
        )
        * own_width;

    let draw_flat =
        mob.flat_stroke != 0.0
        || mob.is_fixed_in_frame != 0.0;

    var facing_normal = safe_normalize(
        frame.camera_position - point,
        mob.unit_normal,
    );

    if (draw_flat) {
        facing_normal = safe_normalize(
            mob.unit_normal,
            vec3f(0.0, 0.0, 1.0),
        );
    }

    /*
    The fill's color varies linearly.
    */
    var own_color = mix(
        colors[0],
        colors[2],
        t,
    );

    if (IS_FILL_BORDER) {
        own_color = fill_color_at(point);
    }

    out.color = finalize_color(
        own_color,
        point,
        facing_normal,
    );

    /*
    Anti-alias width is measured in pixels.
    */
    let aaw = max(
        mob.anti_alias_width * get_pixel_unit_size(),
        1e-8,
    );

    let half_width = 0.5 * (
        width + aaw
    );

    /*
    -------------------------------------------------------------------------
    JOINT INFORMATION
    -------------------------------------------------------------------------
    */
    let at_start =
        !joint_fan
        && index == 0;

    let at_joint =
        joint_fan
        || at_start
        || index == n_steps - 1;

    var neighbor = vec3f(0.0);

    if (at_joint) {
        neighbor = neighbor_tangent(
            i32(record),
            subpath,
            at_start,
            point,
        );
    }

    var has_joint =
        at_joint
        && !is_degenerate(neighbor);

    /*
    Calculate the miter shift only if both directions are usable.
    */
    var shift = 0.0;

    if (has_joint) {
        if (at_start) {
            shift = -joint_shift(
                neighbor,
                tangent,
                facing_normal,
            );
        } else {
            shift = joint_shift(
                tangent,
                neighbor,
                facing_normal,
            );
        }
    }

    /*
    -------------------------------------------------------------------------
    STEP / DISTANCE
    -------------------------------------------------------------------------
    */
    var step = vec3f(0.0);
    var dist_to_curve = 0.0;
    var edge_dist = 0.0;

    if (joint_fan) {
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

    } else {
        /*
        Regular polyline segment.
        */
        step = step_to_corner(
            tangent,
            facing_normal,
            shift,
            draw_flat,
        );

        edge_dist = 0.5 * width;

        dist_to_curve =
            corner.y * half_width;
    }

    /*
    Final AA parameters.
    */
    out.half_width_to_aaw =
        edge_dist / aaw;

    out.dist_to_aaw =
        dist_to_curve / aaw;

    /*
    Project the final 3D stroke position.
    */
    let projection = project_point(
        point
        + dist_to_curve * step,
    );

    out.position = projection.position;
    out.clip_distances = projection.clip_distances;

    return out;
}


// -----------------------------------------------------------------------------
// Fragment shader
// -----------------------------------------------------------------------------

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4f {
    clip_test(
        in.clip_distances
    );

    var color = in.color;

    /*
    Signed distance to the region around the curve which is to be colored.
    */
    let signed_dist_to_region =
        abs(in.dist_to_aaw)
        - in.half_width_to_aaw;

    color.a *=
        1.0
        - smoothstep(
            -0.5,
            0.5,
            signed_dist_to_region,
        );

    return color;
}