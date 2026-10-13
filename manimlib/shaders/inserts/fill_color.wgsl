fn fill_has_alpha() -> bool {
    if (mob.gradient_mode > 0.5) {
        let count = u32(clamp(mob.gradient_count, 2.0, 16.0));
        var index = 0u;
        loop {
            if (index >= count) { break; }
            if (mob.gradient_colors[index].a > 0.0) { return true; }
            index += 1u;
        }
        return false;
    }
    return max(mob.fill_rgba.a, mob.fill_rgba_end.a) > 0.0;
}

fn spread_gradient_t(t_in: f32) -> f32 {
    // Blend between spread modes instead of switching at the midpoint. This keeps
    // repeat/reflect/pad changes continuous during a Transform.
    let pad_t = clamp(t_in, 0.0, 1.0);
    let repeat_t = t_in - floor(t_in);
    let cycle = t_in - 2.0 * floor(t_in / 2.0);
    let reflect_t = select(2.0 - cycle, cycle, cycle <= 1.0);
    let mode = clamp(mob.gradient_spread, 0.0, 2.0);
    if (mode <= 1.0) {
        return mix(pad_t, repeat_t, mode);
    }
    return mix(repeat_t, reflect_t, mode - 1.0);
}

// SVG 2: color-interpolation selects encoded sRGB (default) or linearRGB.
// Gradient stops are interpolated channel by channel; alpha stays unassociated
// and is interpolated independently, as specified for SVG paint servers.
fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(
        c / 12.92,
        pow((c + vec3f(0.055)) / 1.055, vec3f(2.4)),
        c > vec3f(0.04045),
    );
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let safe_c = max(c, vec3f(0.0));
    return select(
        12.92 * safe_c,
        1.055 * pow(safe_c, vec3f(1.0 / 2.4)) - vec3f(0.055),
        safe_c > vec3f(0.0031308),
    );
}

fn multi_stop_color_at(t_in: f32) -> vec4f {
    let count = u32(clamp(mob.gradient_count, 2.0, 16.0));
    let t = spread_gradient_t(t_in);
    let first_offset = mob.gradient_stops[0u].x;
    if (t <= first_offset) {
        return mob.gradient_colors[0u];
    }

    var previous_offset = first_offset;
    var previous_color = mob.gradient_colors[0u];
    var index = 1u;
    loop {
        if (index >= count) { break; }
        let next_offset = mob.gradient_stops[index].x;
        let next_color = mob.gradient_colors[index];
        if (t <= next_offset) {
            let span = next_offset - previous_offset;
            let mix_t = select(
                0.0,
                clamp((t - previous_offset) / max(span, 1e-8), 0.0, 1.0),
                span > 1e-8,
            );
            let srgb_rgb = mix(previous_color.rgb, next_color.rgb, mix_t);
            let linear_rgb = linear_to_srgb(mix(
                srgb_to_linear(previous_color.rgb),
                srgb_to_linear(next_color.rgb),
                mix_t,
            ));
            let rgb = mix(
                srgb_rgb,
                linear_rgb,
                clamp(mob.gradient_interpolation, 0.0, 1.0),
            );
            let alpha = mix(previous_color.a, next_color.a, mix_t);
            return vec4f(rgb, alpha);
        }
        previous_offset = next_offset;
        previous_color = next_color;
        index += 1u;
    }
    return previous_color;
}

fn fill_color_at(point: vec3f) -> vec4f {
    if (mob.gradient_mode > 0.5) {
        let linear_axis = mob.gradient_end - mob.gradient_start;
        let linear_t = dot(point - mob.gradient_start, linear_axis)
            / max(dot(linear_axis, linear_axis), 1e-8);

        // Evaluate the radial field as well. During a linear <-> radial style
        // morph, gradient_mode moves smoothly from 1 to 2 and blends these two
        // coordinates, rather than taking a hard branch at the halfway point.
        var center = mob.gradient_radial_center.xy;
        var axis_x = mob.gradient_radial_axis_x.xy - center;
        var axis_y = mob.gradient_radial_axis_y.xy - center;
        var determinant = axis_x.x * axis_y.y - axis_x.y * axis_y.x;
        if (abs(determinant) < 1e-8) {
            // Invalid/degenerate user geometry gets a stable fallback, but ordinary
            // linear -> radial interpolation never passes through a singular basis.
            if (length(axis_x) < 1e-6) {
                axis_x = vec2f(1.0, 0.0);
            }
            axis_y = vec2f(-axis_x.y, axis_x.x);
            determinant = axis_x.x * axis_y.y - axis_x.y * axis_y.x;
        }
        let delta = point.xy - center;
        let point_x = (delta.x * axis_y.y - delta.y * axis_y.x) / determinant;
        let point_y = (axis_x.x * delta.y - axis_x.y * delta.x) / determinant;
        let focus_delta = mob.gradient_radial_focus.xy - center;
        let focus_x = (focus_delta.x * axis_y.y - focus_delta.y * axis_y.x) / determinant;
        let focus_y = (axis_x.x * focus_delta.y - axis_x.y * focus_delta.x) / determinant;
        let ray = vec2f(point_x - focus_x, point_y - focus_y);
        let ray_length_squared = dot(ray, ray);
        var radial_t = 0.0;
        if (ray_length_squared >= 1e-8) {
            let ray_focus = vec2f(focus_x, focus_y);
            let ray_focus_dot = dot(ray_focus, ray);
            let discriminant = max(
                ray_focus_dot * ray_focus_dot
                + ray_length_squared * (1.0 - dot(ray_focus, ray_focus)),
                0.0,
            );
            let edge_parameter = max(
                (-ray_focus_dot + sqrt(discriminant)) / ray_length_squared,
                1e-8,
            );
            radial_t = 1.0 / edge_parameter;
        }
        let shape_mix = clamp(mob.gradient_mode - 1.0, 0.0, 1.0);
        let t = mix(linear_t, radial_t, shape_mix);
        return multi_stop_color_at(t);
    }

    /*
    The original two-color linear gradient remains the fast path for set_fill's
    existing two-color API.
    */
    let axis = mob.gradient_end - mob.gradient_start;
    let t = dot(point - mob.gradient_start, axis) / max(dot(axis, axis), 1e-8);
    return mix(mob.fill_rgba, mob.fill_rgba_end, t);
}
