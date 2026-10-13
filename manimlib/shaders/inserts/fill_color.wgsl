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
    if (mob.gradient_spread < 0.5) {
        return clamp(t_in, 0.0, 1.0);
    }
    if (mob.gradient_spread < 1.5) {
        return t_in - floor(t_in);
    }
    let cycle = t_in - 2.0 * floor(t_in / 2.0);
    return select(2.0 - cycle, cycle, cycle <= 1.0);
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
            var previous_rgb = previous_color.rgb;
            var next_rgb = next_color.rgb;
            if (mob.gradient_interpolation > 0.5) {
                previous_rgb = srgb_to_linear(previous_rgb);
                next_rgb = srgb_to_linear(next_rgb);
            }
            var rgb = mix(previous_rgb, next_rgb, mix_t);
            if (mob.gradient_interpolation > 0.5) {
                rgb = linear_to_srgb(rgb);
            }
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
        var t = 0.0;
        if (mob.gradient_mode > 1.5) {
            let center = mob.gradient_start.xy;
            let axis_x = mob.gradient_end.xy - center;
            let axis_y = mob.gradient_axis_y.xy - center;
            let delta = point.xy - center;
            var determinant = axis_x.x * axis_y.y - axis_x.y * axis_y.x;
            if (abs(determinant) < 1e-8) {
                determinant = select(-1e-8, 1e-8, determinant >= 0.0);
            }
            let point_x = (delta.x * axis_y.y - delta.y * axis_y.x) / determinant;
            let point_y = (axis_x.x * delta.y - axis_x.y * delta.x) / determinant;
            let focus_delta = mob.gradient_focus.xy - center;
            let focus_x = (focus_delta.x * axis_y.y - focus_delta.y * axis_y.x) / determinant;
            let focus_y = (axis_x.x * focus_delta.y - axis_x.y * focus_delta.x) / determinant;
            let ray = vec2f(point_x - focus_x, point_y - focus_y);
            let ray_length_squared = dot(ray, ray);
            if (ray_length_squared < 1e-8) {
                t = 0.0;
            } else {
                let ray_focus_dot = dot(vec2f(focus_x, focus_y), ray);
                let discriminant = max(
                    ray_focus_dot * ray_focus_dot
                    + ray_length_squared * (1.0 - dot(vec2f(focus_x, focus_y), vec2f(focus_x, focus_y))),
                    0.0,
                );
                let edge_parameter = max(
                    (-ray_focus_dot + sqrt(discriminant)) / ray_length_squared,
                    1e-8,
                );
                t = 1.0 / edge_parameter;
            }
        } else {
            let axis = mob.gradient_end - mob.gradient_start;
            t = dot(point - mob.gradient_start, axis) / max(dot(axis, axis), 1e-8);
        }
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
