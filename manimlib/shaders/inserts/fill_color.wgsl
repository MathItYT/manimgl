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
            return mix(previous_color, next_color, mix_t);
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
            let radial_x = (delta.x * axis_y.y - delta.y * axis_y.x) / determinant;
            let radial_y = (axis_x.x * delta.y - axis_x.y * delta.x) / determinant;
            t = length(vec2f(radial_x, radial_y));
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
