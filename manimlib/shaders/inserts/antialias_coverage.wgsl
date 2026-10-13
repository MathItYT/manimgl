// Shared coverage curve used by both ordinary strokes and filled Bezier boundaries.
// The input is signed distance to the visible region, measured in antialias-width units.
fn stroke_aa_coverage(signed_dist_to_region: f32) -> f32 {
    return 1.0 - smoothstep(-0.5, 0.5, signed_dist_to_region);
}
