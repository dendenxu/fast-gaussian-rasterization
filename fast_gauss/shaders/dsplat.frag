#version 330
#pragma vscode_glsllint_stage : frag

/**
  Almost empty fragment shader for computing the final output color.
  Note that the geometry should've been emitted in the order they want to be rendered (back to front) and blended normally
 */

in vec2 vPosition;
flat in vec4 vColor;
flat in float vDepth;

uniform bool useDepth = false;
uniform bool solidMode = false;
uniform bool edgeMode = false;

uniform float eight = 8;
uniform float minAlpha = 0.004;
uniform float maxAlpha = 0.99;
uniform float edgeThreshold = 0.025;

layout(location = 0) out vec4 write_color;
// layout(location = 1) out vec4 write_depth;

void main() {
    // Compute the positional squared distance from the center of the splat to the current fragment.
    float A = dot(vPosition, vPosition);

    // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
    // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
    // defined by the rectangle formed by vPosition. It also means it's farther
    // away than sqrt(8) standard deviations from the mean.
    if (A > eight) discard;
    float power = -0.5 * A;
    // if (power > 0.0f) discard;

    // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
    // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
    // and since 'mean' is zero, we have X * X, which is the same as A:
    float opacity = exp(power) * vColor.a;
    // float opacity = exp(-0.5 * A) * vColor.a;
    if (opacity < minAlpha) discard;
    // opacity = min(maxAlpha, opacity);

    vec3 color = vColor.rgb;
    if (solidMode) opacity = 1.0;
    if (useDepth) color = vec3(vDepth);
    if (edgeMode && (exp(power) < edgeThreshold)) {
    // color = vec3(1.0, 0.0, 0.0); // red edge
    // if (exp(power) <= edgeThreshold) {
        color = vec3(0.0, 0.0, 0.0); // red edge
        opacity = 1.0;
    }
    // write_color = vec4(vec3(edgeMode), opacity);
    write_color = vec4(color, opacity);
}
