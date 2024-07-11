#version 330
#pragma vscode_glsllint_stage : frag

/**
  Almost empty fragment shader for computing the final output color.
  Note that the geometry should've been emitted in the order they want to be rendered (back to front) and blended normally
 */

in vec2 vPosition;
flat in vec4 vColor;

uniform float eight = 8;
uniform float minAlpha = 1 / 255;

void main() {
    // Compute the positional squared distance from the center of the splat to the current fragment.
    float A = dot(vPosition, vPosition);

    // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
    // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
    // defined by the rectangle formed by vPosition. It also means it's farther
    // away than sqrt(8) standard deviations from the mean.
    if (A > eight) discard;
    float power = -0.5 * A;
    // if (power > 0.0f)
    //     discard;

    // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
    // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
    // and since 'mean' is zero, we have X * X, which is the same as A:
    float opacity = exp(power) * vColor.a;
    // float opacity = exp(-0.5 * A) * vColor.a;
    if (opacity < minAlpha)
        discard;
    // opacity = min(0.99, opacity);

    gl_FragColor = vec4(vColor.rgb, opacity);
}
