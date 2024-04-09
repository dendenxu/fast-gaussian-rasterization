#version 330
#pragma vscode_glsllint_stage : vert

/**
  Compute covariance relatec computation & projection etc
  This emits the 2 2D basis vector for geometry shader's quad construction
  The RGBA color of the GS is also passed as is to the geometry shader
 */

uniform mat4x4 P;
uniform mat4x4 VM;
uniform vec2 focal;

uniform float minAlpha = 1 / 255;
uniform float maxScreenSpaceSplatSize = 2048.0;

const float sqrt8 = sqrt(8);

layout(location = 0) in vec3 aPos;     // xyz
layout(location = 1) in vec3 aCov0_3;  // cov6
layout(location = 2) in vec3 aCov3_6;  // cov6
layout(location = 3) in vec4 aColor;   // rgba

out vec2 basisVector0;
out vec2 basisVector1;
out vec4 gColor;  // pass through

void main() {
    if (aColor.a < minAlpha) {
        gColor.a = 0.0;  // will not emit things in geometry shader
        return;
    }

    // Compute the view and clip space coordinates of the center of the ellipse
    vec4 viewCenter = VM * vec4(aPos, 1.0);
    vec4 clipCenter = P * vec4(aPos, 1.0);
    clipCenter = clipCenter / clipCenter.w;  // perspective division

    // Construct the 3D covariance matrix
    mat3 Vrk = mat3(
        aCov0_3[0], aCov0_3[1], aCov0_3[2],
        aCov0_3[1], aCov3_6[0], aCov3_6[1],
        aCov0_3[2], aCov3_6[1], aCov3_6[2]);

    // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
    float s = 1.0 / (viewCenter.z * viewCenter.z);
    mat3 J = mat3(
        focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
        0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
        0., 0., 0.);

    // Concatenate the projection approximation with the model-view transformation
    mat3 W = transpose(mat3(VM));
    mat3 T = W * J;

    // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
    mat3 cov2Dm = transpose(T) * Vrk * T;
    cov2Dm[0][0] += 0.3;
    cov2Dm[1][1] += 0.3;

    // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
    // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
    // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
    // need cov2Dm[1][0] because it is a symetric matrix.
    vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

    // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
    // so that we can determine the 2D basis for the splat. This is done using the method described
    // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
    // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * eigen-value), which is
    // equal to scaling them by sqrt(8) standard deviations.
    //
    // This is a different approach than in the original work at INRIA. In that work they compute the
    // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
    // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
    // times the maximum eigen-value, or 3 standard deviations. They then use the inverse 2D covariance
    // matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by calculating the
    // full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
    float a = cov2Dv.x;
    float d = cov2Dv.z;
    float b = cov2Dv.y;
    float D = a * d - b * b;
    float trace = a + d;
    float traceOver2 = 0.5 * trace;
    float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
    float eigenValue0 = traceOver2 + term2;
    float eigenValue1 = traceOver2 - term2;

    if (eigenValue1 <= 0.0) {
        gColor.a = 0.0;  // will not emit things
        return;
    }

    vec2 eigenVector0 = normalize(vec2(b, eigenValue0 - a));
    // since the eigen vectors are orthogonal, we derive the second one from the first
    vec2 eigenVector1 = vec2(eigenVector0.y, -eigenVector0.x);

    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    basisVector0 = eigenVector0 * min(sqrt8 * sqrt(eigenValue0), maxScreenSpaceSplatSize);
    basisVector1 = eigenVector1 * min(sqrt8 * sqrt(eigenValue1), maxScreenSpaceSplatSize);

    gl_Position = clipCenter;  // doing a perspective projection to ndc space
    gColor = aColor;           // passing through
}