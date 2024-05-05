#version 330
#pragma vscode_glsllint_stage : geom

/**
  Given center point and computed basis vectors, emit the four quad positions
  This computes the virtual position to be used by fragment shader to compute the gaussian fall off
  The RGBA color of the GS is also passed as is to the geometry shader
 */

uniform vec2 basisViewport;
uniform float discardAlpha = 0.0001;

const float sqrt8 = sqrt(9);

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec2 basisVector0[];
in vec2 basisVector1[];
in vec4 gColor[];

out vec2 vPosition;
flat out vec4 vColor;  // pass through

vec2 computeNDCOffset(vec2 basisVector0, vec2 basisVector1, vec2 vPosition) {
    return (vPosition.x * basisVector0 + vPosition.y * basisVector1) * basisViewport * 2.0;
}

void main() {
    if (gColor[0].a < discardAlpha) {
        return;  // will not emit any quad for later rendering
    }

    vec2 ndcOffset;
    vec3 ndcCenter = gl_in[0].gl_Position.xyz;

    vPosition.x = -1;
    vPosition.y = -1;
    ndcOffset = computeNDCOffset(basisVector0[0], basisVector1[0], vPosition);  // compute offset
    gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);             // store output
    vPosition *= sqrt8;                                                         // store output
    vColor = gColor[0];                                                         // pass through
    EmitVertex();

    vPosition.x = -1;
    vPosition.y = 1;
    ndcOffset = computeNDCOffset(basisVector0[0], basisVector1[0], vPosition);  // compute offset
    gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);             // store output
    vPosition *= sqrt8;                                                         // store output
    vColor = gColor[0];                                                         // pass through
    EmitVertex();

    vPosition.x = 1;
    vPosition.y = -1;
    ndcOffset = computeNDCOffset(basisVector0[0], basisVector1[0], vPosition);  // compute offset
    gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);             // store output
    vPosition *= sqrt8;                                                         // store output
    vColor = gColor[0];                                                         // pass through
    EmitVertex();

    vPosition.x = 1;
    vPosition.y = 1;
    ndcOffset = computeNDCOffset(basisVector0[0], basisVector1[0], vPosition);  // compute offset
    gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);             // store output
    vPosition *= sqrt8;                                                         // store output
    vColor = gColor[0];                                                         // pass through
    EmitVertex();

    EndPrimitive();
}
