import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


DOPE_DIMENSIONS_M = np.array([0.09602415084838867, 0.1913010025024414, 0.05824894905410733])
SAMPLE_COUNT = 96


def read_vertices(obj_path):
    vertices = []
    with obj_path.open("r", encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if len(vertices) < 4:
        raise ValueError("The OBJ file does not contain enough vertices.")
    return np.asarray(vertices, dtype=np.float64)


def convert_to_dope_frame(vertices):
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    dimensions = maximum - minimum

    # YCB uses Z as the bottle axis; DOPE's mustard cuboid uses Y.
    points = vertices[:, [0, 2, 1]] - center[[0, 2, 1]]
    points[:, 2] *= -1.0
    points *= DOPE_DIMENSIONS_M / dimensions[[0, 2, 1]]
    return points


def build_cao(vertices):
    sample_indices = np.linspace(0, len(vertices) - 1, SAMPLE_COUNT, dtype=int)
    extrema_indices = np.concatenate((vertices.argmin(axis=0), vertices.argmax(axis=0)))
    sampled = vertices[np.unique(np.concatenate((sample_indices, extrema_indices)))]
    hull = ConvexHull(sampled)

    hull_vertices = sampled[hull.vertices]
    index_map = {int(index): local for local, index in enumerate(hull.vertices)}
    faces = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        ordered = [index_map[int(index)] for index in simplex]
        a, b, c = hull_vertices[ordered]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            ordered[1], ordered[2] = ordered[2], ordered[1]
        faces.append(ordered)

    lines = ["V1", "# Convex hull of the YCB google_16k mesh in the DOPE object frame.",
             str(len(hull_vertices))]
    lines.extend("{:.8f} {:.8f} {:.8f}".format(*point) for point in hull_vertices)
    lines.extend(["0", "0", str(len(faces))])
    lines.extend("3 {} {} {}".format(*face) for face in faces)
    lines.extend(["0", "0"])
    return "\n".join(lines) + "\n", hull_vertices, len(faces)


def main():
    parser = argparse.ArgumentParser(description="Convert a YCB OBJ mesh to a compact ViSP CAO hull.")
    parser.add_argument("obj", type=Path)
    parser.add_argument("cao", type=Path)
    args = parser.parse_args()

    vertices = convert_to_dope_frame(read_vertices(args.obj))
    cao_text, hull_vertices, face_count = build_cao(vertices)
    bounds = hull_vertices.max(axis=0) - hull_vertices.min(axis=0)
    if not np.allclose(bounds, DOPE_DIMENSIONS_M, atol=1e-6):
        raise RuntimeError("Converted mesh bounds do not match the DOPE Mustard dimensions.")

    args.cao.parent.mkdir(parents=True, exist_ok=True)
    args.cao.write_text(cao_text, encoding="ascii")
    print("Wrote {} with {} vertices and {} triangular faces.".format(
        args.cao, len(hull_vertices), face_count
    ))


if __name__ == "__main__":
    main()
