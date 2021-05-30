import numpy as np
import trimesh

def parse_obj_file(filename):
    faces = []
    vertices = []
    texture = []
    normals = []

    with open(filename, 'r') as stream:
        for line in stream:
            elements = line.replace('  ', ' ').strip().split(' ')
            if elements[0] == 'f':
                faces.append(elements[1:])
            elif elements[0] == 'vt':
                texture.append(elements[1:])
            elif elements[0] == 'v':
                vertices.append(elements[1:])
            elif elements[0] == 'vn':
                normals.append(elements[1:])
            else:
                # print(elements)
                pass
    return faces, vertices, texture, normals

def parse_faces(faces):
    new_faces = []
    for face in faces:
        int_face = [ int(vi.split('/')[0])-1 for vi in face ]
        new_faces.append(int_face)
    return new_faces

def extract_texture_correspondences(faces, num_vertices):
    vertices = [ set() for i in range(num_vertices) ]

    for face_els in faces:
        for el in face_els:
            vert = [ int(num)-1 for num in el.split('/') ]
            vertices[vert[0]].add(vert[1])

    return [ list(el) for el in vertices ]

def extract_uv_mapping(vertices, uv, faces, normals=[]):
    correspondences = extract_texture_correspondences(faces, len(vertices))

    real_points = []
    uv_grid     = []
    real_normals = []

    # there are points which do not appear in the mesh (do not belong to any face)
    for idx, el in enumerate(correspondences):
        if len(el) > 0 :
            point = uv[el[0]]
            uv_grid.append([ float(e) for e in point ])
            real_points.append([ float(e) for e in vertices[idx] ])
            if len(normals) > 0:
                real_normals.append([ float(e) for e in normals[idx] ])

    return np.array(real_points), np.array(uv_grid), np.array(real_normals)


def fix_face_indexing(faces, points):
    max_index = 0
    for face in faces:
        if max(face) > max_index:
            max_index = max(face)
    verts_active = np.zeros(max_index+1, np.uint8)
    for face in faces:
        verts_active[face[0]] = 1
        verts_active[face[1]] = 1
        verts_active[face[2]] = 1
    fix_indices = np.cumsum(verts_active)

    new_faces = []
    for face in faces:
        new_faces.append([ fix_indices[idx]-1 for idx in face ])
    return new_faces

def read_mesh_from_obj(obj_file):
    faces, vertices, texture, normals = parse_obj_file(obj_file)
    points, uv_grid, normals = extract_uv_mapping(vertices, texture, faces, normals)
    faces = parse_faces(faces)
    faces = fix_face_indexing(faces, points)
    return points, uv_grid, faces, normals

def write_faces(file, faces):
    file_stream = open(file, 'w')
    for item in faces:
        file_stream.write("{0} {1} {2}\n".format(item[0],item[1],item[2]))

    file_stream.close()


def write_obj_file(filename, vertices, faces, texture_coords, normals):
    file_stream = open(filename, 'w')
    for item in vertices:
        file_stream.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in normals:
        file_stream.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in texture_coords:
        file_stream.write("vt {0} {1}\n".format(item[0],item[1]))

    for item in faces:
        file_stream.write("f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n".format(int(item[0]+1),int(item[1]+1),int(item[2]+1)))

    file_stream.close()


