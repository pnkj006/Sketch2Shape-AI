
import os
import numpy as np
import trimesh


class MeshValidator:
    """
    Validate, clean and export 3D meshes.
    """

    def __init__(self, vertices, faces, colors=None):
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            vertex_colors=colors,
            process=True
        )

        self.removed_component_count = 0

    def remove_small_components(self, min_face_count=100):
        """
        Remove tiny disconnected mesh artifacts.

        Always preserve the largest component.
        Also preserve other components with at least
        min_face_count faces.
        """

        components = self.mesh.split(
            only_watertight=False
        )

        if not components:
            raise ValueError("No valid mesh components found.")

        # Sort components by face count, largest first
        components = sorted(
            components,
            key=lambda component: len(component.faces),
            reverse=True
        )

        # Always retain the largest component
        valid_components = [components[0]]

        # Retain other meaningful components
        for component in components[1:]:
            if len(component.faces) >= min_face_count:
                valid_components.append(component)

        self.removed_component_count = (
            len(components) - len(valid_components)
        )

        # Recombine retained components
        self.mesh = trimesh.util.concatenate(
            valid_components
        )

        return self.mesh

    def validate(self):
        """
        Perform mesh validation after cleaning.
        """

        finite_vertices = np.isfinite(
            self.mesh.vertices
        ).all()

        finite_faces = np.isfinite(
            self.mesh.faces
        ).all()

        edge_counts = np.bincount(
            self.mesh.edges_unique_inverse
        )

        boundary_edges = int(
            np.sum(edge_counts == 1)
        )

        components = self.mesh.split(
            only_watertight=False
        )

        return {
            "vertex_count": len(self.mesh.vertices),
            "face_count": len(self.mesh.faces),

            "watertight": bool(self.mesh.is_watertight),
            "winding_consistent": bool(
                self.mesh.is_winding_consistent
            ),
            "is_volume": bool(self.mesh.is_volume),

            "finite_vertices": bool(finite_vertices),
            "finite_faces": bool(finite_faces),

            "boundary_edges": boundary_edges,
            "connected_components": len(components),
            "removed_components": self.removed_component_count,

            "surface_area": float(self.mesh.area),
            "volume": float(self.mesh.volume),
            "bounds": self.mesh.bounds.tolist()
        }

    def export(self, output_dir, filename="generated_model"):
        """
        Export cleaned mesh in GLB and OBJ formats.
        """

        os.makedirs(output_dir, exist_ok=True)

        glb_path = os.path.join(
            output_dir,
            f"{filename}.glb"
        )

        obj_path = os.path.join(
            output_dir,
            f"{filename}.obj"
        )

        self.mesh.export(glb_path)
        self.mesh.export(obj_path)

        return {
            "glb_path": glb_path,
            "obj_path": obj_path,
            "glb_size_mb": round(
                os.path.getsize(glb_path) / (1024 ** 2),
                2
            ),
            "obj_size_mb": round(
                os.path.getsize(obj_path) / (1024 ** 2),
                2
            )
        }

    def get_mesh(self):
        """
        Return the processed Trimesh object.
        """

        return self.mesh
