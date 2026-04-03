"""

Multi-Layer Mesh Generator

"""



import os as _os

from pathlib import Path as _Path

BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))



import numpy as np

import nibabel as nib

from pathlib import Path

import logging

from typing import Dict, List

import json

from skimage import measure



logger = logging.getLogger(__name__)





class MultiLayerMeshGenerator:

    

    def __init__(self):

        self.logger = logging.getLogger(self.__class__.__name__)

    

    def generate_organ_mesh(

        self,

        organ_volume_path: str,

        organ_label: int,

        organ_name: str,

        spacing: tuple,

        target_faces: int = 5000

    ) -> Dict:

        self.logger.info(f"Generating mesh for {organ_name} (label {organ_label})...")

        

        organ_nii = nib.load(organ_volume_path)

        organ_volume = organ_nii.get_fdata()

        

        organ_mask = (organ_volume == organ_label).astype(np.uint8)

        num_voxels = np.sum(organ_mask)

        

        if num_voxels == 0:

            self.logger.warning(f"No voxels found for {organ_name}")

            return None

        

        self.logger.info(f"  Organ voxels: {num_voxels:,}")

        

        try:

            verts, faces, normals, values = measure.marching_cubes(

                organ_mask,

                level=0.5,

                spacing=spacing,

                step_size=1

            )

            

            self.logger.info(f"  Marching Cubes: {len(verts):,} vertices, {len(faces):,} faces")

            

            if len(faces) > target_faces:

                ratio = target_faces / len(faces)

                step = int(1 / ratio)

                

                faces_simplified = faces[::step]

                used_verts = np.unique(faces_simplified.flatten())

                vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_verts)}

                

                verts_simplified = verts[used_verts]

                faces_simplified = np.array([[vert_map[v] for v in face] for face in faces_simplified])

                

                self.logger.info(f"  Simplified: {len(verts_simplified):,} vertices, {len(faces_simplified):,} faces")

                

                verts = verts_simplified

                faces = faces_simplified

            

            bounds = {

                'xmin': float(verts[:, 0].min()),

                'xmax': float(verts[:, 0].max()),

                'ymin': float(verts[:, 1].min()),

                'ymax': float(verts[:, 1].max()),

                'zmin': float(verts[:, 2].min()),

                'zmax': float(verts[:, 2].max()),

            }

            

            mesh_data = {

                'name': organ_name,

                'label_id': int(organ_label),

                'num_vertices': len(verts),

                'num_faces': len(faces),

                'vertices': verts.tolist(),

                'faces': faces.tolist(),

                'bounds': bounds

            }

            

            return mesh_data

            

        except Exception as e:

            self.logger.error(f"Failed to generate mesh for {organ_name}: {e}")

            return None

    

    def generate_tumor_mesh_by_organ(

        self,

        tumor_volume_path: str,

        tumor_ids: List[int],

        organ_name: str,

        spacing: tuple,

        target_faces: int = 3000

    ) -> Dict:

        self.logger.info(f"Generating tumor mesh for {organ_name} ({len(tumor_ids)} tumors)...")

        

        tumor_nii = nib.load(tumor_volume_path)

        tumor_volume = tumor_nii.get_fdata()

        

        combined_mask = np.zeros_like(tumor_volume, dtype=np.uint8)

        

        # FIX: Volume has float IDs (1.07, 2.14, ...), need to match to int tumor_ids

        # Find closest matches by rounding

        for tumor_id in tumor_ids:

            # Find voxels with ID closest to tumor_id (within 0.6 range)

            # This handles scipy.ndimage.label creating float IDs

            mask = np.abs(tumor_volume - tumor_id) < 0.6

            combined_mask[mask] = 1

        

        num_voxels = np.sum(combined_mask)

        

        if num_voxels == 0:

            self.logger.warning(f"No tumor voxels for {organ_name}")

            return None

        

        self.logger.info(f"  Total tumor voxels: {num_voxels:,}")

        

        try:

            verts, faces, normals, values = measure.marching_cubes(

                combined_mask,

                level=0.5,

                spacing=spacing,

                step_size=1

            )

            

            self.logger.info(f"  Marching Cubes: {len(verts):,} vertices, {len(faces):,} faces")

            

            if len(faces) > target_faces:

                ratio = target_faces / len(faces)

                step = int(1 / ratio)

                

                faces_simplified = faces[::step]

                used_verts = np.unique(faces_simplified.flatten())

                vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_verts)}

                

                verts_simplified = verts[used_verts]

                faces_simplified = np.array([[vert_map[v] for v in face] for face in faces_simplified])

                

                self.logger.info(f"  Simplified: {len(verts_simplified):,} vertices, {len(faces_simplified):,} faces")

                

                verts = verts_simplified

                faces = faces_simplified

            

            mesh_data = {

                'name': f"tumors_{organ_name}",

                'organ': organ_name,

                'num_tumors': len(tumor_ids),

                'num_vertices': len(verts),

                'num_faces': len(faces),

                'vertices': verts.tolist(),

                'faces': faces.tolist()

            }

            

            return mesh_data

            

        except Exception as e:

            self.logger.error(f"Failed to generate tumor mesh for {organ_name}: {e}")

            return None

    

    def generate_all_meshes(

        self,

        organ_volume_path: str,

        tumor_volume_path: str,

        tumor_mapping_path: str,

        output_dir: str

    ):

        self.logger.info("Generating all multi-layer meshes...")

        

        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        

        organ_nii = nib.load(organ_volume_path)

        spacing = organ_nii.header.get_zooms()

        

        with open(tumor_mapping_path, 'r') as f:

            mapping = json.load(f)

        

        organs_with_tumors = mapping['summary']['tumors_by_organ']

        

        ORGAN_LABELS = {

            2: {'name': 'fat', 'color': '#FFD700'},

            3: {'name': 'lung_tissue', 'color': '#87CEEB'},

            4: {'name': 'muscle', 'color': '#CD5C5C'},

            5: {'name': 'liver', 'color': '#8B4513'},

            6: {'name': 'soft_tissue', 'color': '#FFB6C1'},

            7: {'name': 'bone', 'color': '#FFFFFF'},

        }

        

        mesh_catalog = {

            'organs': {},

            'tumors': {}

        }

        

        for label_id, organ_info in ORGAN_LABELS.items():

            organ_name = organ_info['name']

            

            organ_mesh = self.generate_organ_mesh(

                organ_volume_path=organ_volume_path,

                organ_label=label_id,

                organ_name=organ_name,

                spacing=spacing,

                target_faces=5000

            )

            

            if organ_mesh:

                organ_mesh['color'] = organ_info['color']

                

                mesh_file = output_path / f"{organ_name}_mesh.json"

                with open(mesh_file, 'w') as f:

                    json.dump(organ_mesh, f, indent=2)

                

                mesh_catalog['organs'][organ_name] = str(mesh_file)

                self.logger.info(f"  Saved {organ_name} mesh")

            

            if organ_name in organs_with_tumors:

                tumor_ids = organs_with_tumors[organ_name]['tumor_ids']

                

                tumor_mesh = self.generate_tumor_mesh_by_organ(

                    tumor_volume_path=tumor_volume_path,

                    tumor_ids=tumor_ids,

                    organ_name=organ_name,

                    spacing=spacing,

                    target_faces=3000

                )

                

                if tumor_mesh:

                    tumor_file = output_path / f"{organ_name}_tumors_mesh.json"

                    with open(tumor_file, 'w') as f:

                        json.dump(tumor_mesh, f, indent=2)

                    

                    mesh_catalog['tumors'][organ_name] = str(tumor_file)

                    self.logger.info(f"  Saved {organ_name} tumor mesh")

        

        catalog_file = output_path / "mesh_catalog.json"

        with open(catalog_file, 'w') as f:

            json.dump(mesh_catalog, f, indent=2)

        

        self.logger.info(f"\nMesh catalog saved: {catalog_file}")

        self.logger.info(f"Total organ meshes: {len(mesh_catalog['organs'])}")

        self.logger.info(f"Total tumor meshes: {len(mesh_catalog['tumors'])}")





def main():

    logging.basicConfig(

        level=logging.INFO,

        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    )

    

    generator = MultiLayerMeshGenerator()

    

    organ_volume = BASE_DIR / "output/organs_simple/organs_multilabel_hu.nii_refined.nii.gz"

    tumor_volume = BASE_DIR / "output/tumor_organ_mapping/tumors_unique_3d.nii.gz"

    tumor_mapping = BASE_DIR / "output/tumor_organ_mapping/tumor_organ_mapping_improved.json"

    output_dir = BASE_DIR / "output/meshes_multilayer"

    

    print("=" * 80)

    print("Multi-Layer Mesh Generation")

    print("=" * 80)

    print()

    

    generator.generate_all_meshes(

        organ_volume_path=organ_volume,

        tumor_volume_path=tumor_volume,

        tumor_mapping_path=tumor_mapping,

        output_dir=output_dir

    )

    

    print("\n" + "=" * 80)

    print("Complete!")

    print(f"Output: {output_dir}")

    print("=" * 80)





if __name__ == "__main__":

    main()

