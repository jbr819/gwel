import os
import random
import yaml
import shutil
from pathlib import Path
from gwel.protocol import Exporter


class yolo_exporter(Exporter):

    def __init__(self, dataset, zipped: bool =False):
        self.dataset = dataset
        self.zipped = zipped


    def export(
        self,
        path: str,
        bbox: bool = True,
        val_split: float = 0.1,
        seed: int = 42,
    ):

        random.seed(seed)
        path = Path(path)

        for split in ["train", "val"]:
            (path / "images" / split).mkdir(parents=True, exist_ok=True)
            (path / "labels" / split).mkdir(parents=True, exist_ok=True)
        image_paths = list(self.dataset.images)
        random.shuffle(image_paths)

        split_idx = int(len(image_paths) * (1 - val_split))
        train_images = image_paths[:split_idx]
        val_images = image_paths[split_idx:]
        for split, images in [("train", train_images), ("val", val_images)]:
            for image_name in images:
                label_path = path / "labels" / split / f"{image_name.split('.')[0]}.txt"

                detections = self.dataset.object_detections.get(image_name)
                if not detections.get('image_size', None):
                    continue

                
                img_h, img_w = detections["image_size"]


                with open(label_path, "w") as f:
                    if bbox:
                            for box, class_id in zip(detections["bbox"], detections["class_id"]):
                                x_min, y_min, w_box, h_box = box
                                x_center = (x_min + w_box / 2) / img_w
                                y_center = (y_min + h_box / 2) / img_h
                                w_norm = w_box / img_w
                                h_norm = h_box / img_h

                                f.write(f"{class_id-1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                    else:
                        for poly_group, class_id in zip(
                            detections["polygons"],
                            detections["class_id"]
                        ):
                            polygon = poly_group[0]

                            coords = []
                            for x, y in polygon:
                                coords.append(x / img_w)
                                coords.append(y / img_h)

                            coords_str = " ".join(f"{c:.6f}" for c in coords)
                            f.write(f"{class_id-1} {coords_str}\n")

                shutil.copy(
                    image_name,
                    path / "images" / split / image_name
                )

        class_names = self.dataset.object_detections["class_names"]

        yaml_data = {
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": [class_names[i+1] for i in range(len(class_names))]
        }

        with open(path / "data.yaml", "w") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False)

        print("✅ YOLO dataset exported successfully")
        if self.zipped:
            path = Path(path)
            zip_path = path.with_suffix(".zip")

            # Create zip
            shutil.make_archive(
                base_name=str(path),
                format="zip",
                root_dir=str(path),
            )

            # Verify zip was created
            if not zip_path.exists():
                raise RuntimeError("Zip creation failed — original dataset preserved")

            # Remove original directory ONLY after zip success
            shutil.rmtree(path)

            print(f"📦 Dataset zipped to: {zip_path}")


