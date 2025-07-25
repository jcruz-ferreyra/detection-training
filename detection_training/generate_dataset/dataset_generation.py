import logging
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET

from PIL import Image
import supervision as sv
from supervision.dataset.utils import build_class_index_mapping, map_detections_class_id
import yaml

from .types import DatasetGenerationContext

logger = logging.getLogger(__name__)


def _validate_output_directory(ctx: DatasetGenerationContext):
    """Validate that output directory doesn't already exist."""

    if ctx.output_dir.exists():
        logger.error(f"Directory already exists: {ctx.output_dir}")
        raise ValueError(
            f"Directory already exists: {ctx.output_dir}. "
            "Please remove it before running or choose a different name."
        )

    logger.info("Output directory validation passed")


def _fix_pascal_voc_annotations(img_dir: Path, ann_dir: Path) -> None:
    """Fix Pascal VOC annotation coordinate and format issues."""
    logger.info(f"Fixing annotations in: {ann_dir}")

    def _clamp(val: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(val, max_val))

    fixed_count = 0
    error_count = 0
    total_files = len(list(ann_dir.glob("*.xml")))

    for xml_file in ann_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_tag = root.find("filename")

            # Check for filename tag
            filename_tag = root.find("filename")
            if filename_tag is None:
                logger.warning(f"Skipping {xml_file.name}: no filename tag")
                error_count += 1
                continue

            # Check if corresponding image exists
            image_path = img_dir / filename_tag.text
            if not image_path.exists():
                logger.warning(f"Image {filename_tag.text} not found for {xml_file.name}")
                error_count += 1
                continue

            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logger.warning(f"Failed to open image {filename_tag.text}: {e}")
                error_count += 1
                continue

            # Fix bounding box coordinates
            modified = False
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                for tag in ["xmin", "ymin", "xmax", "ymax"]:
                    coord_elem = bndbox.find(tag)
                    if coord_elem is None:
                        continue

                    val = coord_elem.text.strip()
                    try:
                        int_val = int(float(val))  # allow float-to-int conversion
                    except ValueError:
                        logger.warning(f"{xml_file.name}: Invalid {tag} value: {val}")
                        continue

                    # Clamp the coordinates to image bounds
                    if tag in ("xmin", "xmax"):
                        clamped = _clamp(int_val, 0, img_width - 1)
                    else:  # ymin, ymax
                        clamped = _clamp(int_val, 0, img_height - 1)

                    if clamped != int_val or val != str(int_val):
                        modified = True
                        coord_elem.text = str(clamped)

            # Save modified file
            if modified:
                tree.write(xml_file, encoding="utf-8", xml_declaration=True)
                print(f"Fixed: {xml_file.name}")
                fixed_count += 1

        except ET.ParseError as e:
            logger.error(f"Failed to parse {xml_file.name}: {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {xml_file.name}: {e}")
            error_count += 1

    logger.info(
        f"Annotation fixing completed: {fixed_count} fixed, {error_count} errors, {total_files} total"
    )


def _fix_input_datasets_annotations(ctx: DatasetGenerationContext) -> None:
    """Fix annotations across all input datasets."""
    logger.info("Fixing annotations across all input datasets")

    total_datasets = sum(len(datasets) for datasets in ctx.split_datasets.values())
    logger.info(f"Processing {total_datasets} datasets")

    processed_count = 0

    for split_name, datasets in ctx.split_datasets.items():
        logger.info(f"Processing {split_name} datasets: {len(datasets)} folders")

        for dataset_folder in datasets:
            img_dir = ctx.input_dir / dataset_folder / "images"
            ann_dir = ctx.input_dir / dataset_folder / "annotations"

            # Validate directories exist
            if not img_dir.exists():
                logger.warning(f"Images directory not found: {img_dir}")
                continue
            if not ann_dir.exists():
                logger.warning(f"Annotations directory not found: {ann_dir}")
                continue

            logger.debug(f"Processing dataset: {dataset_folder}")
            _fix_pascal_voc_annotations(img_dir, ann_dir)
            processed_count += 1

    logger.info(f"Completed fixing annotations for {processed_count}/{total_datasets} datasets")


def _standardize_class_indices(
    ctx: DatasetGenerationContext, ds: sv.DetectionDataset
) -> sv.DetectionDataset:
    """Standardize class indices across datasets to ensure consistent YOLO class mapping."""
    logger.debug(
        f"Standardizing class indices from {len(ds.classes)} to {len(ctx.target_classes)} classes"
    )

    # Build mapping from source to target class indices
    class_index_mapping = build_class_index_mapping(
        source_classes=ds.classes, target_classes=ctx.target_classes
    )

    # Create new annotations dict with remapped class indices
    new_annotations = {}
    for image_path in ds.image_paths:
        new_annotations[image_path] = map_detections_class_id(
            source_to_target_mapping=class_index_mapping,
            detections=ds.annotations[image_path],
        )

    logger.debug(f"Remapped annotations for {len(new_annotations)} images")

    return sv.DetectionDataset(
        classes=ctx.target_classes,
        images=ds.image_paths,
        annotations=new_annotations,
    )


def _load_pascal_voc_datasets(ctx: DatasetGenerationContext) -> Dict[str, sv.DetectionDataset]:
    """Load and merge Pascal VOC datasets by split."""
    logger.info("Loading Pascal VOC datasets by split")

    split_ds = {}

    for split_name, datasets in ctx.split_datasets.items():
        logger.info(f"Loading {split_name} split: {len(datasets)} datasets")

        ds_list = []
        total_images = 0

        for dataset_folder in datasets:
            logger.debug(f"Loading dataset: {dataset_folder}")

            img_dir = ctx.input_dir / dataset_folder / "images"
            ann_dir = ctx.input_dir / dataset_folder / "annotations"

            # Validate directories exist
            if not img_dir.exists():
                logger.warning(f"Images directory not found: {img_dir}")
                continue
            if not ann_dir.exists():
                logger.warning(f"Annotations directory not found: {ann_dir}")
                continue

            try:
                # Load dataset and append to split's datasets list
                ds = sv.DetectionDataset.from_pascal_voc(
                    images_directory_path=str(img_dir),
                    annotations_directory_path=str(ann_dir),
                )

                ds_list.append(ds)
                total_images += len(ds)
                logger.debug(f"Loaded {len(ds)} images from {dataset_folder}")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_folder}: {e}")
                continue

        # Merge datasets for current split
        if ds_list:
            try:
                merged_ds = sv.DetectionDataset.merge(ds_list)
                merged_ds = _standardize_class_indices(ctx, merged_ds)

                split_ds[split_name] = merged_ds
                logger.info(f"Merged {split_name} split: {total_images} total images")

            except Exception as e:
                logger.error(f"Failed to merge {split_name} datasets: {e}")
                split_ds[split_name] = sv.DetectionDataset.empty()
        else:
            logger.warning(f"No valid datasets found for {split_name} split")
            split_ds[split_name] = sv.DetectionDataset.empty()

    logger.info(f"Successfully loaded {len(split_ds)} splits")
    for split_name, ds in split_ds.items():
        logger.info(f"  {split_name}: {len(ds)} images")

    return split_ds


def _save_dataset_as(ctx, split_ds):
    """Export datasets to specified format (YOLO or COCO) with proper directory structure."""
    logger.info(f"Saving datasets in {ctx.output_format} format")

    if ctx.output_format == "yolo":
        yaml_path = str(ctx.output_dir / "data.yaml")
        path_args = {"path": "."}

        for split_name, ds in split_ds.items():
            img_output_dir = ctx.output_dir / "images" / split_name
            ann_output_dir = ctx.output_dir / "labels" / split_name

            try:
                ds.as_yolo(
                    images_directory_path=str(img_output_dir),
                    annotations_directory_path=str(ann_output_dir),
                    data_yaml_path=yaml_path,
                )

                path_args[split_name] = f"images/{split_name}"
                logger.debug(f"Successfully exported {split_name} split")

            except Exception as e:
                logger.error(f"Failed to export {split_name} split: {e}")
                raise

        # Update data.yaml with correct paths
        try:
            with open(yaml_path, "r") as f:
                original_yaml = yaml.safe_load(f)

            original_yaml.update(path_args)

            if "val" not in original_yaml and "train" in original_yaml:
                original_yaml["val"] = original_yaml["train"]

            with open(yaml_path, "w") as f:
                yaml.dump(original_yaml, f, default_flow_style=False)

            logger.info(f"Updated data.yaml with paths: {list(path_args.keys())}")

        except Exception as e:
            logger.error(f"Failed to update data.yaml: {e}")
            raise

        logger.info("YOLO format export completed successfully")

    elif ctx.output_format == "coco":
        for split_name, ds in split_ds.items():
            logger.info(f"Exporting {split_name} split: {len(ds)} images")

            # Create output directories
            coco_output_dir = ctx.output_dir / "coco"
            img_output_dir = coco_output_dir / "images" / split_name
            ann_output_dir = coco_output_dir / "annotations"

            try:
                ds.as_coco(
                    images_directory_path=str(img_output_dir),
                    annotations_path=str(ann_output_dir / f"{split_name}.json"),
                )

                logger.debug(f"Successfully exported {split_name} split")

            except Exception as e:
                logger.error(f"Failed to export {split_name} split to COCO: {e}")
                raise

        logger.info("COCO format export completed successfully")

    else:
        logger.error(f"Unsupported output format: {ctx.output_format}")
        raise ValueError(
            f"Unsupported output format: {ctx.output_format}. Supported formats: 'yolo', 'coco'"
        )


def _zip_dataset(ctx: DatasetGenerationContext) -> None:
    """Create a compressed zip archive of the generated dataset for drive storage."""
    logger.info(f"Creating zip archive of dataset: {ctx.output_dir}")

    # Create zip file path (same directory as output, with .zip extension)
    zip_path = ctx.output_dir.parent / f"{ctx.output_dir.name}.zip"

    try:
        import shutil

        # Create zip archive of the entire output directory
        shutil.make_archive(
            base_name=str(zip_path.with_suffix("")),  # Remove .zip extension (added automatically)
            format="zip",
            root_dir=str(ctx.output_dir),
            base_dir=".",
        )

        # Get file sizes for logging
        original_size = sum(f.stat().st_size for f in ctx.output_dir.rglob("*") if f.is_file())
        zip_size = zip_path.stat().st_size
        compression_ratio = (1 - zip_size / original_size) * 100 if original_size > 0 else 0

        logger.info("Dataset archived successfully:")
        logger.info(f"  Archive: {zip_path}")
        logger.info(f"  Original size: {original_size / (1024*1024):.1f} MB")
        logger.info(f"  Compressed size: {zip_size / (1024*1024):.1f} MB")
        logger.info(f"  Compression: {compression_ratio:.1f}%")

    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise


def generate_dataset(ctx: DatasetGenerationContext) -> None:
    """
    Generate training dataset by combining and processing input datasets.

    Args:
        ctx: DatasetGenerationContext containing input/output paths and configuration
    """
    logger.info("Starting dataset generation process")

    # Validate setup
    _validate_output_directory(ctx)

    # Fix any annotation issues in input datasets
    _fix_input_datasets_annotations(ctx)

    split_ds = _load_pascal_voc_datasets(ctx)

    _save_dataset_as(ctx, split_ds)

    if ctx.output_storage == "drive":
        _zip_dataset(ctx)
