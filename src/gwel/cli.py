# src/gwel/cli.py
import os
import json
import yaml
import typer
import pandas as pd
from typing import List
from gwel.dataset import ImageDataset
from gwel.viewer import Viewer
import numpy as np



app = typer.Typer(add_completion = True, help="GWEL CLI tool", invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # For Python <3.8

try:
    VERSION = version("gwel")
except PackageNotFoundError:
    VERSION = "unknown"


ASCII_ART = r"""
   ______              __
  / ____/      _____  / /
 / / __| | /| / / _ \/ / 
/ /_/ /| |/ |/ /  __/ /  
\____/ |__/|__/\___/_/   
                         """


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False, "--version", "-v", help="Show GWEL CLI version"
    ),
):
    """
    Callback function that runs when 'gwel' is called without a subcommand.
    """

    if version_flag:
        typer.secho(f"GWEL CLI version {VERSION}", fg=typer.colors.GREEN, bold=True)
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print(ASCII_ART)
        typer.secho(f"GWEL CLI version {VERSION}", fg=typer.colors.CYAN, bold=True)
        typer.echo("Available commands:")

        # Use rich-like table formatting with Typer
        for name, cmd in ctx.command.commands.items():
            description = cmd.help or cmd.callback.__doc__ or ""
            if description:
                description = description.strip().split("\n")[0]
            typer.secho(f"  {name:<10}", fg=typer.colors.YELLOW, bold=True, nl=False)
            typer.echo(f" {description}")




@app.command()
def view(
    ctx: typer.Context,
    resized_flag: bool = typer.Option(
        False, "--resized", "-r", help="View resized images."),
    detections_flag: bool = typer.Option(
        False, "--detections", "-d", help="Load pre-detected object annotations."), 
    segmentation_flag: bool = typer.Option(
        False, "--segmentation", "-s", help="Load pre-detected segementation."),  
    max_pixels: int = typer.Option(
        800, "--maxpixels", "-p", help="Max Pixels."),
    index: int = typer.Option(
        1, "--index", "-i", help="The index of image to be viewed on opening."),
    contour_thickness: int = typer.Option(
        2, "--thickness", "-t", help="Thickness in pixels of annotations."), 
    col_scheme: bool = typer.Option(
        None, "--col_scheme", "-c", help="Path to color scheme for annotations in YAML format." 
         )
):
    """
    Open the image viewer with images from the current directory.
    """
    directory = os.getcwd()
    
    if not col_scheme:
        col_scheme = os.path.join(directory,".gwel", "colour_scheme.yaml")
    path = col_scheme

    if os.path.exists(path):
        with open(path, "r") as f:
            col_scheme_dict = yaml.safe_load(f) or {}
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        col_scheme_dict = {}
        with open(path, "w") as f:
            yaml.safe_dump(col_scheme_dict, f)


    try:
        dataset = ImageDataset(directory)
        if resized_flag:
            dataset.resize()
        if detections_flag:
            dataset.detect()
        if segmentation_flag:
            dataset.segment() 
        viewer = Viewer(dataset,max_pixels = max_pixels,contour_thickness=contour_thickness,col_scheme=col_scheme_dict)
        viewer.index = index - 1
        if detections_flag:
            viewer.mode = 'instance'
        viewer.open()
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)



@app.command()
def resize(max_pixels: int = typer.Option(
        800, "--maxpixels", "-p", help="Max Pixels.")):
    """
    Create resized copies of the images from the current directory.
    """
    directory = os.getcwd()
    try:
        dataset = ImageDataset(directory)
        dataset.resize(max_size=max_pixels)
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)



@app.command()
def detect(model: str = typer.Argument(...,help="Model type"), 
           weights: str = typer.Argument(...,help="Path to model weights"), 
           slice_size:int = typer.Option(
    None, "--slicesz", "-s", help="Slice size")):
    """
    Run a detector on the images from the current directory.
    """
    directory = os.getcwd()
    try:
        if os.path.exists(weights):
            if model == "YOLOv8":
                from gwel.networks.YOLOv8 import YOLOv8
                if not slice_size:
                    detector = YOLOv8(weights)
                else:
                    detector = YOLOv8(weights,patch_size=(slice_size,slice_size))
            elif model == "YOLOv8seg":
                from gwel.networks.YOLOv8seg import YOLOv8seg
                detector = YOLOv8seg(weights)
            else:
                raise ValueError("Model type unknown.")
        else:
            raise ValueError("No weights found at location {weights}.")
        dataset = ImageDataset(directory)
        dataset.resize()
        dataset.detect(detector)
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)




@app.command()
def segment(model: str = typer.Argument(...,help="Model type"),
            weights: str = typer.Argument(...,help="Path to model weights"), 
            channels: str = typer.Option(os.path.join(".gwel","channels.yaml"),"--channels","-c", help="Segmentation class channels YAML file"), 
            patch_size:int = typer.Option(256, "--patchsz", "-s", help="Patch size"),
            ):
    """
    Run a segmenter on the images from the current directory.
    """
    directory = os.getcwd()
    try:
        if os.path.exists(weights):
            if model == "UNET":
                from gwel.networks.UNET import UNET  
                segmenter = UNET(weights, patch_size, channels)
            else:
                raise ValueError("Model type unknown.")
        else:
            raise ValueError("No weights found at location {weights}.")
        dataset = ImageDataset(directory)
        dataset.segment(segmenter)

    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)

@app.command()
def classify(model: str = typer.Argument(...,help="Model type [Supported: QRreader]"),
            weights: str = typer.Option(None,help="Path to model weights"),
            multiclass: bool = typer.Option(True,'-m','--multiclass', help="Allow multiple classes per image.")):
    """
    Run a classifier on the images in the current directory
    """
    directory = os.getcwd()
    match model:
        case 'QRreader':
            dataset = ImageDataset(directory)
            from gwel.networks.QRreader import QRreader
            classifier = QRreader(merge= not multiclass)
            dataset.classify(classifier)


@app.command()
def crop(path: str = typer.Argument(...,help="Path to output directory.")):
    """
    Crop the images from the current directory based on the bounding boxes of detections.
    """
    directory = os.getcwd()
    try:
        dataset = ImageDataset(directory)
        dataset.detect()
        dataset.crop(path)
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)


@app.command()
def export(protocol :str =typer.Argument(...,help='Protocol used for export.'),
           path: str =typer.Argument(...,help='Path to output file or directory.'),
           zipped: bool = typer.Option(False, '-z','--zipped',help='Export as a zipped file')):
    """
    Export dataset according to a predefined protocol.
    """
    directory = os.getcwd()

    match protocol:
        case 'CSV':
            try:
                dataset = ImageDataset(directory)
                dataset.detect()
                dataset.segment()
                from gwel.protocols.csv import CSV
                exporter = CSV(dataset)
                exporter.export(path) 
            except ValueError as e:
                typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)
    
        case 'YOLO':
            try:
                dataset = ImageDataset(directory)
                dataset.detect()
                from gwel.protocols.yolo import yolo_exporter
                exporter = yolo_exporter(dataset,zipped=zipped)
                exporter.export(path) 
            except ValueError as e:
                typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)

        case 'SEGPOLYS':
            try:
                dataset = ImageDataset(directory)
                dataset.segment()
                dataset.write_segmentation(output_file=path,polys=True)

            except ValueError as e:
                typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)
        case 'SLICE':
            try:
                dataset = ImageDataset(directory)
                slice_size = int(input("Slice size (square):"))
                dataset.slice(slice_size,path)
            except ValueError as e:
                typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)



    





if __name__ == "__main__":
    app()

