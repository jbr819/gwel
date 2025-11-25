# src/gwel/cli.py
import os
import json
import typer
from gwel.dataset import ImageDataset
from gwel.viewer import Viewer



app = typer.Typer(add_completion = True, help="GWEL CLI tool", invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # For Python <3.8

try:
    VERSION = version("gwel")
except PackageNotFoundError:
    VERSION = "unknown"



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
    col_scheme: str = typer.Option(
        {}, "--col_scheme", "-c", help="Color scheme for annotations as JSON string using color names recognized by matplotlib." 
         )
):
    """
    Open the image viewer with images from the current directory.
    """
    directory = os.getcwd()
    try:
        dataset = ImageDataset(directory)
        if resized_flag:
            dataset.resize()
        if detections_flag:
            dataset.detect()
        if segmentation_flag:
            dataset.segment() 
        col_scheme_dict = json.loads(col_scheme) 
        viewer = Viewer(dataset,max_pixels = max_pixels,contour_thickness=contour_thickness,col_scheme=col_scheme_dict)
        viewer.index = index - 1
        if detections_flag:
            viewer.mode = 'instance'
        viewer.open()
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)



@app.command()
def resize():
    """
    Create resized copies of the images from the current directory.
    """
    directory = os.getcwd()
    try:
        dataset = ImageDataset(directory)
        dataset.resize()
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)


@app.command()
def detect(model: str, weights: str, slice_size= typer.Option(
    None, "--slicesz", "-s", help="Slice size (Default: None)")):
    """
    Run a detector on the images from the current directory.
    """
    directory = os.getcwd()
    try:
        if os.path.exists(weights):
            if model == "YOLOv8":
                from gwel.networks.YOLOv8 import YOLOv8
                if slice_size:
                    detector = YOLOv8(weights)
                else:
                    detector = YOLOv8(weights,patch_size=(slice_size,slice_size))
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
def crop(path: str, object_name : str):
    """
    Crop the images from the current directory based on the bounding boxes of detections.
    """
    directory = os.getcwd()
    try:
        dataset = ImageDataset(directory)
        dataset.detect()
        dataset.crop(path,object_name)
    except ValueError as e:
        # Only print the error message, no traceback
        typer.secho(f"Error: {e}", fg=typer.colors.RED, bold=True)


if __name__ == "__main__":
    app()

