"""
Distributed Rendering Via PyTorch DDP
Use the following command for testing
```
torchrun --nproc_per_node=8 serial_render.py --config ./configs/serial.conf
```
"""
import os
import sys
import tqdm
import torch
import random
import signal
import natsort
import imageio
import numpy as np
import configargparse
import torch.distributed as dist
import xml.etree.ElementTree as ET
sys.path.append("../build/")
from pyrender import PythonRenderer

from pathlib import Path
from rich.console import Console
from ddp_render import signal_handler, reduce_rendered_image

CONSOLE   = Console(width = 128)
EXIT_FLAG = False
OFFSET_SCALER = 4201

def save_float_image_as_png(image_array: np.ndarray, path: str):
    """
    Save a (H, W, 3) shape np.ndarray to image
    """
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    imageio.imwrite(path, image_array)


def update_density_value_in_xml(xml_path: str, new_density_path: str, new_emission_path: str = ""):
    """
    example use case: update_density_value_in_xml("example.xml", "/path/to/new/density-82.nvdb")
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for medium in root.findall(".//medium[@type='grid']"):  # find all <medium type="grid">
        density_element = medium.find(".//string[@name='density']")
        if density_element is not None:
            density_element.set("value", new_density_path)
        emission_element = medium.find(".//string[@name='emission']")
        if emission_element is not None and new_emission_path:
            emission_element.set("value", new_emission_path)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def update_minmax_time_in_xml(xml_path: str, min_time: float, max_time: float):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for elem in root.findall(".//renderer/float[@name='min_time']"):
        elem.set("value", str(min_time))
    for elem in root.findall(".//renderer/float[@name='max_time']"):
        elem.set("value", str(max_time))
    
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def get_all_nvdb_files_in_folder(folder_path: str, path_prefix: str):
    nvdb_files = []
    if not folder_path: return nvdb_files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nvdb'):
                absolute_path = os.path.join(path_prefix, file)
                nvdb_files.append(absolute_path)
    
    return nvdb_files

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",       is_config_file = True, help='Config file path')
    parser.add_argument("--max_iterations",     type = int, default = 16,   help = "Number of total iterations (render steps)")
    parser.add_argument("--device_id_offset",   type = int, default = 0,      help = "If an offset is needed.")

    parser.add_argument("--scene_path",         type = str, required=True,  help="Path to the scene XML file.")
    parser.add_argument("--volume_path",        type = str, required=True,  help="Path to the VDB volume files.")
    parser.add_argument("--replace_path_dn",    type = str, required=True,  help="Path to substitute (density) for in the xml file.")
    parser.add_argument("--output_path",        type = str, required=True,  help="Path to the output rendering file.")
    parser.add_argument("--output_name",        type = str, required=True,  help="Name of the output image.")
    parser.add_argument("--emission_path",      type = str, default="", help="Path to the VDB volume emission files.")
    parser.add_argument("--replace_path_em",    type = str, default="", help="Path to substitute (emission) for in the xml file.")
    return parser.parse_args()

def ddp_main(local_rank, args, file_index):
    """
    Single process multi-card rendering initialization
    local_rank: GPU rank (0 ~ nproc_per_node-1)
    """

    if not os.path.exists(args.scene_path):
        CONSOLE.log(f"[yellow][WARN] Scene path '{args.scene_path}' does not exist. Exiting... [/yellow]")
        exit(0)

    device = torch.device(f"cuda:{local_rank}")
    dist.barrier()
    renderer = PythonRenderer(args.scene_path, local_rank, local_rank * OFFSET_SCALER + random.randint(0, 1e8))

    is_main_proc = local_rank == args.device_id_offset

    if is_main_proc:
        signal.signal(signal.SIGINT, signal_handler)

    pbar = tqdm.tqdm(
        range(args.max_iterations),
        disable=(local_rank != args.device_id_offset)
    )
    for step in pbar:
        if EXIT_FLAG:
            if is_main_proc:
                CONSOLE.log("Main process exiting due to signaling...")
            break
        renderer.render()

    if local_rank == args.device_id_offset:
        CONSOLE.log("Finalizing by reducing from all the machines.")
    image = renderer.render()

    image_avg, _ = reduce_rendered_image(image, renderer.counter(), device)

    if local_rank == args.device_id_offset:
        image_avg = image_avg.clamp(0, 1)
        image_avg_cpu = image_avg[..., :-1].detach().cpu().numpy()
        output_file = os.path.join(args.output_path, f"{args.output_name}_{file_index:03d}.png")
        save_float_image_as_png(image_avg_cpu, output_file)
        CONSOLE.log(f"Imaged saved to {output_file}.")

    renderer.release()
    dist.barrier()

def job_steady_vdb_serial(args):
    files_dn = get_all_nvdb_files_in_folder(args.volume_path, args.replace_path_dn)
    files_em = get_all_nvdb_files_in_folder(args.emission_path, args.replace_path_em)
    files_dn = natsort.natsorted(files_dn)
    if not files_em:
        files_em = ["" for _ in files_dn]
    else:
        files_em = natsort.natsorted(files_em)
    for i, (file_dn, file_em) in enumerate(zip(files_dn, files_em)):
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == args.device_id_offset:
            update_density_value_in_xml(args.scene_path, file_dn, file_em)
        ddp_main(local_rank + args.device_id_offset, args, i)
        CONSOLE.rule()
        CONSOLE.log(f"({i + 1:3d} / {len(files_dn):3d}) Current file: '{file_dn}' | '{file_em}'")
        CONSOLE.rule()

def job_tof_rendering(args):
    start_time = 0
    time_interval = 0.022222222222222222222222
    frame = 315
    for i in range(frame, -1, -1):
        local_rank = int(os.environ["LOCAL_RANK"])
        time_min = start_time + float(i) * time_interval
        time_max = time_min + time_interval
        if local_rank == args.device_id_offset:
            update_minmax_time_in_xml(args.scene_path, time_min, time_max)
        ddp_main(local_rank + args.device_id_offset, args, i)
        CONSOLE.rule()
        CONSOLE.log(f"({i + 1:3d} / {frame:3d})")
        CONSOLE.rule()

def main(job):
    dist.init_process_group(backend="nccl", init_method="env://")
    args = parse_args()
    job(args)
    dist.destroy_process_group()

if __name__ == "__main__":
    main(job_tof_rendering)