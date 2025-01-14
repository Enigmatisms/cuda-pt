"""
Distributed Rendering Via PyTorch DDP
Use the following command for testing
```
torchrun --nproc_per_node=2 ddp_render.py --config ./configs/run.conf
```
"""
import os
import sys
import tqdm
import torch
import signal
import shutil
import configargparse
import torch.distributed as dist
sys.path.append("../build/")
from pyrender import PythonRenderer

from pathlib import Path
from datetime import datetime
from rich.console import Console

CONSOLE   = Console(width = 128)
EXIT_FLAG = False
OFFSET_SCALER = 4201

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

def signal_handler(signum, frame):
    global EXIT_FLAG
    CONSOLE.log(f"[Rank={dist.get_rank()}] Received signal {signum}. Setting EXIT_FLAG to True.")
    EXIT_FLAG = True

def get_summary_writer(name: str, del_dir:bool):
    logdir = './logs/'
    logdir_exist = os.path.exists(logdir)
    if logdir_exist and del_dir:
        shutil.rmtree(logdir)
    elif not logdir_exist:
        os.makedirs(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-{1}/".format(datetime.now(), name)
    return SummaryWriter(log_dir = logdir + time_stamp)

def reduce_rendered_image(img: torch.Tensor, this_spp: int, device, spp_total = None) -> torch.Tensor:
    image_sum   = img * this_spp
    dist.all_reduce(image_sum, op=dist.ReduceOp.SUM)
    if spp_total is None:
        spp_tensor  = torch.tensor([float(this_spp)], dtype=torch.float32, device = device)
        dist.all_reduce(spp_tensor, op=dist.ReduceOp.SUM)
        spp_total = spp_tensor.item()
    if spp_total > 0:
        return image_sum / spp_total, spp_total
    return image_sum, 0

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",       is_config_file = True, help='Config file path')
    parser.add_argument("--local_rank",         type = int, default = 0,      help = "Local rank on the node")
    parser.add_argument("--reduce_interval",    type = int, default = 128,    help = "Interval for reduce/average")
    parser.add_argument("--ft_interval",        type = int, default = 64,     help = "Interval for recording the frame time")
    parser.add_argument("--max_iterations",     type = int, default = 100000, help = "Number of total iterations (render steps)")
    parser.add_argument("--device_id_offset",   type = int, default = 0,      help = "If an offset is needed.")

    parser.add_argument("--scene_path",         type = str, required=True,  help="Path to the scene XML file.")

    parser.add_argument("--rm_logs",            default = False, action = "store_true", help = "Remove previous logs")
    parser.add_argument("--record_var",         default = False, action = "store_true", help = "Record variance curves")
    return parser.parse_args()


def ddp_main(local_rank, args):
    """
    Single process multi-card rendering initialization
    local_rank: GPU rank (0 ~ nproc_per_node-1)
    """
    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    if not os.path.exists(args.scene_path):
        CONSOLE.log(f"[yellow][WARN] Scene path '{args.scene_path}' does not exist. Exiting... [/yellow]")
        exit(0)

    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    renderer = PythonRenderer(args.scene_path, local_rank, local_rank * OFFSET_SCALER)
    scene_name = Path(args.scene_path).stem

    writer = get_summary_writer(scene_name, args.rm_logs)

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
        image = renderer.render()

        if (step + 1) % args.reduce_interval == 0:
            image_avg, spp_total = reduce_rendered_image(image, renderer.counter(), device)

            if local_rank == args.device_id_offset:
                image_avg = image_avg.clamp(0, 1)
                image_avg_cpu = image_avg[..., :-1].detach().cpu().permute(2, 0, 1)  # (3, H, W)
                writer.add_image("render/Image", image_avg_cpu, global_step = 0)

            if args.record_var:
                var_image: torch.Tensor = renderer.variance()
                if var_image.numel() is not None:
                    var_avg, _ = reduce_rendered_image(var_image, renderer.counter(), device, spp_total)
                    if local_rank == args.device_id_offset:
                        writer.add_scalar("variance/Average Variance", var_avg.mean() * world_size / spp_total, step+1)

                        var_avg /= torch.quantile(var_avg, 0.99)
                        var_avg  = var_avg.detach().cpu().permute(2, 0, 1)  # (1, H, W)
                        writer.add_image("variance/Image", var_avg, global_step = 0)
        
        if (step + 1) % args.ft_interval == 0:
            local_ft = renderer.avg_frame_time() 
            ft_tensor = torch.tensor([local_ft], dtype=torch.float32, device=device)

            gather_list = [torch.zeros_like(ft_tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, ft_tensor)

            if local_rank == args.device_id_offset: 
                writer.add_scalars(
                    "frame_time/all_ranks",
                    {f"Frame Time/rank-{i}": ft_val for i, ft_val in enumerate(gather_list)},
                    global_step = step + 1
                )

                all_ft = torch.stack(gather_list)
                avg_ft = all_ft.mean().item()
                writer.add_scalar("Frame Time/AVG", avg_ft, step+1)
    
    renderer.release()
    dist.destroy_process_group()

def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    ddp_main(local_rank + args.device_id_offset, args)


if __name__ == "__main__":
    main()