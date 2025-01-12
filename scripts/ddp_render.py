import os
import sys
import torch
import shutil
import datetime
import configargparse
import torch.distributed as dist
sys.path.append("../build/Release")
from pyrender import PythonRenderer

from pathlib import Path
from rich.console import Console
CONSOLE = Console(width = 128)

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

def get_summary_writer(name: str, del_dir:bool):
    logdir = './logs/'
    logdir_exist = os.path.exists(logdir)
    if logdir_exist and del_dir:
        shutil.rmtree(logdir)
    elif not logdir_exist:
        os.makedirs(logdir_exist)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-{name}/".format(datetime.now(), name)
    return SummaryWriter(log_dir = logdir + time_stamp)

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--local_rank",         type = int, default = 0,      help = "Local rank on the node")
    parser.add_argument("--reduce_interval",    type = int, default = 200,    help = "Interval for reduce/average")
    parser.add_argument("--ft_interval",        type = int, default = 64,     help = "Interval for recording the frame time")
    parser.add_argument("--max_iterations",     type = int, default = 100000, help = "Number of total iterations (render steps)")
    parser.add_argument("--device_id_offset",   type = int, default = 0,      help = "If an offset is needed.")
    parser.add_argument("--log_dir", type=str, default="./tb_logs",
                        help="Directory to store tensorboard logs")

    parser.add_argument("--scene_path",         type = str, required=True,  help="Path to the scene XML file.")

    parser.add_argument("--rm_logs",            default = False, action = "store_true", help = "Remove previous logs")
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
    renderer = PythonRenderer(args.scene_path, local_rank)
    scene_name = Path(args.scene_path).stem

    writer = get_summary_writer(scene_name, args.rm_logs)

    for step in range(args.max_iterations):
        image = renderer.render(
            max_bounces=6,
            max_diffuse=4,
            max_specular=2,
            max_transmit=4,
            gamma_corr=False
        )

        # reduce
        if (step + 1) % args.reduce_interval == 0:
            current_spp = renderer.counter() 
            spp_tensor  = torch.tensor([float(current_spp)], dtype=torch.float32, device=device)
            image_sum   = image * spp_tensor
            dist.all_reduce(image_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(spp_tensor, op=dist.ReduceOp.SUM)

            spp_total = spp_tensor.item()
            if spp_total > 0:
                image_avg: torch.Tensor = image_sum / spp_total
            else:
                image_avg: torch.Tensor = image_sum

            if local_rank == args.device_id_offset:
                image_avg = image_avg.clamp(0, 1)
                image_avg_cpu = image_avg[..., :-1].detach().cpu().permute(2, 1, 0)  # (3, H, W)
                writer.add_image("render/Image", image_avg_cpu, global_step = 0)
        
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

    dist.destroy_process_group()

def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    ddp_main(local_rank + args.device_id_offset, args)


if __name__ == "__main__":
    main()