import cProfile
import pstats
import torch
import os
import json
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from training import train
from model import PPO
import line_profiler

import warnings

warnings.filterwarnings("ignore")

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["LINE_PROFILE"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "500"


config = {
    "experiment": "LunarLander-v2",
    "comment": "",
    # Environment
    "num_envs": 4,
    "num_epochs": 10000,
    "steps_per_env": 128,
    "use_intrinsic_reward": False,
    "model_saving_interval": 25,
    # Logging
    "reward_sliding_window_size": 32,
    "return_sliding_window_size": 4096,
    "intrinsic_return_window_size": 4096,
    "reward_scaling_factor_max_epoch_sampling": 100,
    # Model
    "hidden_size": 64,
    "use_memory": True,
    # Training
    "start_epoch": 5,
    "eps_clip": 0.1,
    "lr": 3e-4,
    "gamma": 0.99,
    "batches_per_sequence": 4,
    "ppo_epochs": 3,
    "recalculate_returns": False,
    "recalculate_advantages": True,
    "policy_weight": 1.0,
    "value_weight": 1.0,
    "entropy_weight": 0,
    "early_stopping": True,
    "rnd_weight": 0.0,
    "max_kl_div": 0.05,
    "es_restore_model": False,
    "max_grad_norm": 2.0,
    "use_obs_max": True,
    # Truly PPO
    "use_truly_ppo": True,
    "policy_slope": 20,
    # Cyclic learning rate
    "use_scheduler": False,
    "base_lr": 3e-5,
    "max_lr": 1e-3,
    "step_size_up": 1000,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_pytorch_env():
    torch.set_num_threads(32)
    #torch.set_float32_matmul_precision("high")
    torch.set_printoptions(sci_mode=False)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.utils.deterministic.fill_uninitialized_memory = True


def profile_train():
    profiler = line_profiler.LineProfiler()
    profiler.add_function(PPO.train_epochs_bptt_2)
    profiler.enable_by_count()
    train(config, None)
    profiler.print_stats()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_memory", type=bool, default=True)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config["use_memory"] = args.use_memory
    config["comment"] = args.comment + " " + str(args.index)
    set_pytorch_env()
    set_seed(4020)
    config["comment"] = "GePPO Test, v-trace, geppo adv, mc rets"
    writer = SummaryWriter(
       "Records/" + config["experiment"] + "/" + config["comment"], comment=config["comment"]
    )
    writer.add_text("config", json.dumps(config, indent=4))
    writer.add_text("comment", config["comment"])
    # writer = None
    #profile_train()
    train(config, writer)
