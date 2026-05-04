import multiprocessing
from .engine import DroneSimulation
from src.algorithms.rewards import compute_detect_density
from src.utils.geometry import get_cameras_pointers

_worker_sim_instance = None

def init_worker(config_path):
    global _worker_sim_instance
    _worker_sim_instance = DroneSimulation(config_path, [], [])

def run_single_simulation(args):
    global _worker_sim_instance
    (cdfc_state, cfc_state, camera_range, sim_level) = args

    _worker_sim_instance.update_launch_pads(cdfc_state)
    _worker_sim_instance.update_cameras(cfc_state)
    _worker_sim_instance.level = sim_level
    _worker_sim_instance.init_level()

    # run the simulation
    cdfc_rewards, cfc_rewards, cdfc_scores, cfc_scores, dump_float = _worker_sim_instance.simulate()

    # compute penalty
    penalty = compute_detect_density(cfc_state, camera_range, 30, get_cameras_pointers)

    return cfc_rewards - penalty


def cfc_mean_rewards_parallel(iterations, config_path, cdfc_state, cfc_state,
                                 camera_range, sim_level):
    total_cores = multiprocessing.cpu_count()
    # make sure the parallel process does not need more cores than available
    num_processes = max(1, total_cores - 2)
    if iterations < num_processes:
        num_processes = iterations

    worker_args = [
        (cdfc_state, cfc_state, camera_range, sim_level)
        for _ in range(iterations)
    ]

    # run simulations in parallel
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(config_path,)) as pool:
        cfc_rewards_mean = pool.map(run_single_simulation, worker_args)

    mean_reward = sum(cfc_rewards_mean) / len(cfc_rewards_mean)
    return mean_reward