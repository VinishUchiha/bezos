import marlo
from toolz.dicttoolz import merge
from marlo.utils import launch_clients


class MarloEnvMaker():
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.client_pool = launch_clients(num_processes)

    def make_env(self, env_id):
        params = merge(params_default, {
            'client_pool': self.client_pool
        })
        print("{} Minecraft clients available".format(len(self.client_pool)))
        join_token = marlo.make(env_id,
                                params)

        return marlo.init(join_token[0])


resolution = [300, 400]
params_default = {
    "tick_length": 20,
    "prioritise_offscreen_rendering": True,
    'videoResolution': resolution,
    'forceWorldReset': False
}
