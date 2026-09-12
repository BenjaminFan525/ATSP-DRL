"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait as wait_connections
from abc import ABC, abstractmethod
from onpolicy.utils.util import tile_images

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob, _, _ = env.reset()
            # else:
            #     if np.all(done):
            #         ob, _, _ = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'get_graph':
            ob = env.get_graph()
            remote.send((ob))
        elif cmd == 'render':
            env.render_all(save_path=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'get_nums_fields':
            remote.send(env.num_fields)
        elif cmd == 'get_rewards':
            remote.send(env.calculate_hindsight_rewards())
        elif cmd == 'get_episode_rewards':
            remote.send(env._get_episode_rewards())
        elif cmd == 'call':
            method_name, args, kwargs = data
            remote.send(getattr(env, method_name)(*args, **kwargs))
        elif cmd == 'shuffer_data':
            env.shuffer_data()
        else:
            raise NotImplementedError


class GuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # could cause zombie process
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), rews, np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

from torch_geometric.loader.dataloader import Batch
class GraphSubprocVecEnv(ShareVecEnv):
    def __init__(
        self,
        env_fns,
        spaces=None,
        ipc_timeout_seconds=300.0,
        async_graph_clone_workers=0,
    ):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self._pending_remote_indices = set()
        self.ipc_timeout_seconds = float(ipc_timeout_seconds)
        if self.ipc_timeout_seconds <= 0.0:
            raise ValueError("ipc_timeout_seconds must be positive")
        self.async_graph_clone_workers = int(async_graph_clone_workers)
        if self.async_graph_clone_workers < 0:
            raise ValueError("async_graph_clone_workers must be non-negative")
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # Workers must be forked before any parent-side threads are created.
        # The executor only clones already-received CPU tensors and never
        # touches environment state or random-number generators.
        self._graph_clone_executor = None
        self.resume_async_graph_cloning()

        
        ShareVecEnv.__init__(self, len(env_fns), None, None, None)

    def step_async(self, actions):
        self._ensure_idle("step_async")
        sent_indices = set()
        try:
            for index, (remote, action) in enumerate(zip(self.remotes, actions)):
                remote.send(('step', action.tolist()))
                sent_indices.add(index)
        finally:
            self._pending_remote_indices = sent_indices
            self.waiting = bool(sent_indices)

    @staticmethod
    def _clone_graph_payload(payload):
        """Detach graph tensors from multiprocessing shared-memory handles.

        A 20-worker HKBZ observation contains roughly 600 tensor storages.
        Retaining the previous shared observation while receiving the next can
        exceed the default 1024-FD soft limit.  Cloning immediately makes the
        graph private to the parent and releases the IPC descriptors before
        the next vector step.
        """
        if isinstance(payload, tuple) and payload:
            observation = payload[0]
            clone = getattr(observation, 'clone', None)
            if callable(clone):
                observation = clone()
            return (observation, *payload[1:])
        clone = getattr(payload, 'clone', None)
        return clone() if callable(clone) else payload

    def _recv_all(self, clone_graph_payload=False):
        pending = set(self._pending_remote_indices)
        if not pending:
            return []

        # Test doubles and legacy remote implementations may not expose a file
        # descriptor.  Keep their deterministic sequential behavior.
        if not all(callable(getattr(self.remotes[i], 'fileno', None)) for i in pending):
            results = []
            try:
                for index in sorted(pending):
                    remote = self.remotes[index]
                    try:
                        result = remote.recv()
                    finally:
                        self._pending_remote_indices.discard(index)
                    if clone_graph_payload:
                        result = self._clone_graph_payload(result)
                    results.append(result)
            finally:
                if not self._pending_remote_indices:
                    self.waiting = False
            return results

        deadline = time.monotonic() + self.ipc_timeout_seconds
        results = {}
        clone_executor = getattr(self, '_graph_clone_executor', None)
        remote_index = {id(self.remotes[i]): i for i in pending}
        try:
            while pending:
                dead = [
                    index for index in sorted(pending)
                    if not self.ps[index].is_alive()
                    and not self.remotes[index].poll()
                ]
                if dead:
                    details = [
                        {
                            'index': index,
                            'pid': self.ps[index].pid,
                            'exitcode': self.ps[index].exitcode,
                        }
                        for index in dead
                    ]
                    raise RuntimeError(
                        f"Vector-environment workers exited before replying: {details}"
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    details = [
                        {
                            'index': index,
                            'pid': self.ps[index].pid,
                            'alive': self.ps[index].is_alive(),
                            'exitcode': self.ps[index].exitcode,
                        }
                        for index in sorted(pending)
                    ]
                    raise TimeoutError(
                        "Timed out waiting for vector-environment IPC replies "
                        f"after {self.ipc_timeout_seconds:.1f}s: {details}"
                    )

                ready = wait_connections(
                    [self.remotes[index] for index in pending],
                    timeout=min(1.0, remaining),
                )
                for remote in ready:
                    index = remote_index[id(remote)]
                    try:
                        result = remote.recv()
                    except BaseException as error:
                        raise RuntimeError(
                            "Vector-environment IPC receive failed for "
                            f"worker index={index}, pid={self.ps[index].pid}, "
                            f"exitcode={self.ps[index].exitcode}: {error}"
                        ) from error
                    finally:
                        pending.discard(index)
                        self._pending_remote_indices.discard(index)
                    if clone_graph_payload and clone_executor is not None:
                        result = clone_executor.submit(
                            self._clone_graph_payload, result
                        )
                    elif clone_graph_payload:
                        result = self._clone_graph_payload(result)
                    results[index] = result
        finally:
            if not self._pending_remote_indices:
                self.waiting = False
        ordered = []
        for index in sorted(results):
            result = results[index]
            if clone_graph_payload and clone_executor is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for asynchronous graph cloning "
                        f"after {self.ipc_timeout_seconds:.1f}s."
                    )
                try:
                    result = result.result(timeout=remaining)
                except FutureTimeoutError as error:
                    raise TimeoutError(
                        "Timed out waiting for asynchronous graph cloning "
                        f"after {self.ipc_timeout_seconds:.1f}s."
                    ) from error
            ordered.append(result)
        return ordered

    def _ensure_idle(self, operation):
        if self._pending_remote_indices:
            raise RuntimeError(
                f"Cannot start {operation} while replies are pending for "
                f"workers={sorted(self._pending_remote_indices)}."
            )

    def _send_all(self, command, data=None):
        self._ensure_idle(command)
        sent_indices = set()
        try:
            for index, remote in enumerate(self.remotes):
                remote.send((command, data))
                sent_indices.add(index)
        finally:
            self._pending_remote_indices = sent_indices
            self.waiting = bool(sent_indices)

    def _request_all(self, command, data=None, clone_graph_payload=False):
        self._send_all(command, data)
        return self._recv_all(clone_graph_payload=clone_graph_payload)

    def step_wait(self):
        results = self._recv_all(clone_graph_payload=True)
        obs, rews, dones, infos = zip(*results)
        return obs, np.stack(rews), np.stack(dones), self.stack_infos(infos)

    def reset(self):
        self._send_all('reset')
        results = self._recv_all(clone_graph_payload=True)
        obs, dones, infos = zip(*results)
        return obs, np.stack(dones), self.stack_infos(infos)

    def get_graph(self):
        return self._request_all('get_graph', clone_graph_payload=True)

    def shuffer_data(self):
        for remote in self.remotes:
            remote.send(('shuffer_data', None))

    def get_rewards(self):
        return self._request_all('get_rewards')

    def get_episode_rewards(self):
        return np.stack(self._request_all('get_episode_rewards'))

    def call(self, method_name, *args, **kwargs):
        return self._request_all('call', (method_name, args, kwargs))

    def call_each(self, method_name, args_by_worker, kwargs_by_worker=None):
        """Call one environment method with worker-specific arguments.

        Shared evaluators reuse a fixed process pool across training seeds.
        Reseeding must preserve the historical ``seed + rank`` mapping rather
        than broadcasting one seed to every worker.
        """
        args_by_worker = list(args_by_worker)
        if len(args_by_worker) != len(self.remotes):
            raise ValueError(
                "call_each argument count must match worker count: "
                f"args={len(args_by_worker)}, workers={len(self.remotes)}"
            )
        if kwargs_by_worker is None:
            kwargs_by_worker = [{} for _ in self.remotes]
        else:
            kwargs_by_worker = list(kwargs_by_worker)
        if len(kwargs_by_worker) != len(self.remotes):
            raise ValueError(
                "call_each keyword count must match worker count: "
                f"kwargs={len(kwargs_by_worker)}, workers={len(self.remotes)}"
            )
        self._ensure_idle('call_each')
        sent_indices = set()
        try:
            for index, (remote, worker_args, worker_kwargs) in enumerate(zip(
                self.remotes, args_by_worker, kwargs_by_worker
            )):
                remote.send((
                    'call',
                    (method_name, tuple(worker_args), dict(worker_kwargs)),
                ))
                sent_indices.add(index)
        finally:
            self._pending_remote_indices = sent_indices
            self.waiting = bool(sent_indices)
        return self._recv_all()

    def call_async(self, method_name, *args, **kwargs):
        """Start an ordered read-only environment RPC batch."""
        self._send_all('call', (method_name, args, kwargs))

    def call_wait(self):
        """Finish the RPC started by :meth:`call_async` in worker order."""
        if not self._pending_remote_indices:
            raise RuntimeError("call_wait requires a pending asynchronous call")
        return self._recv_all()

    def suspend_async_graph_cloning(self):
        """Stop parent clone threads before lazily forking another env pool."""
        self._ensure_idle("suspend_async_graph_cloning")
        clone_executor = getattr(self, '_graph_clone_executor', None)
        if clone_executor is None:
            return False
        clone_executor.shutdown(wait=True, cancel_futures=True)
        self._graph_clone_executor = None
        return True

    def resume_async_graph_cloning(self):
        """Restart the parent-only clone executor after a safe fork window."""
        if getattr(self, 'closed', False):
            raise RuntimeError("Cannot resume graph cloning on a closed env pool")
        if getattr(self, '_graph_clone_executor', None) is not None:
            return False
        if int(getattr(self, 'async_graph_clone_workers', 0)) <= 0:
            return False
        self._graph_clone_executor = ThreadPoolExecutor(
            max_workers=self.async_graph_clone_workers,
            thread_name_prefix="hkbz-graph-clone",
        )
        return True
    
    def stack_infos(self, infos):
        stacked_infos = {}
        for key in infos[0].keys():
            stacked_infos[key] = np.stack([info[key] for info in infos])
        return stacked_infos

    def reset_task(self):
        return np.stack(self._request_all('reset_task'))

    def close(self):
        if self.closed:
            return
        # A worker with a pending reply may be blocked inside tensor
        # serialization.  Do not queue a close command behind that reply.
        for index, remote in enumerate(self.remotes):
            if index in self._pending_remote_indices:
                continue
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 10.0
        for process in self.ps:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.ps:
            if process.is_alive():
                process.terminate()
        for process in self.ps:
            process.join(timeout=5.0)
        for process in self.ps:
            if process.is_alive() and hasattr(process, 'kill'):
                process.kill()
        for process in self.ps:
            process.join(timeout=2.0)
        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass
        clone_executor = getattr(self, '_graph_clone_executor', None)
        if clone_executor is not None:
            clone_executor.shutdown(wait=True, cancel_futures=True)
            self._graph_clone_executor = None
        self._pending_remote_indices.clear()
        self.waiting = False
        self.closed = True

    def render(self, save_dir):
        for idx, remote in enumerate(self.remotes):
            remote.send(('render', save_dir + f"/env_{idx}.png"))

class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseguardworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseGuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseguardworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError



class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class ChooseDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class ChooseSimpleDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        obs = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
