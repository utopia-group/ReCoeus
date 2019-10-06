from collections import deque, namedtuple
import json
from statistics import mean
import torch

from event_logger import EventLogger
from model import Policy, EPS

Parameters = namedtuple(
    'Parameters',
    [
        'seed',
        'num_training',
        'num_episodes',
        'batch_size',
        'restart_count',
        'discount_factor',
        'learning_rate',
        'tracking_window',
        'save_interval',
    ]
)

RolloutTrace = namedtuple(
    'RolloutTrace',
    [
        'actions',
        'success',
        'log_probs',
        'rewards',
    ]
)


RolloutStats = namedtuple(
    'RolloutStats',
    [
        'success',
        'length',
        'action_count',
    ]
)


def random_int(low, high):
    return torch.randint(low, high, (1,))[0].item()


class StatsTracker:
    def __init__(self, window_size):
        self.episode_history = deque(maxlen=window_size)

    def _current_episode(self):
        return self.episode_history[-1]

    def _rollout_history(self):
        return [y for x in self.episode_history for y in x]

    def track(self, stats):
        self._current_episode().append(stats)

    def new_episode(self):
        self.episode_history.append([])

    def success_rate(self):
        rollout_history = self._rollout_history()
        if len(rollout_history) == 0:
            return 0.0
        success_count = sum(1 for stats in rollout_history if stats.success)
        return float(success_count) / float(len(rollout_history))

    def average_length(self):
        rollout_history = self._rollout_history()
        if len(rollout_history) == 0:
            return 0.0
        return mean(map(lambda x: x.length, rollout_history))

    def average_action_diversity(self):
        rollout_history = self._rollout_history()
        if len(rollout_history) == 0:
            return 0.0
        return mean(map(lambda x: x.action_count, rollout_history))


class Reinforce:

    def __init__(self, env, policy, params, root_dir=None):
        self.env = env
        self.policy = policy
        self.params = params
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=params.learning_rate)
        self.root_dir = root_dir
        self.event_logger = EventLogger(root_dir)
        self.stats_tracker = StatsTracker(params.tracking_window)
        torch.manual_seed(params.seed)

    def save_policy(self, tag):
        if self.root_dir is not None:
            model_path = self.root_dir / f'policy_{tag}.pt'
            Policy.save(self.policy, model_path)
            self.event_logger.info(f'Current model saved to {model_path}')

    def rollout(self, bench_id):
        actions, log_probs, rewards = [], [], []

        if bench_id is None:
            resp = self.env.restart_rollout()
        else:
            resp = self.env.start_rollout(bench_id)

        while 'reward' not in resp:
            features = resp['features']
            self.event_logger.debug('Current Features: {}'.format(features))
            available_actions = resp['available_actions']
            self.event_logger.debug('Available Actions: {}'.format(available_actions))

            next_action, log_prob = self.policy.sample_action_with_log_probability(
                features, available_actions)
            actions.append(next_action)
            log_probs.append(log_prob)
            # We don't get any reward until the end
            rewards.append(0)

            self.event_logger.debug(f'Taking action {next_action}')
            resp = self.env.take_action(next_action)

        assert len(actions) > 0
        if resp['reward'] == 1:
            success = True
            self.event_logger.info('Rollout succeeded')
            # Slightly favors proofs with shorter length
            # Slightly favors proofs with diverse actions
            reward = 1 + 1 / (len(actions) ** 0.1) + 0.01 * (len(set(actions)) ** 0.5)
        else:
            success = False
            self.event_logger.info('Rollout failed')
            reward = -0.01
        self.event_logger.info(f'Final reward = {reward}')
        rewards[-1] = reward
        return RolloutTrace(actions, success, log_probs, rewards)

    def optimize_loss(self, rollout_traces):
        batch_rewards = []
        batch_log_probs = []
        for rollout_trace in rollout_traces:
            rewards = []
            cumulative_reward = 0
            for reward in reversed(rollout_trace.rewards):
                cumulative_reward = reward + self.params.discount_factor * cumulative_reward
                rewards.append(cumulative_reward)
            batch_rewards.extend(reversed(rewards))
            batch_log_probs.extend(rollout_trace.log_probs)

        reward_tensor = torch.FloatTensor(batch_rewards)
        reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + EPS)
        losses = []
        for log_prob, reward in zip(batch_log_probs, reward_tensor):
            losses.append(-log_prob.reshape(1) * reward)
        total_loss = torch.cat(losses).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def batch_rollout(self, episode_id):
        def single_rollout(rollout_id, restart_id):
            if restart_id > 0:
                self.event_logger.info(f'Restart #{restart_id} on previous benchmark')
                rollout_trace = self.rollout(None)
            else:
                bench_id = random_int(0, self.params.num_training)
                self.event_logger.info(f'Start rollout on benchmark {bench_id}')
                rollout_trace = self.rollout(bench_id)
            return rollout_trace

        rollout_traces = []
        for rollout_id in range(self.params.batch_size):
            self.event_logger.info(f'Batching rollout {rollout_id}...')
            for restart_id in range(self.params.restart_count):
                rollout_trace = single_rollout(rollout_id, restart_id)
                rollout_traces.append(rollout_trace)
                rollout_stats = RolloutStats(
                    length=len(rollout_trace.actions),
                    success=(rollout_trace.success),
                    action_count=len(set(rollout_trace.actions)),
                )
                self.stats_tracker.track(rollout_stats)
                if rollout_trace.success:
                    break
        return rollout_traces

    def train_episode(self, episode_id):
        self.event_logger.warning(f'Starting episode {episode_id}...')

        rollout_traces = self.batch_rollout(episode_id)
        loss = self.optimize_loss(rollout_traces)

        self.event_logger.log_scalar('Training_Success_Rate',
                                     self.stats_tracker.success_rate(), episode_id)
        self.event_logger.log_scalar('Average_Rollout_Length',
                                     self.stats_tracker.average_length(), episode_id)
        self.event_logger.log_scalar('Average_Action_Disversity',
                                     self.stats_tracker.average_action_diversity(), episode_id)
        self.event_logger.log_scalar('Training_Loss', loss, episode_id)

        if self.params.save_interval is not None and\
                episode_id > 0 and \
                episode_id % self.params.save_interval == 0:
            self.save_policy(str(episode_id))

        self.event_logger.warning(f'Finished episode {episode_id}')

    def save_params(self):
        if self.root_dir is not None:
            out_path = self.root_dir / 'parameters.json'
            with open(out_path, 'w') as f:
                json.dump(self.params._asdict(), f)

    def train(self):
        # Save the training parameters first for post-training inspection
        self.save_params()

        try:
            for episode_id in range(self.params.num_episodes):
                self.stats_tracker.new_episode()
                self.train_episode(episode_id)
        except KeyboardInterrupt:
            self.event_logger.warning('Training terminated by user')
        finally:
            self.save_policy('final')
