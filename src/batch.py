import pickle


def new_batch(env, size, policy, episode_length_bound):
    batch = list()
    for _ in range(size):
        episode = new_episode(env, episode_length_bound, policy)
        batch.append(episode)
    return batch


def new_episode(env, episode_length_bound, policy):
    state = env.reset()
    # state = 0
    episode = list()
    for t in range(episode_length_bound):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward, next_state))
        if done:
            break
        state = next_state
    return episode


def save_batch(batch, path):
    with open(path, 'wb') as f:
        pickle.dump(batch, f)


def load_batch(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
