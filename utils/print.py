import numpy as np


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions =  ["◼", "↻", "←", "→"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def print_results(episode_idx, normalized_score, smoothed_normalized_score,
                  completion, smoothed_completion, eps, action_probs):
    print(
        '🚂 Episode {}'
        '\t 🏆 Score: {:.3f}'
        ' Avg: {:.3f}'
        '\t 💯 Done: {:.2f}%'
        ' Avg: {:.2f}%'
        '\t 🎲 Epsilon: {:.3f} '
        '\t 🔀 Action Probs: {}'.format(
            episode_idx,
            normalized_score,
            smoothed_normalized_score,
            100 * completion,
            100 * smoothed_completion,
            eps,
            format_action_prob(action_probs)
        ))
