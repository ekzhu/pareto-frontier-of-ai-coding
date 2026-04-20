import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data points: (autonomy, precision)
# As autonomy increases, precision generally decreases
points = {
    'Classic Coding': (0.2, 10),
    'Tab Autocomplete': (1.5, 9.8),
    'AI Assisted IDE': (3, 9.4),
    'Single Agent CLI': (4.5, 8.8),
    '2 Agents CLI': (6, 7.5),
    '3 Agents CLI': (7.5, 4.5),
    '4 Agents CLI': (9, 2),
    'Multi-Agent Swarm': (10, 0.5)
}

labels = list(points.keys())
autonomy = [v[0] for v in points.values()]
precision = [v[1] for v in points.values()]

with plt.xkcd(scale=1.2, length=120, randomness=3):
    # Override the default xkcd font list (which expects xkcd Script / Humor Sans
    # / Comic Neue — none of those are installed) with the Comic Sans MS font
    # that ships with macOS. This keeps the hand-drawn feel without font warnings.
    plt.rcParams['font.family'] = ['Comic Sans MS']

    fig, ax = plt.subplots(figsize=(11, 9))

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(points)))
    ax.scatter(autonomy, precision, c=colors, s=400,
               edgecolors='black', linewidth=2.5, zorder=5)

    # Per-label offset overrides (in points) to avoid overlap with neighbors
    label_offsets = {
        'Classic Coding': (8, 22),
    }

    for i, label in enumerate(labels):
        ax.annotate(label,
                    (autonomy[i], precision[i]),
                    fontsize=14,
                    xytext=label_offsets.get(label, (8, 8)),
                    textcoords='offset points')

    pareto_autonomy = autonomy
    pareto_precision = precision
    x_smooth = np.linspace(min(pareto_autonomy), max(pareto_autonomy), 300)
    spline = make_interp_spline(pareto_autonomy, pareto_precision, k=3)
    y_smooth = spline(x_smooth)

    ax.plot(x_smooth, y_smooth, 'r-', linewidth=4, label='Pareto Frontier',
            zorder=4, alpha=0.8)
    ax.fill_between(x_smooth, y_smooth, alpha=0.15, color='red')

    ax.set_xlabel('Autonomy  -->', fontsize=18)
    ax.set_ylabel('Precision  -->', fontsize=18)
    ax.set_title('Pareto Frontier of AI Coding', fontsize=22, pad=20)

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 13.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=15)

    def labeled_arrow(ax, tail, tip, label, color, label_pos):
        ax.annotate('', xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3))
        ax.text(label_pos[0], label_pos[1], label,
                fontsize=15, color=color,
                ha='center', va='center')

    # Arrow 1: Single Agent CLI -> Classic Coding, above curve
    labeled_arrow(ax,
                  tail=(5, 11.3), tip=(1.2, 11.3),
                  label='diminishing value\nof hand-written code',
                  color='darkblue',
                  label_pos=(3.1, 12))

    # Arrow 2: 2 Agents CLI -> Multi-Agent Swarm, outside curve, short enough
    # not to touch arrow 3.
    labeled_arrow(ax,
                  tail=(9.8, 7), tip=(9.8, 3),
                  label='intrinsic chaos of\nmulti-agent systems',
                  color='darkred',
                  label_pos=(11.3, 5))

    # Arrow 3: center of curve -> top-right corner, starts outside shaded area
    labeled_arrow(ax,
                  tail=(7.8, 8), tip=(11.5, 12),
                  label='advancement\nof models',
                  color='darkgreen',
                  label_pos=(10.5, 9.5))

    # Credit in bottom-right corner
    ax.text(0.99, 0.01, 'x.com/ekzhu',
            transform=ax.transAxes,
            fontsize=12, color='gray',
            ha='right', va='bottom')

    plt.tight_layout()

    output_path = '/Users/erkang.zhu/code/ai-coding-frontier/pareto_frontier.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Diagram saved to: {output_path}')
    plt.close()
