import matplotlib.pyplot as plt
import numpy as np


# set the style
plt.style.use('seaborn-v0_8-poster')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5
})

k_values = np.arange(3, 13, 2)
dice_scores = np.array([85.63, 85.95, 86.00, 85.78, 85.49])

# create chart
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# draw
line = ax.plot(k_values, dice_scores,
               color='#2ca02c',  #  the recommend color of IEEE
               marker='o',
               markersize=8,
               linewidth=3,
               label='Proposed Method (BePMI)')


for k, score in zip(k_values, dice_scores):
    ax.annotate(f'{score:.2f}%',  # .20
                xy=(k, score),
                xytext=(0, 15),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=12,
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor='white',
                         edgecolor='lightgray',
                         alpha=0.8)
                )

# chart decorate
ax.set_xlabel('Boundary Width $k$', fontsize=18, labelpad=2)
ax.set_ylabel('Dice Score (%)', fontsize=18, labelpad=2)

ax.yaxis.grid(True, linestyle=':', alpha=0.7, color='lightgrey')
ax.xaxis.grid(False)

ax.legend(framealpha=1, loc='lower right',fontsize=18)

ax.set_xlim(2, 12)
ax.set_ylim(80.0, 88.0)
ax.set_xticks([3, 5, 7, 9, 11])
ax.tick_params(axis='both', labelsize=18,pad=2)

plt.tight_layout()

plt.savefig('dice_with_different_k.pdf', bbox_inches='tight', transparent=True)
# plt.savefig('dice_vs_k.png', bbox_inches='tight', dpi=600)

plt.show()
