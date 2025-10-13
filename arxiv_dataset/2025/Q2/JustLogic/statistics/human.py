import matplotlib.pyplot as plt
import numpy as np

def pie_chart(labels, values, colors):
    fig, ax = plt.subplots()
    ax.pie(
        values, labels=labels, colors=colors, wedgeprops=dict(width=0.5),
        autopct='%1.0f%%'
    )
    plt.show()

def donut_chart(recipe, data, colors, title, filename):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(
        data, wedgeprops=dict(width=0.5), startangle=-40, colors=colors,
        autopct='%1.1f%%', pctdistance=0.8
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7)
        autotext.set_weight('bold')

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(title, pad=15)
    plt.savefig(filename, dpi=800)
    plt.show()

if __name__ == '__main__':
    colors = ['#1E40AF', '#AF1E89', '#AF8D1E', '#1EAF44']

    # Education
    labels = ['High School Diploma', "Associate's Degree", "Bachelor's Degree", "Postgraduate Degree"]
    values = [2, 1, 10, 5]
    title = 'What is your highest level of education?'
    filename='human_edu_qual.png'
    # Familiarity
    # labels = ['Yes', 'Somewhat', 'No']
    # values = [8, 2, 8]
    # title = 'Are you familiar with propositional logic?'
    # filename='human_familiar_logic.png'

    # labels = [f"{l} ({'{:.1f}'.format(values[i] / sum(values) * 100)}%)" for i,l in enumerate(labels)]
    donut_chart(labels, values, colors, title, filename)