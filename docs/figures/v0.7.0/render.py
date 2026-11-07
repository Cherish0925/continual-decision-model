"""Render the v0.7.0 implementation and related-method schematics.

Run from any directory: python docs/figures/v0.7.0/render.py
Dependencies: matplotlib and numpy. All figure text remains vector text.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
INK = "#172B3A"
MUTED = "#516574"
BLUE = "#EAF2FA"
BLUE_EDGE = "#4777A1"
TEAL = "#E1F4EE"
TEAL_EDGE = "#15785F"
AMBER = "#FFF1DA"
AMBER_EDGE = "#AF7422"
GREY = "#F1F4F6"
LINE = "#657888"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def canvas(height):
    fig = plt.figure(figsize=(16, height / 10), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 160), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, s, size=12, weight="normal", color=INK,
          ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, weight=weight, color=color,
                   ha=ha, va=va, linespacing=1.35, zorder=5, **kw)


def box(ax, x, y, w, h, title, body="", kind="fixed", ts=13, bs=11.5):
    fill, edge = {"fixed": (BLUE, BLUE_EDGE), "learn": (TEAL, TEAL_EDGE),
                  "explicit": (AMBER, AMBER_EDGE), "plain": (GREY, LINE),
                  "white": ("white", LINE)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.65",
                              linewidth=1.15, edgecolor=edge, facecolor=fill, zorder=3))
    if body:
        label(ax, x+w/2, y+h-3.1, title, ts, "bold", ha="center")
        label(ax, x+w/2, y+(h-4.0)/2, body, bs, ha="center")
    else:
        label(ax, x+w/2, y+h/2, title, ts, "bold", ha="center")


def arrow(ax, pts, color=LINE, dashed=False, width=1.4):
    # Orthogonal routes are specified explicitly to keep connections legible.
    ls = (0, (4, 3)) if dashed else "-"
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=width, ls=ls, zorder=2)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                               mutation_scale=11, linewidth=width,
                               color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                               zorder=2))


def legend(ax, y, comparison=False):
    items = [(7, BLUE, BLUE_EDGE, "Neural module / representation" if comparison else "Weights fixed in this study"),
             (61, TEAL, TEAL_EDGE, "Learned or adapted component" if comparison else "Trained in v0.7.0"),
             (111, AMBER, AMBER_EDGE, "Explicit state / numerical procedure")]
    for x, fill, edge, text in items:
        ax.add_patch(Rectangle((x, y-1), 1.8, 1.8, fc=fill, ec=edge, lw=1))
        label(ax, x+2.7, y-.05, text, 10.5, color=MUTED)


def architecture():
    fig, ax = canvas(160)
    label(ax, 7, 154.5, "Continual Decision Model", 23, "bold")
    label(ax, 7, 149.7, "v0.7.0  |  Deployed numerical-selector configuration  |  Reactor Lab", 13, color=MUTED)
    legend(ax, 144)

    box(ax, 57, 129, 46, 10, "Current public observation", "Image, text, objects, tool returns", "plain")
    box(ax, 7, 108, 42, 14, "Frozen SigLIP 2 encoders", "Image: 16 pooled tokens\nText: token embeddings; width 768")
    box(ax, 60, 108, 40, 14, "Observation and record processing", "Object identities, measurements,\ncalibration results and history", "explicit", ts=12.2)
    box(ax, 112, 108, 40, 14, "Public action candidates", "SigLIP text embeddings\nand numeric action features", "fixed", ts=12.4)
    arrow(ax, [(80,129),(80,125),(28,125),(28,122)])
    arrow(ax, [(80,129),(80,122)])
    arrow(ax, [(80,125),(132,125),(132,122)])

    box(ax, 7, 86, 42, 14, "Multimodal event representation", "Projection + modality / time features\nEvent tokens at width 1024", ts=12.5)
    arrow(ax, [(28,108),(28,100)])
    arrow(ax, [(60,113),(53,113),(53,93),(49,93)], color=AMBER_EDGE)

    # The recurrent state is shown explicitly, rather than as an unspecified memory.
    ax.add_patch(FancyBboxPatch((7,56),42,23, boxstyle="round,pad=0,rounding_size=.65",
                              facecolor=BLUE, edgecolor=BLUE_EDGE, linewidth=1.15, zorder=3))
    label(ax, 28, 75.8, "Persistent neural core", 13.5, "bold", ha="center")
    label(ax, 28, 70.6, r"GRU: $h_t \in \mathbb{R}^{1024}$", 12.8, ha="center")
    label(ax, 28, 65.1, r"Transformer slots: $z_t \in \mathbb{R}^{8\times1024}$", 12.2, ha="center")
    label(ax, 28, 59.9, "One workspace update per observation", 10.7, ha="center", color=MUTED)
    arrow(ax, [(28,86),(28,79)])
    label(ax, 29.8, 82.4, "attention-pooled event", 9.8, color=MUTED)
    arrow(ax, [(7,63),(5,63),(5,76),(7,76)], color=BLUE_EDGE)
    label(ax, 8.5, 52.3, r"$h_{t-1}, z_{t-1}$ are retained across actions", 10.5, color=BLUE_EDGE)

    box(ax, 60, 86, 40, 14, "Numerical hypothesis model", "9 affine hypotheses\nBayesian coefficient / model updates", "explicit", ts=12.4)
    arrow(ax, [(80,108),(80,100)], color=AMBER_EDGE)
    box(ax, 112, 86, 40, 14, "Learned experiment selector", "10 features per candidate plan\nPredict success probability and cost", "learn", ts=12.5)
    arrow(ax, [(100,93),(112,93)], color=AMBER_EDGE)
    label(ax, 106, 95.3, "plans", 10, ha="center", color=MUTED)

    box(ax, 60, 58, 40, 18, "Native action reader", "Base action score + belief residual\nReads neural state and candidates\n125 numeric features per action", "fixed", ts=13)
    arrow(ax, [(49,67.5),(60,67.5)], color=BLUE_EDGE)
    label(ax, 54.5, 70.2, r"$h_t,\ \bar z_t$", 10.5, ha="center", color=BLUE_EDGE)
    arrow(ax, [(80,86),(80,76)], color=AMBER_EDGE)
    label(ax, 82, 81.2, "numeric beliefs", 10, color=MUTED)
    arrow(ax, [(132,108),(132,104),(105,104),(105,71),(100,71)])

    box(ax, 112, 58, 40, 18, "Persistent experiment executor", "Reach setting + local numeric probe\nRetain failed choices for this instance\nFinish on activation or probe end", "explicit", ts=12.2)
    arrow(ax, [(132,86),(132,76)], color=TEAL_EDGE)
    label(ax, 134, 81.2, "selected plan", 10, color=MUTED)

    box(ax, 60, 42, 40, 9, "Action dispatch / public API", kind="plain", ts=12.7)
    arrow(ax, [(80,58),(80,51)])
    arrow(ax, [(132,58),(132,46.5),(100,46.5)], color=AMBER_EDGE)
    label(ax, 116, 49.1, "numeric-dialogue action", 9.8, ha="center", color=MUTED)
    box(ax, 60, 27, 40, 9, "DiscoveryWorld environment", kind="plain", ts=12.7)
    arrow(ax, [(80,42),(80,36)])
    label(ax, 82, 39, r"structured action $a_t$", 10, color=MUTED)
    arrow(ax, [(60,31.5),(2,31.5),(2,134),(57,134)], color=LINE)
    label(ax, 7, 135.7, "next observation + action feedback", 10.3, color=MUTED)
    label(ax, 7, 37.5, "State updates continue during execution.\nNeural weights stay fixed at evaluation.", 11, color=MUTED)
    label(ax, 112, 33, "Selector input excludes neural state\nin this configuration. The action\nreader still uses the recurrent core.", 10.6, color=MUTED)

    ax.plot([7,152],[23,23],color="#CFD8DE",lw=1)
    label(ax, 7, 20.2, "OFFLINE SELECTOR TRAINING", 10.5, "bold", color=TEAL_EDGE)
    box(ax, 7, 5, 42, 12, "Matched training branches", "237 trials / 32 training histories\nSeparate validation: 65 trials / 8 histories", "plain", ts=11.7, bs=10.4)
    box(ax, 60, 5, 40, 12, "Outcome and preference losses", "Observed success + execution cost\nPreferences within the same history", "learn", ts=11.2, bs=10.7)
    box(ax, 112, 5, 40, 12, "317,314-parameter selector", "100 epochs; fixed final checkpoint\nCore, perception and prior held fixed", "learn", ts=11.7, bs=10.7)
    arrow(ax, [(49,11),(60,11)], color=TEAL_EDGE)
    arrow(ax, [(100,11),(112,11)], color=TEAL_EDGE)
    arrow(ax, [(152,11),(157,11),(157,93),(152,93)], color=TEAL_EDGE, dashed=True)
    label(ax, 7, 1.8, "Source: release v0.7.0, commit 67c1d8f. Arrows show data / execution flow; the dashed arrow transfers learned weights.", 9, color=MUTED)
    return fig


def panel(ax,x,y,w,h,letter,title,subtitle):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.7",
                              fc="white",ec="#D2DCE3",lw=1,zorder=0))
    label(ax,x+3,y+h-3.8,letter,13,"bold",color=MUTED)
    label(ax,x+8,y+h-3.8,title,15,"bold")
    label(ax,x+3,y+h-8,subtitle,10.5,color=MUTED)


def core_detail():
    fig, ax = canvas(145)
    label(ax,7,139.7,"Persistent core: one observation update",22,"bold")
    label(ax,7,134.5,"Exact computational structure of the reference-reader path  |  Width 1024; eight latent slots",12.5,color=MUTED)
    box(ax,7,111,32,10,"Previous fast state",r"$h_{t-1}\in\mathbb{R}^{1024}$",ts=12,bs=12)
    box(ax,7,91,32,12,"Current event tokens",r"$E_t$: text, image, numeric"+"\nand measurement-record tokens",ts=12,bs=10.6)
    box(ax,54,101,43,16,"Attention read + input mapping",r"Query: $\mathrm{LN}(h_{t-1})$; keys / values: $E_t$"+"\nNormalize, bridge and event MLP",ts=12.3,bs=10.6)
    box(ax,111,101,41,16,"GRU state update",r"$h_t=\mathrm{GRU}(x_t,h_{t-1})$"+"\nOne update per received observation",ts=13,bs=11.5)
    arrow(ax,[(39,116),(46,116),(46,112),(54,112)],color=BLUE_EDGE)
    arrow(ax,[(39,97),(46,97),(46,106),(54,106)],color=BLUE_EDGE)
    arrow(ax,[(97,109),(111,109)],color=BLUE_EDGE)
    arrow(ax,[(131.5,117),(131.5,126),(23,126),(23,121)],color=BLUE_EDGE)
    label(ax,75,127.7,"carry to the next observation",10,color=BLUE_EDGE,ha="center")

    box(ax,111,81,41,12,"Workspace context",r"$C_t=[h_t;E_t]$"+"\nIncludes observed measurement records",ts=12.5,bs=10.5)
    arrow(ax,[(131.5,101),(131.5,93)],color=BLUE_EDGE)
    arrow(ax,[(39,94),(43,94),(43,87),(111,87)],color=BLUE_EDGE)
    box(ax,7,59,32,13,"Previous workspace",r"$z_{t-1}\in\mathbb{R}^{8\times1024}$"+"\nPersistent latent slots",ts=12.2,bs=11)

    ax.add_patch(FancyBboxPatch((52,36),47,45,boxstyle="round,pad=0,rounding_size=.7",fc="#F7FAFD",ec=BLUE_EDGE,lw=1.2))
    label(ax,75.5,77.8,"Shared Transformer block",12.6,"bold",ha="center")
    box(ax,55,64,41,10,"Slot self-attention + residual",r"$u=z+\mathrm{SA}(\mathrm{LN}(z))$",ts=11.5,bs=11)
    box(ax,55,51,41,10,"Context cross-attention + residual",r"$v=u+\mathrm{CA}(\mathrm{LN}(u),C_t)$",ts=11,bs=11)
    box(ax,55,38,41,10,"MLP + residual + slot normalization",r"$z_t=\mathrm{LN}(v+\mathrm{MLP}(\mathrm{LN}(v)))$",ts=10.6,bs=10.5)
    arrow(ax,[(39,65.5),(46,65.5),(46,69),(55,69)],color=BLUE_EDGE)
    arrow(ax,[(75.5,64),(75.5,61)],color=BLUE_EDGE)
    arrow(ax,[(75.5,51),(75.5,48)],color=BLUE_EDGE)
    arrow(ax,[(131.5,81),(131.5,74),(104,74),(104,56),(96,56)],color=BLUE_EDGE)

    box(ax,111,42,41,17,"Decision state",r"$s_t=[\mathrm{LN}(h_t);\mathrm{LN}(\mathrm{mean}(z_t))]$"+"\n2048 features for action reading",ts=13,bs=10.8)
    arrow(ax,[(96,43),(104,43),(104,48),(111,48)],color=BLUE_EDGE)
    arrow(ax,[(152,109),(157,109),(157,54),(152,54)],color=BLUE_EDGE)
    arrow(ax,[(75.5,38),(75.5,30),(23,30),(23,59)],color=BLUE_EDGE)
    label(ax,49,32.3,"retain workspace for the next observation",10,color=BLUE_EDGE,ha="center")
    label(ax,111,35,"Reference path: one shared-block application.\nNo learned refinement count in this configuration.",10.3,color=MUTED)
    ax.plot([7,152],[25,25],color="#CFD8DE",lw=1)
    label(ax,7,21.5,"SEPARATE PREDICTIVE-PARENT CONFIGURATION",10.5,"bold",color=MUTED)
    box(ax,7,6,42,12,"Candidate-action forecasts","Success, numeric returns and\nreturns over 1 / 8 / 32 / 128 actions",ts=11.8,bs=10.3)
    box(ax,60,6,40,12,"Predicted context tokens","Top-8 candidate embeddings\ncombined with their forecasts",ts=11.8,bs=10.6)
    box(ax,111,6,41,12,"Shared workspace refinement","Two additional fixed cycles\nRecompute forecasts and action scores",ts=11.5,bs=10.3)
    arrow(ax,[(49,12),(60,12)],color=BLUE_EDGE)
    arrow(ax,[(100,12),(111,12)],color=BLUE_EDGE)
    label(ax,7,2,"This prediction-feedback path is not used by the transferred numerical selector that completed 23/24 instances.",9.7,color=MUTED)
    return fig


def comparison():
    fig, ax = canvas(150)
    label(ax,7,144.8,"Where computation, memory and learning reside",22,"bold")
    label(ax,7,139.7,"Architecture comparison  |  Schematics redrawn from code and primary papers",12.5,color=MUTED)
    legend(ax,134,comparison=True)
    panel(ax,7,78,70,50,"a","CDM v0.7.0","Numerical-selector configuration; native action execution")
    box(ax,11,105,28,10,"Multimodal core","GRU + Transformer state",ts=11.4,bs=9.5)
    box(ax,45,105,28,10,"Action reader","Candidate action scores",ts=11.4,bs=9.5)
    arrow(ax,[(39,110),(45,110)],color=BLUE_EDGE)
    box(ax,11,89,28,10,"Numerical beliefs","Measurements + 9 hypotheses","explicit",ts=11.4,bs=9.2)
    box(ax,45,89,28,10,"Experiment selector","Choose a numerical plan","learn",ts=11.2,bs=9.2)
    arrow(ax,[(39,94),(45,94)],color=AMBER_EDGE)
    arrow(ax,[(25,99),(25,102),(59,102),(59,105)],color=AMBER_EDGE)
    label(ax,11,82.5,"Runtime: h / z, numerical posteriors and plan state update.",10.2,color=MUTED)

    panel(ax,82,78,70,50,"b","I2A [1]","Learned interpretation of model-generated futures")
    box(ax,86,105,28,10,"Environment model","Action-conditioned rollouts",ts=11.4,bs=9.5)
    box(ax,120,105,28,10,"Rollout encoder","Encoded imagined outcomes",ts=11.4,bs=9.5)
    arrow(ax,[(114,110),(120,110)],color=BLUE_EDGE)
    box(ax,86,89,28,10,"Real observation","Model-free features",ts=11.4,bs=9.5)
    box(ax,120,89,28,10,"Policy + value","Read real + imagined features","learn",ts=11.4,bs=9.2)
    arrow(ax,[(114,94),(120,94)])
    arrow(ax,[(134,105),(134,99)],color=BLUE_EDGE)
    label(ax,86,82.5,"Predicted trajectories provide context for action selection.",10.2,color=MUTED)

    panel(ax,7,23,70,50,"c","Step-DAD [2]","Sequential experimental design with test-time adaptation")
    box(ax,11,50,28,10,"Design policy","Choose next experiment","learn",ts=11.4,bs=9.5)
    box(ax,45,50,28,10,"Experiment","Acquire an observation","plain",ts=11.4,bs=9.5)
    arrow(ax,[(39,55),(45,55)])
    box(ax,11,34,28,10,"Policy adaptation","Periodic parameter updates","learn",ts=11.4,bs=9.3)
    box(ax,45,34,28,10,"Experiment history","Accumulate design / outcome","explicit",ts=11.4,bs=9.2)
    arrow(ax,[(59,50),(59,44)])
    arrow(ax,[(45,39),(39,39)],color=TEAL_EDGE)
    arrow(ax,[(25,44),(25,50)],color=TEAL_EDGE,dashed=True)
    arrow(ax,[(73,39),(75,39),(75,62),(25,62),(25,60)],color=AMBER_EDGE)
    label(ax,11,27.5,"Observed data refine the deployed experimental-design policy.",10.1,color=MUTED)

    panel(ax,82,23,70,50,"d","EvoBelief [3]","Frozen language model with an evolving external world model")
    box(ax,86,50,28,10,"Frozen LLM actor","Candidate task actions",ts=11.4,bs=9.5)
    box(ax,120,50,28,10,"Prediction + planning","Rule-conditioned consequences",ts=11.2,bs=9.1)
    arrow(ax,[(114,55),(120,55)])
    box(ax,86,34,28,10,"Observations + tests","Fit / test / revise hypotheses","explicit",ts=11.2,bs=9.2)
    box(ax,120,34,28,10,"Persistent world beliefs","Reusable external rules","explicit",ts=10.9,bs=9.5)
    arrow(ax,[(114,39),(120,39)],color=AMBER_EDGE)
    arrow(ax,[(134,44),(134,50)],color=AMBER_EDGE)
    arrow(ax,[(148,55),(150,55),(150,47),(100,47),(100,44)])
    label(ax,86,27.5,"Rules change during deployment; LLM weights remain fixed.",10.2,color=MUTED)

    label(ax,7,17.8,"READING THE COMPARISON",10.5,"bold",color=MUTED)
    label(ax,7,13.4,"CDM maintains a neural state and explicit numerical state; v0.7.0 learns the experiment selector offline.",11.2)
    label(ax,7,9.2,"Its prediction-feedback variant is a separate configuration. These diagrams compare mechanisms, not benchmark rankings.",10.7,color=MUTED)
    label(ax,7,4.9,"[1] Weber et al., NeurIPS 2017, arXiv:1707.06203.   [2] Hedman et al., ICML 2025, PMLR 267.",9.5,color=MUTED)
    label(ax,7,1.7,"[3] Wang et al., 2026, arXiv:2610.12789 (preprint). Schematic connections omit task-specific implementation details.",9.5,color=MUTED)
    return fig


def main():
    figures = [("cdm_v0.7_architecture", architecture()),
               ("cdm_v0.7_core", core_detail()),
               ("cdm_related_architectures", comparison())]
    with PdfPages(OUT / "cdm_v0.7_architecture_comparison.pdf",
                  metadata={"Title":"CDM v0.7.0: architecture and related methods",
                            "Subject":"Implementation-grounded scientific schematics"}) as pdf:
        for stem, fig in figures:
            fig.savefig(OUT / (stem + ".svg"), format="svg")
            fig.savefig(OUT / (stem + ".png"), dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    print("Rendered three vector figures, PNG previews and a three-page PDF.")


if __name__ == "__main__":
    main()
