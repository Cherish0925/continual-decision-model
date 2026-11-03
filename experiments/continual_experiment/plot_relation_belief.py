"""Plot completed response-function comparisons and paired interventions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.input.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={'width_ratios':[1,1.15]})
    colors=['#23648e','#b45d24']
    conditions=[('Attentive reference','attentive_long{}_greedy'),
                ('Shared response functions','modulated{}_information'),
                ('Finite-budget lookahead','modulated{}_finite_plan'),
                ('Inherited CDM + memory','parent{}'),
                ('CDM + response functions','cdm{}')]
    for y,(label,pattern) in enumerate(conditions):
        for i,seed in enumerate((31,32)):
            item=data['conditions'][pattern.format(seed+880 if pattern=='parent{}' else seed)]['all']
            pos=y+(-.13 if i==0 else .13)
            a.scatter(item['operations'],pos,color=colors[i],s=30,zorder=3)
            a.annotate(f"{item['completed']}/{item['tasks']}",(item['operations'],pos),
                xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color=colors[i])
    reference=data['conditions']['known_formula']['all']
    a.scatter(reference['operations'],len(conditions),color='#665778',marker='D',s=30)
    a.annotate(f"{reference['completed']}/{reference['tasks']}",(reference['operations'],len(conditions)),
        xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color='#665778')
    a.set_yticks(range(len(conditions)+1),[x[0] for x in conditions]+['Known-formula reference'])
    a.invert_yaxis();a.set_xlim(90,280)
    a.set_xlabel('Mean operations, including failed tasks')
    a.set_title('A   Independent task streams',loc='left',pad=16)
    comparisons=[('Functions − attentive',lambda s:f'modulated{s}_information:attentive_long{s}_greedy'),
                 ('Memory − no memory',lambda s:f'modulated{s}_information:modulated{s}_no_memory'),
                 ('Lookahead − no hypothetical update',lambda s:f'modulated{s}_finite_plan:modulated{s}_finite_marginal'),
                 ('Lookahead − information rule',lambda s:f'modulated{s}_finite_plan:modulated{s}_information'),
                 ('Integrated − inherited CDM',lambda s:f'cdm{s}:parent{s+880}')]
    for y,(label,key) in enumerate(comparisons):
        for i,seed in enumerate((31,32)):
            item=data['comparisons'][key(seed)]['all']
            low,high=item['interval95'][0];mean=item['mean_difference'][0]
            pos=y+(-.13 if i==0 else .13)
            b.plot([low,high],[pos,pos],color=colors[i],linewidth=1.7)
            b.scatter(mean,pos,color=colors[i],s=25,zorder=3)
    b.set_yticks(range(len(comparisons)),[x[0] for x in comparisons]);b.invert_yaxis()
    b.axvline(0,color='#7e8790',linewidth=.8,linestyle='--')
    b.set_xlabel('Paired difference in operations (left is fewer)')
    b.set_title('B   Representation, memory and planning',loc='left',pad=16)
    for ax in (a,b):
        ax.grid(axis='x',color='#e5e8eb',linewidth=.6);ax.set_axisbelow(True)
    handles=[plt.Line2D([0],[0],marker='o',linewidth=0,color=c,label=f'Initialization {i+1}')
             for i,c in enumerate(colors)]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.035))
    fig.text(.02,.01,'576 tasks in 48 streams. Counts show completion. Intervals resample complete streams (95%).',
             color='#555e66',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1),w_pad=2.8)
    for ext in ('pdf','svg','png'):
        fig.savefig(args.output.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    svg=args.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
