"""Reliability of moving bar / edge responses across repeats (27 Nov 2026). Times from the REAL triggers and the
recording's own sampling rate (not frame / Frame_Rate). Reliability = mean correlation of 0.5 s PSTHs between repeats;
chance = each repeat shifted by its own random offset. Usage: python edge_reliability.py 22_07_2026:6 14_08_2026:15 ...
Writes results/edge_rel_<recording>_<stimulus index>.csv"""
import pickle, numpy as np, pandas as pd, sys
from pathlib import Path
RES = Path(__file__).resolve().parent / 'results'
rng=np.random.default_rng(0); BIN=0.5
def run(rec, idx, ph='Phase_00'):
    root=f'/mnt/mea_nas/Laura/zebrafish_{rec}/{ph}'
    o=pickle.load(open(root+'/overview','rb')); sd=o.stimulus_df; sp=pd.read_parquet(root+'/alldata.parquet')
    r=sd[sd.stimulus_index==idx].iloc[0]; trig=np.asarray(r.trigger_fr_relative)+r.begin_fr; fs=r.sampling_freq; R=int(r.nr_repeats)
    n_per=(len(trig)-1)//R
    edges=[(k,np.arange(trig[k*n_per],trig[min((k+1)*n_per,len(trig)-1)],BIN*fs)) for k in range(R)]
    L=min(len(e) for _,e in edges); edges=[(k,e[:L]) for k,e in edges]
    m=(sp.times>=trig[0])&(sp.times<trig[-1]); res=[]
    span=trig[-1]-trig[0]
    def shift_each(t):
        out=[]
        for k,e in edges:
            a,b=e[0],e[-1]; tk=t[(t>=a)&(t<b)]-a; L=b-a
            out.append((tk+rng.uniform(.1,.9)*L)%L+a)
        return np.sort(np.concatenate(out))
    def rel(t):
        V=[np.histogram(t,e)[0].astype(float) for _,e in edges]
        return np.nanmean([np.corrcoef(V[a],V[b])[0,1] for a in range(R) for b in range(a+1,R)])
    for c,gg in sp[m].groupby('cell_index'):
        t=gg.times.values
        if len(t)<100: continue
        res.append(dict(cell=c,real=rel(t),null=max(rel(shift_each(t)) for _ in range(5))))
    d=pd.DataFrame(res); thr=np.nanpercentile(d.null,99); d['rec']=rec; d['idx']=idx; d.to_csv(RES / f'edge_rel_{rec}_{idx}.csv',index=False)
    print(f'{rec} {ph} {idx:2d} {r.stimulus_name[:45]:45s} {fs/np.median(np.diff(trig)):5.1f} fps x{R}: cells {len(d)}, reliability median {d.real.median():.2f}, above chance {(d.real>thr).sum()}, >0.3 {(d.real>0.3).sum()}', flush=True)
for a in sys.argv[1:]:
    rec,idx=a.split(':'); run(rec,int(idx))
