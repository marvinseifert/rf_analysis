"""Signal vs noise of RF STAs per noise rate, same 27 cells (15/10/2026). Signal = peak contrast of the
RF-centre time course (3x3 blocks); noise = SD over blocks > 300 µm from the centre at the same lag."""
from pathlib import Path
import pickle, h5py, hdf5plugin, numpy as np, pandas as pd, torch
from scipy import ndimage
ROOT="/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"; STIM="/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026/32px_{}Hz_20mins_shuffle_x4.h5"
RATES={5:9,10:10,15:11,20:12}; B,FS=8,20000; LAGS=-800+10*np.arange(100)
HERE = Path(__file__).resolve().parents[1] / "results"   # stimulus_frequency_exploration/results
t=pd.read_csv(f"{HERE}/rf_timecourse_by_noise_rate.csv"); g=t[t.rf]
cells=list(g.pivot(index='cell',columns='rate',values='t_peak').dropna().index)
o=pickle.load(open(f"{ROOT}/overview","rb")); sp=pd.read_parquet(f"{ROOT}/alldata.parquet")
rows=[]
for hz,idx in RATES.items():
    r=o.stimulus_df[o.stimulus_df.stimulus_index==idx].iloc[0]; trig=np.asarray(r.trigger_fr_relative)
    with h5py.File(STIM.format(hz),"r") as f: S=f["Noise"][:]
    nf=S.shape[0]; S=(S[:,::B,::B]==255).astype(np.float32); hb,wb=S.shape[1:]
    S_t=torch.from_numpy(S.reshape(nf,-1)).cuda(); S_t-=S_t.mean(0,keepdim=True); del S
    end=trig[nf]; lag=(LAGS*FS//1000).astype(np.int64)
    m=(sp.times>=r.begin_fr)&(sp.times<r.begin_fr+end)
    spikes={c:gg.times.values-r.begin_fr for c,gg in sp[m].groupby("cell_index")}
    yy,xx=np.mgrid[:hb,:wb]
    for c in cells:
        tt=spikes[c]; fi=np.searchsorted(trig[:nf+1],tt[:,None]+lag[None,:],side="right")-1; ok=(fi>=0)&(fi<nf)
        W=np.zeros((100,nf),np.float32)
        for li in range(100): W[li]=np.bincount(fi[ok[:,li],li],minlength=nf)
        sta=(torch.from_numpy(W).cuda()@S_t).cpu().numpy().reshape(100,hb,wb)/len(tt)
        v=ndimage.gaussian_filter(sta,(2,1,1)).var(axis=0); cy,cx=np.unravel_index(v.argmax(),v.shape)
        tc=sta[:,max(cy-1,0):cy+2,max(cx-1,0):cx+2].mean(axis=(1,2)); pre=LAGS<=0
        j=np.argmax(np.abs(tc)*pre)
        far=np.hypot(yy-cy,xx-cx)*16>300
        # noise of a 3x3-block average, so it is comparable with the centre signal
        sm=ndimage.uniform_filter(sta[j],3)
        rows.append(dict(cell=c,rate=hz,n=len(tt),signal=abs(tc[j]),noise=sm[far].std()))
    print(hz,"done",flush=True); del S_t; torch.cuda.empty_cache()
d=pd.DataFrame(rows); d['snr']=d.signal/d.noise
d.to_csv(f"{HERE}/signal_noise_by_rate.csv",index=False)
med=d.groupby('rate')[['n','signal','noise','snr']].median()
print(med.round(4).to_string()); print((med/med.loc[5]).round(2).to_string())
# noise expected if it only scaled with 1/sqrt(spikes)
print('noise x sqrt(spikes) (should be constant if only spike count mattered):', (d.noise*np.sqrt(d.n)).groupby(d.rate).median().round(3).to_dict())
