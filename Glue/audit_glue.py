from pathlib import Path
import json, os, tempfile, contextlib, io
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
ROOT=Path(__file__).resolve().parent.parent
nb=json.loads((ROOT/'Glue/Glue.ipynb').read_text())
# Execute a fresh kernel-equivalent namespace; all generated files go to a temporary directory.
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp); (p/'Glue').mkdir(); (p/'Christina_results').symlink_to(ROOT/'Christina_results',target_is_directory=True)
    (p/'Glue/classification_64.csv').symlink_to(ROOT/'Glue/classification_64.csv')
    os.chdir(p/'Glue'); os.environ['MPLBACKEND']='Agg'
    ns={}
    with contextlib.redirect_stdout(io.StringIO()):
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code': exec(compile(''.join(c['source']),f'Glue.ipynb cell {i}', 'exec'),ns)
    for name in ['confusion_by_z.csv','glue_by_z.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(name),pd.read_csv(ROOT/'Glue'/name))
    print('FRESH EXECUTION: all code cells passed; both saved CSVs match')
    df=ns['df_classifications']; cm=ns['cm']; rc=ns['rate_counts']; results=ns['results']; glue=ns['glue']
    print('DATA',len(df),'unique IDs',df.SNID.nunique(),'redshift null',df.SIM_REDSHIFT_CMB.isna().sum(),'outside bins',((df.SIM_REDSHIFT_CMB<0)|(df.SIM_REDSHIFT_CMB>=5)).sum())
    print('True SLSN label/redshift mismatches',(((df.true_label=='lo')&(df.SIM_REDSHIFT_CMB>=1))|((df.true_label=='hi')&(df.SIM_REDSHIFT_CMB<1))).sum())
    # Independently count using histogram2d-like groupby, with numeric rounded edges.
    zbin=pd.cut(df.SIM_REDSHIFT_CMB,np.linspace(0,5,26),right=False,labels=False)
    grouped=df.groupby([zbin,'true_label','pred_label']).size()
    for i,row in cm.iterrows():
        for t in ns['LABELS']:
            for pred in ns['LABELS']: assert row[f'n_{t}_as_{pred}']==grouped.get((i,t,pred),0)
    for model,f in ns['rate_files'].items():
        raw=pd.read_csv(f)
        assert rc[f'N_{model}'].sum()==raw.n_success_ELAsTiCC.sum()
        for i,row in rc.iterrows():
            exp=raw.loc[(raw.z_lo>=row.z_low-1e-8)&(raw.z_hi<=row.z_high+1e-8),'n_success_ELAsTiCC'].sum()
            assert exp==row[f'N_{model}']
    print('Independent confusion counts and all rate-bin sums: PASS')
    print('RESULTS\n',results.to_string())
    for a,b in ns['pairs']:
        z10=abs(results.loc[a,'R']-results.loc[b,'R'])/np.hypot(results.loc[a,'sigma_R'],results.loc[b,'sigma_R'])
        exposure=ns['survey_days']*(5/z10)**2
        label=64+float(ns['days_to_5sigma'](a,b,**ns['oracle']))
        print('TIME',a,b,'curve 5-sigma day',64+exposure,'displayed label',label,'actual curve Z at label',z10*np.sqrt((label-64)/ns['survey_days']))
    print('N_CLASS',ns['N_class'])
    print('ORACLE',ns['oracle'])
    for model in ns['rate_files']:
        simple={c:ns['oracle'][f'recall_{c}']*ns['N_class'][model][c]+ns['oracle'][f'recall_{c}']*ns['N_class']['naive'][c]*(1-ns['oracle'][f'precision_{c}'])/ns['oracle'][f'precision_{c}'] for c in ['lo','hi']}
        leakage={c:float((rc[f'N_{model}'].where(ns['has_rates'],0)*cm[f'f_{"hi" if c=="lo" else "lo"}_as_{c}']).sum()) for c in ['lo','hi']}
        print('SIMPLIFIED_COUNTS',model,simple,'LEAKAGE',leakage)
    print('CONTAMINANTS_BY_CLASS\n',df.loc[(df.true_label=='oth')&(df.pred_label!='oth')].groupby(['true_g','pred_label']).size().unstack(fill_value=0).sort_values('hi',ascending=False).to_string())
    # First-order classifier covariance for each multinomial true-SLSN row,
    # and Poisson contaminant numerators; rates and denominators conditional on their observed totals.
    models=list(ns['rate_files']); gradients={}; statvar={}
    for m in models:
        R=results.loc[m,'R']; plo=results.loc[m,'P_lo']
        N=rc[f'N_{m}'].to_numpy(); N0=rc.N_naive.to_numpy()
        grad=np.array([-R,1.])/plo
        blocks=[]
        for i,row in cm.iterrows():
            nt=row.n_true_slsn
            if nt==0: continue
            t='hi' if row.z_low>=1 else 'lo'
            probs=np.array([row[f'f_{t}_as_lo'],row[f'f_{t}_as_hi']])
            cov=(np.diag(probs)-np.outer(probs,probs))/nt
            ev,vec=np.linalg.eigh(cov)
            blocks.extend((N[i]*grad @ (vec*np.sqrt(np.clip(ev,0,None)))).tolist())
            blocks.extend((N0[i]*grad*np.sqrt(np.array([row.n_oth_as_lo,row.n_oth_as_hi]))/nt).tolist())
        gradients[m]=np.array(blocks)
        print('CALIBRATION_CONDITIONAL',m,'sigma_R',np.linalg.norm(gradients[m]),'Poisson',results.loc[m,'sigma_R'])
    for a,b in ns['pairs']:
        print('CALIBRATION_CONTRAST',a,b,np.linalg.norm(gradients[a]-gradients[b]))
    # Background sensitivity comparison, maintaining the original heuristic statistic only.
    for a,b in ns['pairs']:
        ra,sa,da=ns['ratio_and_errors'](a,**ns['oracle']); rb,sb,db=ns['ratio_and_errors'](b,**ns['oracle'])
        print('FP_SYS',a,b,'contrast derivative',da-db,'derivatives',da,db,'delta R',abs(ra-rb))
    # Check API invalid values without aborting the audit.
    for r,p0 in [(0,.5),(-.1,.5),(1.1,.5),(.8,0),(.8,1.1)]:
        try:
            with np.errstate(all='ignore'): val=ns['days_to_5sigma']('Fe','O',r,p0)
            print('INVALID_INPUT_ACCEPTED',r,p0,str(val))
        except Exception as ex: print('INVALID_INPUT_EXCEPTION',r,p0,type(ex).__name__)
