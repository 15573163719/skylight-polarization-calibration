"""Export additional method-example results from verified calibration outputs."""
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from demo_common import ROOT,read_config,wrap180
from calibration_rotation import group_headings,reference_steps


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):return float(np.sqrt(np.mean(np.asarray(values,dtype=float)**2)))


def validate_summary(summary,cfg,root=ROOT):
    if not summary.get('optimizer_success'):
        raise ValueError('Cannot export final example artifacts from an unconverged calibration')
    hashes=summary.get('source_sha256',{})
    if not hashes or any(not (root/name).is_file() or sha(root/name)!=digest for name,digest in hashes.items()):
        raise ValueError('Stale source hashes: rerun calibration before generating figures')
    for key,path in [('config_sha256',cfg['config_path']),('manifest_sha256',cfg['manifest_path']),
                     ('sensor_calibration_sha256',cfg['calibration_path'])]:
        if summary.get(key)!=sha(path):raise ValueError('Stale input: '+key)
    effective={k:v for k,v in cfg.items() if not isinstance(v,Path)}
    if summary.get('effective_config')!=effective:
        raise ValueError('This exporter expects the bundled example configuration, not an overridden ablation')
    if summary.get('optimizer_settings')!=cfg['optimizer']:
        raise ValueError('Changed optimizer settings: rerun the bundled example configuration')
    if summary.get('prior_selection',{}).get('policy')!=cfg['prior_policy']:
        raise ValueError('Changed prior policy: rerun the bundled example configuration')


def validate_frame_manifest(frames,cfg):
    with cfg['manifest_path'].open(encoding='utf-8') as handle:
        records=list(csv.DictReader(handle))
    if len(records)!=len(frames):raise ValueError('Frame manifest length mismatch')
    for frame,record in zip(frames,records):
        if frame['image']!=record['image']:
            raise ValueError('Frame image/order mismatch')
        if datetime.fromisoformat(frame['timestamp_utc'])!=datetime.fromisoformat(record['timestamp_utc']):
            raise ValueError('Frame timestamp mismatch')
        if float(frame['reference_heading_deg'])!=float(record['reference_heading_deg']):
            raise ValueError('Frame reference heading mismatch')


def consistency_from_frames(headings,truth,cfg,summary):
    if cfg['mode']=='rotation':
        grouped=group_headings(headings,cfg['frames_per_position'])
        if grouped is None:raise ValueError('Invalid circular means')
        return wrap180(np.diff(grouped)-reference_steps(cfg,truth))
    ref=summary['reference_frame_index']
    return np.delete(wrap180((headings-headings[ref])-(truth-truth[ref])),ref)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate(results_dir=None):
    results=Path(results_dir).resolve() if results_dir else ROOT/'results'
    panel_data={};table=[];inputs={}
    for mode in ['rotation','static']:
        cfg=read_config(mode)
        summary_path=results/(mode+'_summary.json')
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        validate_summary(summary,cfg)
        frame_path=results/(mode+'_frames.csv')
        with frame_path.open(encoding='utf-8') as handle:frames=list(csv.DictReader(handle))
        if len(frames)!=summary['frame_count']:raise ValueError('Frame count mismatch')
        validate_frame_manifest(frames,cfg)
        truth=np.array([float(r['reference_heading_deg']) for r in frames])
        values={};residual_rows=[]
        for state in ['before','after']:
            headings=np.array([float(r['heading_'+state+'_deg']) for r in frames])
            errors=np.array([float(r['error_'+state+'_deg']) for r in frames])
            if not np.isfinite(headings).all() or not np.isfinite(errors).all():raise ValueError('Nonfinite exported angles')
            np.testing.assert_allclose(wrap180(headings-truth),errors,atol=1e-10,rtol=0)
            residuals=consistency_from_frames(headings,truth,cfg,summary)
            np.testing.assert_allclose(rms(errors),summary[state]['fixed_bias_corrected_rmse_deg'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(rms(residuals),summary['data_residual_rms_'+state+'_deg'],atol=1e-10,rtol=0)
            params=summary['initial_intrinsics_px' if state=='before' else 'calibrated_intrinsics_px']
            table.append(dict(method=mode,state=state,f_px=params[0],u0_row_px=params[1],v0_column_px=params[2],
                              residual_weighting=cfg.get('residual_weighting','equal'),
                              focal_regularization_enabled=True,
                              consistency_rms_deg=rms(residuals),heading_rmse_deg=rms(errors),
                              heading_rmse_without_fixed_bias_deg=rms(wrap180(errors+cfg['fixed_bias_deg']))))
            values[state]=dict(errors=errors,residuals=residuals)
        if mode=='rotation':
            count=len(values['before']['residuals'])
            steps=reference_steps(cfg,truth)
            position_angle=np.r_[0.,np.cumsum(steps)]
            pair_x=(position_angle[:-1]+position_angle[1:])/2
            frame_x=np.repeat(position_angle,cfg['frames_per_position'])
            for i in range(count):
                residual_rows.append(dict(pair_index=i+1,position_start_deg=position_angle[i],
                    position_end_deg=position_angle[i+1],reference_rotation_deg=steps[i],
                    residual_before_deg=values['before']['residuals'][i],residual_after_deg=values['after']['residuals'][i]))
        else:
            local=[datetime.fromisoformat(r['timestamp_utc']).astimezone(timezone(timedelta(hours=8))) for r in frames]
            frame_x=np.array([d.hour+d.minute/60+d.second/3600+d.microsecond/3.6e9 for d in local])
            ref=summary['reference_frame_index'];indices=np.delete(np.arange(len(frames)),ref)
            pair_x=frame_x[indices]
            for i,index in enumerate(indices):
                residual_rows.append(dict(frame_index=int(index),reference_frame_index=ref,
                    timestamp_utc=frames[index]['timestamp_utc'],
                    residual_before_deg=values['before']['residuals'][i],residual_after_deg=values['after']['residuals'][i]))
        panel_data[mode]=dict(values=values,frame_x=frame_x,pair_x=pair_x,rows=residual_rows,cfg=cfg)
        inputs[mode]=dict(summary_sha256=sha(summary_path),frames_sha256=sha(frame_path),
                         source_sha256=summary['source_sha256'],config_sha256=summary['config_sha256'])
    out=results/'example_results';out.mkdir(exist_ok=True)
    write_csv(out/'example_summary.csv',table)
    lines=['# Additional method demonstration — generated from this run','',
           'Effective intrinsics use original-image pixels. Both initial vectors are [1300,360,640].',
           'These are reconstructed heading residuals, not pixel-level AoP errors.',
           'Both calibration modes include focal stabilization around the author-reported camera initial value of 1300 px; see README.md for the objective and input provenance.','',
           '| Method | State | f (px) | u0 row (px) | v0 column (px) | Consistency RMS (deg) | Heading RMSE (deg) | RMSE without fixed bias subtraction (deg) |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        lines.append(f"| {r['method']} | {r['state']} | {r['f_px']:.6f} | {r['u0_row_px']:.6f} | {r['v0_column_px']:.6f} | {r['consistency_rms_deg']:.6f} | {r['heading_rmse_deg']:.6f} | {r['heading_rmse_without_fixed_bias_deg']:.6f} |")
    lines+=['','Absolute errors use the declared +45 degree coordinate convention, identically before and after. Fixed bias subtraction is '+str({m:d['cfg']['fixed_bias_deg'] for m,d in panel_data.items()})+' degrees. No additional mounting correction is applied.',
            'Both before and after use the same fixed empirical AoP preprocessing from subsequent work. These results describe intrinsic calibration within that common processing pipeline.',
            'These additional examples demonstrate execution of the method; these selected subsets do not numerically reproduce the manuscript tables and figures.']
    (out/'example_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for mode,data in panel_data.items():write_csv(out/(mode+'_consistency_residuals.csv'),data['rows'])
    draw_figure(panel_data,out)
    (out/'example_caption.txt').write_text(
        f"Additional method demonstration. Calibration-set evaluation from {len(panel_data['rotation']['frame_x'])} rotation and {len(panel_data['static']['frame_x'])} static images. "
        '(a-c) Relative-rotation consistency residuals, reconstructed heading errors and their distributions. '
        '(d-f) Time-separated static counterparts. Rotation retains one frame per position and uses '
        '13 adjacent intervals, with reference kind '+panel_data['rotation']['cfg'].get('rotation_reference_kind','nominal_command')+'. '
        'Reference increments are frozen identically before and after calibration. INS-referenced calibration-set errors are not independent validation. '
        'Static residuals use matched INS heading differences relative to the fixed reference frame. Both modes use INS inputs frozen before fitting. Static time labels use actual manifest timestamps (UTC+8). Both modes retain the declared '
        '+45 degree coordinate convention; fixed evaluation biases are '+str({m:d['cfg']['fixed_bias_deg'] for m,d in panel_data.items()})+' degrees, without an additional mounting correction. '
        'Both conditions include the same empirical first-order dQ/dU AoP correction from subsequent work; its coefficients are fixed and their development-data overlap is unverified. '
        f"Objective weighting: rotation={panel_data['rotation']['cfg'].get('residual_weighting','equal')}, static={panel_data['static']['cfg'].get('residual_weighting','equal')}. "
        'Both modes include focal regularization. Anchors in pixels: '+str({m:d['cfg']['focal_regularization'].get('anchor_px',d['cfg']['initial_intrinsics_px'][0]) for m,d in panel_data.items()})+'. Scale and weight are engineering choices, not measured uncertainties. Focal proximity is not independent geometric validation. '
        'The plotted consistency residuals remain the original unweighted differences; the weighted fitting criterion, if used, is reported separately in the summaries. '
        'Boxes show quartiles and medians, whiskers extend to the most extreme observations within '
        '1.5 IQR, and points show every selected frame. Statistics are descriptive calibration-set results, '
        'not reproduction of the manuscript tables or figures.\n',encoding='utf-8')
    metadata=dict(inputs=inputs,generator_sha256=sha(Path(__file__)),
                  generated_from='Verified current-run frame outputs, not historical plot CSVs',
                  historical_table_reproduction_claimed=False,
                  outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='generation_manifest.json'})
    (out/'generation_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Additional example results written to',out,flush=True)
    return table


def draw_figure(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
        'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,'axes.spines.right':False,
        'axes.spines.top':False,'axes.linewidth':.8,'legend.frameon':False})
    colors=['#526A89','#BD7957']
    fig,axes=plt.subplots(2,3,figsize=(11.8,6.7),layout='constrained')
    for row,mode in enumerate(['rotation','static']):
        d=data[mode];states=d['values'];ax=axes[row,0]
        spacing=np.min(np.diff(d['pair_x']));width=.34*spacing
        for i,state in enumerate(['before','after']):
            vals=states[state]['residuals']
            ax.bar(d['pair_x']+(i-.5)*width,width=width,height=vals,color=colors[i],
                   label=f'{state.title()} (RMS {rms(vals):.3f}°)')
        ax.set_ylabel('Consistency residual (°)');ax.legend(fontsize=7.5,loc='best')
        ax=axes[row,1]
        for i,state in enumerate(['before','after']):
            vals=states[state]['errors']
            ax.plot(d['frame_x'],vals,'o-',ms=2.3,lw=.65,color=colors[i],
                    label=f'{state.title()} (RMSE {rms(vals):.3f}°)')
        ax.set_ylabel('Heading error (°)');ax.legend(fontsize=7.5,loc='best')
        ax=axes[row,2]
        samples=[states[s]['errors'] for s in ['before','after']]
        boxes=ax.boxplot(samples,tick_labels=['Before','After'],patch_artist=True,showfliers=False,widths=.42,
                         medianprops=dict(color='#222222',linewidth=1.2))
        for patch,color in zip(boxes['boxes'],colors):patch.set_facecolor(color);patch.set_alpha(.3)
        rng=np.random.default_rng(20261116)
        for i,(vals,color) in enumerate(zip(samples,colors)):
            ax.scatter(i+1+rng.uniform(-.12,.12,len(vals)),vals,s=7,color=color,alpha=.65,linewidths=0)
        ax.set_ylabel('Heading error (°)')
        for col in [0,1]:
            rotation_label='INS relative position (°)' if d['cfg'].get('rotation_reference_kind')=='matched_ins' else 'Nominal rotation position (°)'
            axes[row,col].set_xlabel(rotation_label if mode=='rotation' else 'Local time (UTC+8)')
            if mode=='static':
                axes[row,col].xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(round(x*60))//60:02d}:{int(round(x*60))%60:02d}'))
            else:
                axes[row,col].set_xticks(np.arange(0,361,60))
        for col in range(3):
            axes[row,col].axhline(0,color='#999999',lw=.6,ls='--',zorder=0)
            axes[row,col].text(-.12,1.05,chr(97+row*3+col),transform=axes[row,col].transAxes,fontweight='bold',fontsize=12)
            axes[row,col].set_title(('Relative rotation' if mode=='rotation' else 'Time-separated static') if col==0 else '',fontsize=10)
    fig.suptitle(f"Additional method examples ({len(data['rotation']['frame_x'])} rotation / {len(data['static']['frame_x'])} static frames)",fontsize=11)
    for extension in ['png','pdf','svg']:fig.savefig(out/('example_calibration.'+extension),dpi=300)
    plt.close(fig)


if __name__=='__main__':generate()
