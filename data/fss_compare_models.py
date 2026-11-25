#!/usr/bin/env python3
"""Local model comparison on the same six FSS samples; never replace originals.

uv run python data/fss_compare_models.py --backend all
All workers use the project root Python environment.
Parakeet's Korean NGC/NIM prerequisites are recorded, not replaced with English weights.
"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

if __package__:
    from . import fss_transcriber as stt
    from . import fss_compare_quality as quality
else:
    import fss_transcriber as stt
    import fss_compare_quality as quality

LOG=logging.getLogger('fss_compare_models')
V2_REVISION='f0fe81560cb8b68660e564f55dd99207059c092e'
QWEN_ID='Qwen/Qwen3-ASR-1.7B-hf'
QWEN_REVISION='bcd2b5b7f32b480ab5790554cfa8347f246a14f3'
ALIGN_ID='Qwen/Qwen3-ForcedAligner-0.6B-hf'
ALIGN_REVISION='c07281df297b9905d24a508279258cccf987a064'
LABELS=['whisper_large_v3_vad','whisper_large_v2_vad','qwen3_asr_1_7b']
CHUNK_SEC=30
TOKEN_RECOVERY=dict(strategy='bisect_token_limited_chunks_and_rebalance_short_tail',max_depth=3,min_chunk_sec=3.75,
                    rebalance_short_tail_below_sec=3.75)
TIMESTAMP_RECOVERY=dict(TOKEN_RECOVERY,rebalance_short_tail_below_sec=30.)
TIMESTAMP_RETRY=dict(TIMESTAMP_RECOVERY,min_chunk_sec=1.,retry_alignment_outside_audio=True)
RECOVERY_POLICIES=(TOKEN_RECOVERY,TIMESTAMP_RECOVERY,TIMESTAMP_RETRY)


def input_chunk_windows(frames,sample_rate,policy=None):
    windows=[(start,min(start+CHUNK_SEC*sample_rate,frames),0) for start in range(0,frames,CHUNK_SEC*sample_rate)]
    if policy and len(windows)>1 and windows[-1][1]-windows[-1][0]<round(policy['rebalance_short_tail_below_sec']*sample_rate):
        start=windows[-2][0];end=windows[-1][1];middle=(start+end)//2
        windows[-2:]=[(start,middle,0),(middle,end,0)]
    return windows


def split_token_limited_chunk(start,end,sample_rate,depth,policy):
    if depth>=policy['max_depth'] or end-start<2*round(policy['min_chunk_sec']*sample_rate):
        return None
    middle=(start+end)//2
    return [(start,middle,depth+1),(middle,end,depth+1)]


def selected(root,all_files=False):
    result=[]
    manifest={x['id']:x for x in [json.loads(line) for line in (root/'transcription_manifest.jsonl').read_text().splitlines()]}
    samples=[(p.stem,'full_dataset') for p in stt.discover(root)] if all_files else quality.SAMPLES
    for id,group in samples:
        jp=root/'transcripts'/(id+'.json');data=stt.read_json(jp)
        assert stt.sha256(jp)==manifest[id]['json_sha256'], 'Original JSON changed: '+id
        assert stt.sha256(jp.with_suffix('.txt'))==manifest[id]['txt_sha256'], 'Original TXT changed: '+id
        video=root/data['video_path'];wav=root/data['audio_path']
        source_hash=stt.sha256(video);audio_hash=stt.sha256(wav)
        if source_hash!=data['source_sha256'] or audio_hash!=data['audio_sha256']:
            raise ValueError('Original source/audio changed: '+id)
        duration=stt.validate_wav(wav,stt.audio_duration(stt.probe(video),video))
        result.append((id,group,video,dict(path=wav,duration_sec=duration,source_sha256=source_hash,audio_sha256=audio_hash)))
    return result


def settings(backend,env):
    if backend=='whisper-v2':
        sig=stt.signature(stt.stt_settings(True),env)
        sig.update(model='large-v2',model_repository='Systran/faster-whisper-large-v2',model_revision=V2_REVISION)
        return sig
    return dict(schema_version=1,model=QWEN_ID,model_revision=QWEN_REVISION,
      alignment_model=ALIGN_ID,alignment_revision=ALIGN_REVISION,
      backend='transformers',device='cuda',compute_type='float16',packages=env['packages'],
      settings=dict(language='Korean',prompt=None,do_sample=False,num_beams=1,max_new_tokens=1024,
                    chunk_sec=CHUNK_SEC,chunk_overlap_sec=0,vad_filter=False,attn_implementation='sdpa'))


def torch_environment():
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise stt.EnvironmentFailure('PyTorch CUDA unavailable; no CPU fallback')
    x=torch.ones((64,64),device='cuda',dtype=torch.float16)
    if not torch.isfinite(x@x).all():raise stt.EnvironmentFailure('CUDA FP16 verification failed')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    return dict(python=sys.version.split()[0],gpu=stt.gpu_sample(),torch_cuda=torch.version.cuda,
      compute_capability=list(torch.cuda.get_device_capability()),packages={n:version(n) for n in ('torch','transformers','accelerate','soynlp','soundfile','librosa')})


def qwen_load():
    import torch
    from transformers import AutoProcessor,AutoModelForMultimodalLM,AutoModelForTokenClassification
    common=dict(local_files_only=True,dtype=torch.float16,attn_implementation='sdpa')
    processor=AutoProcessor.from_pretrained(QWEN_ID,revision=QWEN_REVISION,local_files_only=True)
    asr=AutoModelForMultimodalLM.from_pretrained(QWEN_ID,revision=QWEN_REVISION,**common).to('cuda').eval()
    align_processor=AutoProcessor.from_pretrained(ALIGN_ID,revision=ALIGN_REVISION,local_files_only=True)
    aligner=AutoModelForTokenClassification.from_pretrained(ALIGN_ID,revision=ALIGN_REVISION,**common).to('cuda').eval()
    for model in (asr,aligner):
        if model.device.type!='cuda' or model.dtype!=torch.float16:
            raise stt.EnvironmentFailure('Model is not CUDA FP16')
    return processor,asr,align_processor,aligner


def word_character_spans(text,words):
    """Map exact aligner tokens back to raw text without changing either string."""
    cursor=0
    for word in words:
        position=text.find(word['word'],cursor)
        if position<0:
            word['transcript_char_start']=None;word['transcript_char_end']=None
        else:
            word['transcript_char_start']=position
            word['transcript_char_end']=position+len(word['word'])
            cursor=position+len(word['word'])


def qwen_transcribe(models,audio,video,sig,out):
    import torch
    import soundfile as sf
    processor,asr,align_processor,aligner=models
    waveform,sr=sf.read(str(audio['path']),dtype='float32')
    if sr!=16000 or waveform.ndim!=1:raise ValueError('Qwen input must be mono 16kHz')
    segments=[];chunks=[];discarded=[]
    policy=sig['settings'].get('token_limit_recovery')
    pending=collections.deque(input_chunk_windows(len(waveform),sr,policy))
    while pending:
        frame,end_frame,depth=pending.popleft();index=len(chunks)
        chunk=waveform[frame:end_frame];offset=frame/sr
        LOG.info('%s Qwen chunk %d (%.3f-%.3fs, split depth %d)',video.stem,index+1,offset,end_frame/sr,depth)
        chunk_started=time.perf_counter()
        inputs=processor.apply_transcription_request(chunk,language='Korean',
          processor_kwargs={'audio_kwargs':{'sampling_rate':sr}}).to('cuda',torch.float16)
        torch.cuda.synchronize();asr_started=time.perf_counter()
        with torch.inference_mode():
            ids=asr.generate(**inputs,max_new_tokens=1024,do_sample=False,num_beams=1)
        torch.cuda.synchronize();asr_elapsed=time.perf_counter()-asr_started
        generated=ids[:,inputs['input_ids'].shape[1]:]
        eos=asr.generation_config.eos_token_id
        eos_ids=set(eos if isinstance(eos,list) else [eos])
        text=processor.decode(generated,return_format='transcription_only')[0]
        raw=processor.decode(generated)[0]
        if generated.shape[1]>=1024 and int(generated[0,-1]) not in eos_ids:
            discarded.append(dict(start_frame=frame,end_frame=end_frame,split_depth=depth,text=text,
                raw_decoded_output=raw,generated_tokens=int(generated.shape[1]),asr_time_sec=asr_elapsed,
                processing_time_sec=time.perf_counter()-chunk_started,reason='token_limit_without_eos'))
            stt.atomic_text(out/(video.stem+'.partial.json'),stt.json_text(dict(status='token_limit',
                source_sha256=audio['source_sha256'],reproducibility=sig,chunks=chunks,discarded_generation_attempts=discarded)))
            pieces=split_token_limited_chunk(frame,end_frame,sr,depth,policy) if policy else None
            if pieces:
                LOG.warning('%s splitting token-limited %.3f-%.3fs without changing model text',video.stem,offset,end_frame/sr)
                pending.extendleft(reversed(pieces));continue
            raise ValueError('Qwen output reached token limit; refusing silent truncation')
        words=[];raw_alignment=[];alignment_elapsed=0.
        if text.strip():
            align_started=time.perf_counter()
            ai,word_lists=align_processor.prepare_forced_aligner_inputs(audio=chunk,transcript=text,language='Korean',
                processor_kwargs={'audio_kwargs':{'sampling_rate':sr}})
            ai=ai.to('cuda',torch.float16)
            with torch.inference_mode():aligned=aligner(**ai)
            torch.cuda.synchronize()
            raw_alignment=align_processor.decode_forced_alignment(logits=aligned.logits,input_ids=ai['input_ids'],
                word_lists=word_lists,timestamp_token_id=aligner.config.timestamp_token_id)[0]
            alignment_elapsed=time.perf_counter()-align_started
            words=[dict(start=float(w['start_time'])+offset,end=float(w['end_time'])+offset,
                        word=w['text'],probability=None) for w in raw_alignment]
            word_character_spans(text,words)
        if policy and policy.get('retry_alignment_outside_audio') and any(w['start']<0 or w['end']>audio['duration_sec']+1/sr for w in words):
            discarded.append(dict(start_frame=frame,end_frame=end_frame,split_depth=depth,text=text,
                raw_decoded_output=raw,raw_alignment=raw_alignment,generated_tokens=int(generated.shape[1]),
                asr_time_sec=asr_elapsed,alignment_time_sec=alignment_elapsed,
                processing_time_sec=time.perf_counter()-chunk_started,reason='alignment_outside_source_audio'))
            stt.atomic_text(out/(video.stem+'.partial.json'),stt.json_text(dict(status='alignment_retry',
                source_sha256=audio['source_sha256'],reproducibility=sig,chunks=chunks,discarded_generation_attempts=discarded)))
            pieces=split_token_limited_chunk(frame,end_frame,sr,depth,policy)
            if pieces:
                LOG.warning('%s splitting alignment-invalid %.3f-%.3fs',video.stem,offset,end_frame/sr)
                pending.extendleft(reversed(pieces));continue
            raise ValueError('Alignment timestamp outside audio after bounded recovery')
        segment=dict(id=index,start=min((w['start'] for w in words),default=offset),
                     end=max((w['end'] for w in words),default=(frame+len(chunk))/sr),text=text,
                     avg_logprob=None,no_speech_prob=None,compression_ratio=None,words=words,
                     timestamp_source=ALIGN_ID if words else 'input_chunk_bounds_no_alignment')
        segments.append(segment)
        chunks.append(dict(index=index,start_frame=frame,end_frame=frame+len(chunk),sample_rate=sr,
                           split_depth=depth,
                           text=text,raw_decoded_output=raw,raw_alignment=raw_alignment,
                           generated_tokens=int(generated.shape[1]),asr_time_sec=asr_elapsed,
                           alignment_time_sec=alignment_elapsed,processing_time_sec=time.perf_counter()-chunk_started))
        stt.atomic_text(out/(video.stem+'.partial.json'),stt.json_text(dict(status='running',source_sha256=audio['source_sha256'],reproducibility=sig,chunks=chunks,discarded_generation_attempts=discarded)))
    elapsed=sum(x['processing_time_sec'] for x in chunks+discarded)
    return dict(id=video.stem,source='FSS',video_path='raw/'+video.name,audio_path='audio/'+audio['path'].name,
      language='ko',model=QWEN_ID,backend='transformers',device='cuda',compute_type='float16',
      duration_sec=audio['duration_sec'],processing_time_sec=elapsed,rtf=elapsed/audio['duration_sec'],
      asr_time_sec=sum(x['asr_time_sec'] for x in chunks+discarded),alignment_time_sec=sum(x.get('alignment_time_sec',0) for x in chunks+discarded),
      text=''.join(s['text'] for s in segments),segments=segments,chunks=chunks,discarded_generation_attempts=discarded,
      empty_transcript=not any(s['text'].strip() for s in segments),
      source_sha256=audio['source_sha256'],audio_sha256=audio['audio_sha256'],reproducibility=sig,
      timestamp_note='Forced alignment timestamps, offset to source audio. Segment text is unmodified per ASR input chunk; see exact frame bounds and any recovery policy. Boundaries can split speech.',
      created_at=datetime.now(timezone.utc).isoformat())


def worker(backend,root,out,force=False,all_files=False,recover_token_limit=False):
    directory=out/('transcripts' if all_files else 'whisper_large_v2_vad' if backend=='whisper-v2' else 'qwen3_asr_1_7b');directory.mkdir(parents=True,exist_ok=True)
    summary_path=directory/'run_summary.json'
    old=stt.read_json(summary_path) if summary_path.exists() else {}
    old_rows={x['id']:x for x in old.get('results',[])}
    manifest_path=directory/'transcription_manifest.jsonl'
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            record=json.loads(line);old_rows[record['id']]=record
    records=dict(old_rows)
    comparison=root/'model_comparison/v1/qwen3_asr_1_7b'
    comparison_rows={}
    if all_files and (comparison/'run_summary.json').exists():
        comparison_rows={x['id']:x for x in stt.read_json(comparison/'run_summary.json').get('results',[])}
    summary=dict(status='running',backend=backend,started_at=datetime.now(timezone.utc).isoformat(),
                 results=[],model_load_count=0,model_load_time_sec=0.,new_inference_time_sec=0.,all_files=all_files,
                 recover_token_limit=recover_token_limit)
    before=quality.protected_snapshot(root);monitor=None;model=None;started=time.perf_counter()
    def save():
        stt.atomic_text(summary_path,stt.json_text(summary))
        stt.atomic_text(manifest_path,''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in records.values()))
    try:
        env=stt.environment() if backend=='whisper-v2' else torch_environment()
        summary['environment']=env;sig=settings(backend,env);summary['reproducibility']=sig
        monitor=stt.GpuMonitor();monitor.thread.start()
        items=selected(root,all_files=all_files)
        summary['total_files']=len(items)
        for id,group,video,audio in items:
            jp=directory/(id+'.json');tp=jp.with_suffix('.txt');row=dict(id=id,group=group)
            try:
                previous_record=old_rows.get(id,{})
                policy=previous_record.get('recovery_policy')
                previous_error=previous_record.get('error','').lower()
                recoverable='token limit' in previous_error or ('timestamp' in previous_error and 'outside audio' in previous_error)
                if recover_token_limit and previous_record.get('status')=='failed' and recoverable:
                    policy=(TIMESTAMP_RETRY if policy in (TIMESTAMP_RECOVERY,TIMESTAMP_RETRY) else TIMESTAMP_RECOVERY) if 'timestamp' in previous_error else TOKEN_RECOVERY
                file_sig=copy.deepcopy(sig)
                if policy:
                    if policy not in RECOVERY_POLICIES:raise ValueError('Unsupported recorded recovery policy')
                    file_sig['settings']['token_limit_recovery']=policy;row['recovery_policy']=policy
                data=None if force else stt.reusable(jp,tp,audio,file_sig,video,previous_record)
                if data is None and all_files and not force:
                    previous=comparison/(id+'.json')
                    data=stt.reusable(previous,previous.with_suffix('.txt'),audio,file_sig,video,comparison_rows.get(id,{}))
                    if data is not None:
                        stt.atomic_text(tp,data['text']);stt.atomic_text(jp,stt.json_text(data))
                        row['reused_from']=str(previous.relative_to(root))
                if data is None:
                    if model is None:
                        load_started=time.perf_counter()
                        if backend=='whisper-v2':
                            from faster_whisper import WhisperModel
                            model=WhisperModel('large-v2',device='cuda',compute_type='float16',revision=V2_REVISION,local_files_only=True,num_workers=1)
                            if model.model.device!='cuda' or model.model.compute_type!='float16':raise stt.EnvironmentFailure('Whisper v2 must use CUDA FP16')
                        else:model=qwen_load()
                        summary['model_load_time_sec']=time.perf_counter()-load_started;summary['model_load_count']=1
                    LOG.info('%s %s transcribing',id,backend)
                    if backend=='whisper-v2':
                        data=stt.transcribe(model,audio,video,file_sig);data['model']='large-v2'
                    else:data=qwen_transcribe(model,audio,video,file_sig,directory)
                    summary['new_inference_time_sec']+=data['processing_time_sec']
                    try:data['timestamp_warnings']=stt.validate_transcript(data,data['text'],audio['duration_sec'])
                    except (ValueError,KeyError,TypeError) as exc:
                        stt.atomic_text(jp.with_suffix('.invalid.json'),stt.json_text(dict(data,validation_error=str(exc))));raise
                    stt.atomic_text(tp,data['text']);stt.atomic_text(jp,stt.json_text(data))
                    jp.with_suffix('.partial.json').unlink(missing_ok=True)
                    row['status']='ok'
                else:row['status']='reused'
                row.update(json_sha256=stt.sha256(jp),txt_sha256=stt.sha256(tp),characters=len(data['text']),
                           segments=len(data['segments']),words=sum(len(s['words']) for s in data['segments']),
                           duration_sec=audio['duration_sec'],processing_time_sec=data['processing_time_sec'],rtf=data['rtf'])
                LOG.info('%s %s %s: %d characters, RTF %.4f',id,backend,row['status'],row['characters'],row['rtf'])
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                LOG.error('%s %s failed: %s',id,backend,exc)
                if stt.gpu_failure(exc) or isinstance(exc,(stt.EnvironmentFailure,OSError)):
                    summary['results'].append(row);records[id]=dict(row);save();raise
            summary['results'].append(row);records[id]=dict(row);save()
        summary['status']='completed_with_errors' if any(x['status']=='failed' for x in summary['results']) else 'completed'
    except KeyboardInterrupt:summary['status']='interrupted'
    except Exception as exc:summary.update(status='aborted',fatal_error=f'{type(exc).__name__}: {exc}')
    finally:
        if monitor:
            monitor.close();summary.update(gpu_max_memory_used_mib=monitor.max_memory,gpu_max_utilization_percent=monitor.max_util,
                                          gpu_measurement_scope='Device 0 total, including other processes')
        if backend=='qwen':
            import torch
            summary['torch_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        summary['protected_originals_unchanged']=before==quality.protected_snapshot(root)
        if not summary['protected_originals_unchanged']:summary.update(status='aborted',fatal_error='Original results changed')
        complete=[x for x in summary['results'] if x['status'] in ('ok','reused')]
        fresh=[x for x in complete if x['status']=='ok']
        fresh_duration=sum(x['duration_sec'] for x in fresh)
        summary.update(stt_success=len(fresh),reused_count=sum(x['status']=='reused' for x in complete),
          stt_failed=sum(x['status']=='failed' for x in summary['results']),
          total_audio_duration_sec=sum(x['duration_sec'] for x in complete),
          newly_inferred_audio_duration_sec=fresh_duration,
          overall_rtf=summary['new_inference_time_sec']/fresh_duration if fresh_duration else None,
          total_characters=sum(x['characters'] for x in complete),total_segments=sum(x['segments'] for x in complete),
          total_words=sum(x['words'] for x in complete))
        summary['pipeline_time_sec']=time.perf_counter()-started;summary['finished_at']=datetime.now(timezone.utc).isoformat();save()
        runs=directory/'runs';runs.mkdir(exist_ok=True)
        stt.atomic_text(runs/(summary['started_at'].replace(':','').replace('+','_')+'.json'),stt.json_text(summary))
    return 0 if summary['status']=='completed' else 130 if summary['status']=='interrupted' else 1


def qwen_sample_text(data,start,end):
    snippets=[];fallback=False
    for segment in data['segments']:
        chosen=[w for w in segment['words'] if start<=(w['start']+w['end'])/2<end]
        if not chosen:continue
        if any(w.get('transcript_char_start') is None for w in chosen):
            snippets.append(segment['text']);fallback=True
        else:
            a=chosen[0]['transcript_char_start'];b=chosen[-1]['transcript_char_end']
            if b==max((w.get('transcript_char_end') or 0 for w in segment['words']),default=0):b=len(segment['text'])
            snippets.append(segment['text'][a:b])
    return ''.join(snippets),fallback


def package(root,out):
    all_data={};status={}
    existing=root/'quality_comparison/v1/B_vad_on'
    existing_summary=stt.read_json(root/'quality_comparison/v1/comparison_summary.json')
    existing_rows={(x['id'],x['variant']):x for x in existing_summary['results']}
    for id,_ in quality.SAMPLES:
        all_data[id]={}
        for label in LABELS:
            jp=(existing if label=='whisper_large_v3_vad' else out/label)/(id+'.json')
            if jp.exists():
                data=stt.read_json(jp);stt.validate_transcript(data,jp.with_suffix('.txt').read_text(),data['duration_sec'])
                if label=='whisper_large_v3_vad':record=existing_rows[(id,'B_vad_on')]
                else:record=next(x for x in stt.read_json(out/label/'run_summary.json')['results'] if x['id']==id)
                if stt.sha256(jp)!=record.get('json_sha256') or stt.sha256(jp.with_suffix('.txt'))!=record.get('txt_sha256'):
                    raise ValueError('Comparison output hash mismatch: '+str(jp))
                all_data[id][label]=data
    for label in LABELS:
        status[label]=dict(completed_files=sum(label in x for x in all_data.values()))
    parakeet=dict(status='unavailable',model='Parakeet RNNT 1.1B Multilingual (Korean)',
      reasons=['NGC access probes returned HTTP 401 Unauthorized; no authorized Korean checkpoint acquired.',
               'Docker and NGC CLI are not installed in this WSL environment.',
               'Official NIM offline profile is about 25.56 GB before overhead; GPU has 16 GB.'],
      source='https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-rnnt.html',
      note='The public Hugging Face nvidia/parakeet-rnnt-1.1b checkpoint is English-only and was not substituted. No cloud audio API was called.')
    stt.atomic_text(out/'parakeet_status.json',stt.json_text(parakeet))
    full=io.StringIO(newline='');writer=csv.DictWriter(full,fieldnames=['id','group','duration_sec',*LABELS,'parakeet_status']);writer.writeheader()
    text_directory=out/'comparison_text';text_directory.mkdir(exist_ok=True)
    review=io.StringIO(newline='');rw=csv.DictWriter(review,fieldnames=['id','position','start_sec','end_sec','audio_path',*LABELS,'qwen_sample_has_coarse_boundary','reviewed','reference_text']);rw.writeheader()
    for id,group in quality.SAMPLES:
        variants=all_data[id];first=next(iter(variants.values()))
        writer.writerow(dict(id=id,group=group,duration_sec=first['duration_sec'],parakeet_status='unavailable',
                             **{label:variants[label]['text'] if label in variants else '' for label in LABELS}))
        sections=[f'[{id}]']
        for label in LABELS:
            sections.extend([f'[{label}]',variants[label]['text'] if label in variants else '(No validated output available)'])
        sections.extend(['[parakeet]','Not run: see parakeet_status.json'])
        stt.atomic_text(text_directory/(id+'.txt'),'\n\n'.join(sections)+'\n')
        for position,a,b in quality.sample_windows(first['duration_sec']):
            row=dict(id=id,position=position,start_sec=round(a,6),end_sec=round(b,6),audio_path=f'review_audio/{id}_{position}.wav',reviewed=0,reference_text='')
            for label in LABELS:
                if label not in variants:row[label]=''
                elif label=='qwen3_asr_1_7b':row[label],row['qwen_sample_has_coarse_boundary']=qwen_sample_text(variants[label],a,b)
                else:row[label]=quality.sample_text(variants[label],a,b)
            rw.writerow(row)
    stt.atomic_text(out/'full_text_comparison.csv','\ufeff'+full.getvalue())
    if not (out/'review_samples.csv').exists():stt.atomic_text(out/'review_samples.csv','\ufeff'+review.getvalue())
    quality.write_review_audio(root,out,all_data)
    stt.atomic_text(out/'models_summary.json',stt.json_text(dict(status='partial_requested_models_unavailable',
      all_requested_models_completed=False,models=status,parakeet=parakeet,human_accuracy_verified=False,cer=None,
      sample_ids=[id for id,_ in quality.SAMPLES],note='Qwen uses non-overlapping 30 second chunks and a separate forced aligner. Whisper uses sequential decoding with VAD. This is a pipeline comparison, not an isolated architecture benchmark.')))
    LOG.info('Comparison package ready at %s',out)


def audit_full_qwen(root,out,archive=False):
    """Validate the completed corpus without GPU inference or transcript edits."""
    directory=out/'transcripts'
    summary=stt.read_json(directory/'run_summary.json')
    records={x['id']:x for x in [json.loads(line) for line in (directory/'transcription_manifest.jsonl').read_text().splitlines()]}
    original={x['id']:x for x in [json.loads(line) for line in (root/'transcription_manifest.jsonl').read_text().splitlines()]}
    metadata={x['record_id']:x for x in [json.loads(line) for line in (root/'metadata.jsonl').read_text().splitlines()] if x.get('status')=='ok'}
    rows=[]
    for video in stt.discover(root):
        row=dict(id=video.stem,integrity_errors=[])
        try:
            jp=directory/(video.stem+'.json');tp=jp.with_suffix('.txt');data=stt.read_json(jp)
            record=records[video.stem];audio=root/data['audio_path']
            if record['status'] not in ('ok','reused'):raise ValueError('Manifest does not mark successful transcription')
            if stt.sha256(jp)!=record['json_sha256'] or stt.sha256(tp)!=record['txt_sha256']:raise ValueError('Output SHA mismatch')
            if stt.sha256(video)!=data['source_sha256'] or data['source_sha256']!=metadata[video.stem]['sha256']:raise ValueError('Source SHA mismatch')
            if stt.sha256(audio)!=data['audio_sha256']:raise ValueError('Audio SHA mismatch')
            expected=copy.deepcopy(summary['reproducibility'])
            recovery=data['reproducibility']['settings'].get('token_limit_recovery')
            if recovery:
                if recovery not in RECOVERY_POLICIES or recovery!=record.get('recovery_policy'):raise ValueError('Unverified recovery settings')
                expected['settings']['token_limit_recovery']=recovery
            if data['reproducibility']!=expected:raise ValueError('STT settings mismatch')
            if data['model']!=QWEN_ID or data['device']!='cuda' or data['compute_type']!='float16':raise ValueError('Incorrect model/device/precision')
            duration=stt.validate_wav(audio,stt.audio_duration(stt.probe(video),video))
            warnings=stt.validate_transcript(data,tp.read_text(encoding='utf-8'),duration)
            cursor=0
            for i,chunk in enumerate(data['chunks']):
                if chunk['index']!=i or chunk['start_frame']!=cursor or chunk['sample_rate']!=16000:raise ValueError('Invalid chunk sequence')
                if not cursor<chunk['end_frame']<=cursor+CHUNK_SEC*16000:raise ValueError('Invalid chunk extent')
                cursor=chunk['end_frame']
            if cursor!=round(duration*16000) or data['text']!=''.join(x['text'] for x in data['chunks']):raise ValueError('Incomplete chunk/audio/text coverage')
            if len(data['segments'])!=len(data['chunks']):raise ValueError('Chunk/segment count mismatch')
            old=root/'transcripts'/(video.stem+'.json')
            if stt.sha256(old)!=original[video.stem]['json_sha256'] or stt.sha256(old.with_suffix('.txt'))!=original[video.stem]['txt_sha256']:
                raise ValueError('Original Whisper result changed')
            counts=collections.Counter(x['text'].strip() for x in data['segments'] if x['text'].strip())
            maximum=max(counts.values(),default=0);flags=[]
            empty=not data['text'].strip()
            if empty:flags.append('empty_transcript')
            if len(data['text'])/duration<2:flags.append('low_text_density')
            if maximum>=3 and maximum/max(1,len(data['segments']))>=.5:flags.append('dominant_repeated_chunk')
            if any(x in data['text'] for x in ('자막 제공','자막제공','한글자막','시청해 주셔서','시청해주셔서')):flags.append('subtitle_or_outro_phrase')
            row.update(duration_sec=duration,processing_time_sec=data['processing_time_sec'],rtf=data['rtf'],
                       asr_time_sec=data['asr_time_sec'],alignment_time_sec=data['alignment_time_sec'],
                       characters=len(data['text']),segments=len(data['segments']),words=sum(len(x['words']) for x in data['segments']),
                       empty_transcript=empty,timestamp_warning_count=len(warnings),review_flags=flags,text=data['text'],
                       token_limit_recovery_attempts=sum(x.get('reason')=='token_limit_without_eos' for x in data.get('discarded_generation_attempts',[])),
                       alignment_recovery_attempts=sum(x.get('reason')=='alignment_outside_source_audio' for x in data.get('discarded_generation_attempts',[])),
                       recovery_policy_applied=bool(recovery))
        except Exception as exc:row['integrity_errors'].append(f'{type(exc).__name__}: {exc}')
        rows.append(row)
    passed=[x for x in rows if not x['integrity_errors']]
    report=dict(status='passed_integrity' if len(passed)==len(rows) else 'failed_integrity',
      total_files=len(rows),integrity_pass=len(passed),failed_files=[x for x in rows if x['integrity_errors']],
      total_audio_duration_sec=sum(x['duration_sec'] for x in passed),total_characters=sum(x['characters'] for x in passed),
      total_segments=sum(x['segments'] for x in passed),total_words=sum(x['words'] for x in passed),
      empty_transcripts=sum(x['empty_transcript'] for x in passed),
      total_timestamp_warnings=sum(x['timestamp_warning_count'] for x in passed),
      token_limit_recovered_files=[x['id'] for x in passed if x['token_limit_recovery_attempts']],
      recovery_files=[x['id'] for x in passed if x['recovery_policy_applied']],
      flagged_files=sum(bool(x['review_flags']) for x in passed),human_accuracy_verified=False,cer=None,
      note='Integrity checks and review heuristics only. Raw model strings are unchanged. 30-second boundaries can split speech.',
      files=[{k:v for k,v in x.items() if k!='text'} for x in rows])
    stt.atomic_text(out/'audit_summary.json',stt.json_text(report))
    buffer=io.StringIO(newline='')
    fields=['id','duration_sec','processing_time_sec','rtf','asr_time_sec','alignment_time_sec','characters','segments','words',
            'empty_transcript','timestamp_warning_count','token_limit_recovery_attempts','alignment_recovery_attempts','recovery_policy_applied','review_flags','integrity_errors','text']
    writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    stt.atomic_text(out/'qwen_transcripts.csv','\ufeff'+buffer.getvalue())
    if archive and report['status']=='passed_integrity':
        target=stt.ROOT/f'fss_qwen_{len(rows)}.zip'
        fd,name=tempfile.mkstemp(prefix='.fss_qwen_',suffix='.zip',dir=target.parent);os.close(fd)
        try:
            with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for row in rows:
                    for suffix in ('.json','.txt'):
                        p=directory/(row['id']+suffix);z.write(p,Path('qwen')/'transcripts'/p.name)
                for p in (out/'audit_summary.json',out/'qwen_transcripts.csv',directory/'run_summary.json',directory/'transcription_manifest.jsonl'):
                    z.write(p,Path('qwen')/p.relative_to(out))
                for p in sorted((directory/'runs').glob('*.json')):
                    z.write(p,Path('qwen')/p.relative_to(out))
            with zipfile.ZipFile(name) as z:
                if z.testzip() is not None:raise ValueError('Archive CRC check failed')
            os.replace(name,target)
        finally:Path(name).unlink(missing_ok=True)
        LOG.info('Archive ready: %s (%.2f MiB)',target,target.stat().st_size/1024**2)
    LOG.info('Audit: %d/%d passed, %d review flags, %d empty transcripts',len(passed),len(rows),report['flagged_files'],report['empty_transcripts'])
    return 0 if report['status']=='passed_integrity' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['all','whisper-v2','qwen','package','audit'],default='all')
    p.add_argument('--input',type=Path,default=stt.DEFAULT_INPUT)
    p.add_argument('--force',action='store_true',help='Repeat only new comparison inference')
    p.add_argument('--all-files',action='store_true',help='Qwen only: process every input into INPUT/qwen/transcripts; preserve comparison results')
    p.add_argument('--archive',action='store_true',help='With --backend audit --all-files: create a local ZIP after integrity checks pass')
    p.add_argument('--recover-token-limit','--recover-failed',dest='recover_token_limit',action='store_true',
                   help='Qwen full run only: recover previous token-limit/timestamp-bound failures with shorter chunks; preserve provenance')
    args=p.parse_args();root=args.input.expanduser().resolve();out=root/'model_comparison/v1'
    if args.all_files:
        if args.backend not in ('qwen','audit'):p.error('--all-files requires --backend qwen or audit')
        out=root/'qwen'
    if args.backend=='audit' and not args.all_files:p.error('--backend audit requires --all-files')
    if args.archive and args.backend!='audit':p.error('--archive requires --backend audit --all-files')
    if args.recover_token_limit and (args.backend!='qwen' or not args.all_files):p.error('--recover-token-limit requires --backend qwen --all-files')
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['DO_NOT_TRACK']='1'
    if args.backend=='whisper-v2':stt.prepare_cuda_libraries(reexec_script=__file__)
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
      handlers=[logging.StreamHandler(),logging.FileHandler(out/(args.backend+'.log'),encoding='utf-8')])
    if args.backend=='all':
        commands=[[sys.executable,str(Path(__file__).resolve()),'--backend',backend]
                  for backend in ('whisper-v2','qwen')]
        codes=[]
        for command in commands:
            command+=['--input',str(root)]
            if args.force:command+=['--force']
            child_env=dict(os.environ);child_env.pop('VIRTUAL_ENV',None)
            codes.append(subprocess.call(command,cwd=stt.ROOT.parent,env=child_env))
        with (root/'.transcription.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);package(root,out)
        return 1 if any(codes) else 0
    with (root/'.transcription.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.backend=='audit':return audit_full_qwen(root,out,args.archive)
        if args.backend=='package':package(root,out);return 0
        return worker(args.backend,root,out,args.force,all_files=args.all_files,recover_token_limit=args.recover_token_limit)


if __name__=='__main__':sys.exit(main())
