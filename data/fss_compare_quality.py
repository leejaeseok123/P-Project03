#!/usr/bin/env python3
"""Compare four large-v3 CUDA FP16 settings without replacing FSS transcripts.

Run: uv run python data/fss_compare_quality.py
Outputs stay under the ignored FSS dataset. No automatic accuracy verdict or
reference text is generated. Use --references with human-reviewed samples for CER.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import unicodedata
import wave

if __package__:
    from . import fss_transcriber as stt
else:
    import fss_transcriber as stt

LOG = logging.getLogger('fss_compare_quality')
SAMPLES = [
    ('217131_b080fc464f05', 'severe_repetition'),
    ('18323_f122cf1cf432', 'severe_repetition'),
    ('18255_ac3e79605e22', 'low_text_density'),
    ('18282_dbd3fa1f08ad', 'low_text_density'),
    ('217134_3f1b0a57fd30', 'unflagged_control'),
    ('130785_fc6aa4e83741', 'unflagged_control'),
]
VARIANTS = {
    'baseline': dict(condition_on_previous_text=True, vad_filter=False),
    'A_previous_off': dict(condition_on_previous_text=False, vad_filter=False),
    'B_vad_on': dict(condition_on_previous_text=True, vad_filter=True),
    'C_previous_off_vad_on': dict(condition_on_previous_text=False, vad_filter=True),
}


def settings_for(variant):
    return dict(stt.stt_settings(False), **VARIANTS[variant])


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def intersection_duration(left, right):
    i = j = 0
    total = 0.
    while i < len(left) and j < len(right):
        total += max(0., min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def quality_metrics(data, speech):
    seg = data['segments']
    counts = collections.Counter(s['text'].strip() for s in seg if s['text'].strip())
    count = max(counts.values(), default=0)
    words = [w for s in seg for w in s['words']]
    word_spans = union_intervals((w['start'], w['end']) for w in words)
    speech_duration = sum(b-a for a,b in speech)
    covered = intersection_duration(speech, word_spans)
    repeated_fraction = count / max(1, len(seg))
    flags = []
    if count >= 3 and repeated_fraction >= .5:
        flags.append('dominant_repeated_segment')
    if len(data['text']) / data['duration_sec'] < 2:
        flags.append('low_text_density')
    if any(s in data['text'] for s in ('자막 제공','자막제공','한글자막','시청해 주셔서','시청해주셔서')):
        flags.append('subtitle_or_outro_phrase')
    return dict(characters=len(data['text']), segments=len(seg), words=len(words),
                unique_segment_texts=len(counts), top_repeat_count=count,
                top_repeat_fraction=repeated_fraction,
                characters_per_sec=len(data['text']) / data['duration_sec'],
                vad_speech_duration_sec=speech_duration,
                vad_speech_outside_word_timestamps_sec=max(0.,speech_duration-covered),
                vad_speech_covered_by_words_fraction=covered / speech_duration if speech_duration else None,
                timestamp_warning_count=len(data.get('timestamp_warnings', [])),
                review_flags=flags)


def sample_text(data, start, end):
    # Word midpoint selection avoids copying an entire long segment into a sample.
    return ''.join(w['word'] for s in data['segments'] for w in s['words']
                   if start <= (w['start']+w['end'])/2 < end)


def sample_windows(duration):
    width = min(30., duration)
    return [('start', 0., width), ('middle', (duration-width)/2, (duration+width)/2),
            ('end', duration-width, duration)]


def canonical_text(text):
    # Compare NFC characters with whitespace removed; punctuation is retained.
    return ''.join(c for c in unicodedata.normalize('NFC', text) if not c.isspace())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(previous[j]+1, current[j-1]+1, previous[j-1]+(a!=b)))
        previous = current
    return previous[-1]


def score_references(path, outputs):
    scored = []
    seen = set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('reviewed', '').strip() != '1':
                continue
            key = row['id'], row['position']
            if key in seen or row['id'] not in outputs:
                raise ValueError('Duplicate or unknown reviewed sample: '+str(key))
            seen.add(key)
            first = next(iter(outputs[row['id']].values()))
            expected = {name:(a,b) for name,a,b in sample_windows(first['duration_sec'])}
            bounds = float(row['start_sec']), float(row['end_sec'])
            if row['position'] not in expected or any(abs(a-b)>.0001 for a,b in zip(bounds,expected[row['position']])):
                raise ValueError('Reference window does not match comparison samples: '+str(key))
            reference = canonical_text(row['reference_text'])
            for variant, data in outputs[row['id']].items():
                hypothesis = canonical_text(sample_text(data,*bounds))
                distance = edit_distance(reference,hypothesis)
                scored.append(dict(id=row['id'], position=row['position'], variant=variant,
                                   reference_characters=len(reference), edits=distance,
                                   cer=distance/len(reference) if reference else None))
    totals = []
    for variant in VARIANTS:
        selected = [x for x in scored if x['variant']==variant]
        denominator = sum(x['reference_characters'] for x in selected)
        numerator = sum(x['edits'] for x in selected)
        totals.append(dict(variant=variant, reviewed_samples=len(selected), edits=numerator,
                           reference_characters=denominator, cer=numerator/denominator if denominator else None))
    return dict(normalization='Unicode NFC; whitespace removed; punctuation retained; word-midpoint sample selection',
                samples=scored, totals=totals)


def protected_snapshot(root):
    paths = [root/'metadata.jsonl', root/'transcription_manifest.jsonl', root/'transcription_summary.json']
    paths += sorted((root/'transcripts').glob('*'))
    return {str(p.relative_to(root)):stt.sha256(p) for p in paths if p.is_file()}


def write_tables(out, rows, outputs):
    fields = ['id','group','variant','status','duration_sec','processing_time_sec','rtf',
              'characters','segments','words','unique_segment_texts','top_repeat_count',
              'top_repeat_fraction','vad_speech_covered_by_words_fraction','timestamp_warning_count','review_flags']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(x for x in rows if x['status'] in ('ok','reused'))
    stt.atomic_text(out/'comparison_metrics.csv','\ufeff'+buffer.getvalue())
    path = out/'review_samples.csv'
    # Never overwrite a user's reference annotations on resume.
    if not path.exists():
        buffer = io.StringIO(newline='')
        columns = ['id','position','start_sec','end_sec','audio_path','source_audio_path','reviewed','reference_text',*VARIANTS]
        writer = csv.DictWriter(buffer,fieldnames=columns);writer.writeheader()
        for id, variants in outputs.items():
            first = next(iter(variants.values()))
            for position,a,b in sample_windows(first['duration_sec']):
                row = dict(id=id,position=position,start_sec=round(a,6),end_sec=round(b,6),
                           audio_path=f'review_audio/{id}_{position}.wav',
                           source_audio_path=first['audio_path'],reviewed='0',reference_text='')
                row.update({variant:sample_text(data,a,b) for variant,data in variants.items()})
                writer.writerow(row)
        stt.atomic_text(path,'\ufeff'+buffer.getvalue())


def write_review_audio(root, out, outputs):
    """Create exact PCM slices for human review; leave original WAVs untouched."""
    directory=out/'review_audio';directory.mkdir(exist_ok=True)
    for id,variants in outputs.items():
        if not variants:
            continue
        first=next(iter(variants.values()))
        with wave.open(str(root/first['audio_path']),'rb') as source:
            for position,a,b in sample_windows(first['duration_sec']):
                source.setpos(round(a*source.getframerate()))
                frames=source.readframes(round((b-a)*source.getframerate()))
                with tempfile.NamedTemporaryFile(dir=directory,suffix='.wav',delete=False) as temp:
                    name=Path(temp.name)
                    try:
                        with wave.open(temp,'wb') as target:
                            target.setparams(source.getparams())
                            target.writeframes(frames)
                        temp.flush();os.fsync(temp.fileno())
                        os.replace(name,directory/f'{id}_{position}.wav')
                    finally:
                        name.unlink(missing_ok=True)


def compare(root, out, force=False, references=None):
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    started = time.perf_counter()
    before = protected_snapshot(root)
    monitor = None
    results = []
    outputs = {}
    model = None
    summary_path = out/'comparison_summary.json'
    prior = stt.read_json(summary_path) if summary_path.exists() else {}
    prior_rows = {(x['id'],x['variant']):x for x in prior.get('results',[])}
    summary = dict(started_at=datetime.now(timezone.utc).isoformat(), status='running',
                   input_root=str(root), output_root=str(out), sample_selection=SAMPLES,
                   variants=VARIANTS, results=results, model_load_time_sec=0.,
                   model_load_count=0, cer=None, human_accuracy_verified=False,
                   diagnostic_note='Repetition and VAD/word timestamp overlap are heuristics, not accuracy scores. Selected sample is biased toward known failures.')
    def save():
        stt.atomic_text(summary_path,stt.json_text(summary))
    try:
        env = stt.environment()
        summary['environment'] = env
        monitor = stt.GpuMonitor();monitor.thread.start()
        manifest = {x['id']:x for x in [json.loads(l) for l in (root/'transcription_manifest.jsonl').read_text().splitlines()]}
        for id,group in SAMPLES:
            baseline = stt.read_json(root/'transcripts'/(id+'.json'))
            video = root/baseline['video_path'];wav = root/baseline['audio_path']
            source_hash = stt.sha256(video);audio_hash = stt.sha256(wav)
            if source_hash != baseline['source_sha256'] or audio_hash != baseline['audio_sha256']:
                raise ValueError('Changed original video/audio: '+id)
            if (stt.sha256(root/'transcripts'/(id+'.json')) != manifest[id]['json_sha256'] or
                    stt.sha256(root/'transcripts'/(id+'.txt')) != manifest[id]['txt_sha256']):
                raise ValueError('Changed original transcript: '+id)
            duration = stt.validate_wav(wav,stt.audio_duration(stt.probe(video),video))
            audio = dict(path=wav,source_sha256=source_hash,audio_sha256=audio_hash,duration_sec=duration)
            speech = get_speech_timestamps(decode_audio(str(wav)),VadOptions(threshold=.5,min_silence_duration_ms=250,speech_pad_ms=0))
            speech = [(x['start']/16000,x['end']/16000) for x in speech]
            outputs[id] = {}
            for variant in VARIANTS:
                row = dict(id=id,group=group,variant=variant)
                directory = out/variant;directory.mkdir(exist_ok=True)
                jp = directory/(id+'.json');tp = jp.with_suffix('.txt')
                sig = stt.signature(settings_for(variant),env)
                data = None if force else stt.reusable(jp,tp,audio,sig,video,prior_rows.get((id,variant),{}))
                try:
                    if data is None:
                        if model is None:
                            model,summary['model_load_time_sec'] = stt.load_model(sig['settings'])
                            summary['model_load_count'] += 1
                        LOG.info('%s %s: CUDA FP16 inference',id,variant)
                        data = stt.transcribe(model,audio,video,sig)
                        try:
                            data['timestamp_warnings'] = stt.validate_transcript(data,data['text'],duration)
                        except (ValueError,KeyError,TypeError) as exc:
                            stt.atomic_text(jp.with_suffix('.invalid.json'),stt.json_text(dict(data,validation_error=str(exc))))
                            raise
                        stt.atomic_text(tp,data['text']);stt.atomic_text(jp,stt.json_text(data))
                        row['status']='ok'
                    else:
                        row['status']='reused'
                    row.update(duration_sec=duration,processing_time_sec=data['processing_time_sec'],rtf=data['rtf'],
                               json_sha256=stt.sha256(jp),txt_sha256=stt.sha256(tp),**quality_metrics(data,speech))
                    outputs[id][variant]=data
                    LOG.info('%s %s: %s, chars=%d, repeat=%.3f, RTF=%.4f',id,variant,row['status'],row['characters'],row['top_repeat_fraction'],row['rtf'])
                except Exception as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    LOG.error('%s %s failed: %s',id,variant,exc)
                    if isinstance(exc,stt.EnvironmentFailure) or stt.gpu_failure(exc) or isinstance(exc,OSError):
                        results.append(row);save();raise
                results.append(row);save()
        write_tables(out,results,outputs)
        write_review_audio(root,out,outputs)
        if references:
            summary['cer'] = score_references(references,outputs)
            stt.atomic_text(out/'reference_scores.json',stt.json_text(summary['cer']))
        aggregate=[]
        for variant in VARIANTS:
            selected=[x for x in results if x['variant']==variant and x['status'] in ('ok','reused')]
            duration=sum(x['duration_sec'] for x in selected);elapsed=sum(x['processing_time_sec'] for x in selected)
            aggregate.append(dict(variant=variant,files=len(selected),flagged_files=sum(bool(x['review_flags']) for x in selected),
              dominant_repeat_files=sum('dominant_repeated_segment' in x['review_flags'] for x in selected),
              total_characters=sum(x['characters'] for x in selected),inference_time_sec=elapsed,rtf=elapsed/duration if duration else None))
        summary['aggregate']=aggregate
        summary['status']='completed_with_errors' if any(x['status']=='failed' for x in results) else 'completed'
    except KeyboardInterrupt:
        summary['status']='interrupted'
    except Exception as exc:
        summary.update(status='aborted',fatal_error=f'{type(exc).__name__}: {exc}')
        LOG.error('Comparison aborted: %s',exc)
    finally:
        if monitor:
            monitor.close()
            summary.update(gpu_max_memory_used_mib=monitor.max_memory,gpu_max_utilization_percent=monitor.max_util,
                           gpu_measurement_scope='Device 0 total including other processes; 0.5 second samples')
        summary['protected_originals_unchanged']=before==protected_snapshot(root)
        if not summary['protected_originals_unchanged']:
            summary.update(status='aborted',fatal_error='Original dataset changed during comparison')
        summary['pipeline_time_sec']=time.perf_counter()-started
        summary['new_inference_time_sec']=sum(x.get('processing_time_sec',0) for x in results if x['status']=='ok')
        summary['finished_at']=datetime.now(timezone.utc).isoformat();save()
        runs=out/'runs';runs.mkdir(exist_ok=True)
        run_id=summary['started_at'].replace(':','').replace('+','_')
        stt.atomic_text(runs/(run_id+'.json'),stt.json_text(summary))
    return 0 if summary['status']=='completed' else 130 if summary['status']=='interrupted' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=stt.DEFAULT_INPUT)
    p.add_argument('--output',type=Path,help='Separate comparison directory; default INPUT/quality_comparison/v1')
    p.add_argument('--force',action='store_true',help='Repeat comparison inference only; original transcripts are never replaced')
    p.add_argument('--references',type=Path,help='Human-reviewed review_samples.csv with reviewed=1 for completed rows')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();root=args.input.expanduser().resolve()
    out=args.output.expanduser().resolve() if args.output else root/'quality_comparison/v1'
    # Guard against writing into any established source/result directory.
    if out==root or out in root.parents or any(out==root/d or (root/d) in out.parents for d in ('raw','audio','transcripts','runs')):
        p.error('--output must be separate from original source and result directories')
    if args.dry_run:
        print(stt.json_text(dict(input=str(root),output=str(out),samples=SAMPLES,variants=VARIANTS)),end='');return 0
    stt.prepare_cuda_libraries(reexec_script=__file__)
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'comparison.log',encoding='utf-8')])
    with (root/'.transcription.lock').open('a') as lock, (out/'.comparison.lock').open('a') as comparison_lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(comparison_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.error('Dataset or comparison is already in use');return 2
        return compare(root,out,args.force,args.references)


if __name__=='__main__':
    sys.exit(main())
