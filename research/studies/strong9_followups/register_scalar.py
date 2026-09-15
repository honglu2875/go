"""Freeze a reviewed scalar intervention after the preceding run has closed."""
import argparse
from pathlib import Path
import time

import launch_scalar
import scalar_source
from lr_compare import audited, read, sha

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--review-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.review) != a.review_sha256:
        raise ValueError('Completed review changed')
    review = read(a.review)
    if review['status'] != 'completed':
        raise ValueError('A completed scientific review is required')
    prior = launch_scalar.pinned(review['preceding_outcome'])
    if prior['status'] != 'passed' or review['created'] < prior['created']:
        raise ValueError('Review must follow the completed preceding outcome')
    qualification_path = STUDY/'scalar-cpu-001.json'
    qualification = read(qualification_path)
    if qualification['status'] != 'passed':
        raise ValueError('Scalar source and decision qualification failed')
    for name, digest in qualification['source_sha256'].items():
        if sha(STUDY/name) != digest:
            raise ValueError('Qualified scalar operator changed: '+name)
    base = read(STUDY/'lr15-registration-001.json')
    stages = {}
    for stage in ('seed1', 'seed2'):
        item = review['parent_audits'][stage]
        parent = audited(ROOT, ROOT/item['path'], item['sha256'])
        snapshot = scalar_source.clone(parent['snapshot'], review['mechanism'],
                                       review['candidate_value'], ROOT/'.gozero/snapshots')
        isolation = scalar_source.identical_source(parent['snapshot'], snapshot, review['mechanism'])
        stages[stage] = dict(parent_snapshot=parent['snapshot'].name, snapshot=snapshot.name,
                             parent_audit=item, **isolation)
    plan = dict(kind='strong9_scalar_intervention', created=time.time(), trial=review['trial'],
        mechanism=review['mechanism'], candidate_value=review['candidate_value'],
        parent_label=review['parent_label'], candidate_label=review['candidate_label'],
        preceding_conclusion=base['preceding_conclusion'], preceding_outcome=review['preceding_outcome'],
        review=dict(path=str(a.review.resolve().relative_to(ROOT)), sha256=a.review_sha256),
        qualification=base['qualification'], cnn_controls=base['cnn_controls'], stages=stages,
        screen_min_relative_gain=.005, source_helper_sha256=sha(STUDY/'scalar_source.py'),
        comparison_operator_sha256=sha(STUDY/'scalar_compare.py'),
        paired_helper_sha256=sha(STUDY/'lr_compare.py'),
        launch_operator_sha256=sha(STUDY/'launch_scalar.py'),
        finalize_operator_sha256=sha(STUDY/'finalize_scalar.py'),
        registration_operator_sha256=sha(Path(__file__)),
        scalar_qualification=dict(path=str(qualification_path.relative_to(ROOT)), sha256=sha(qualification_path)),
        initialization_preparation=dict(path='research/studies/strong9_followups/encoder-scale-preparation-001.json',
                                       sha256=sha(STUDY/'encoder-scale-preparation-001.json')),
        invariants=['Only the declared scalar changes; all frozen source files remain identical within each seed.',
            'Same complete4096-update horizon,64-update warmup, complete games/D4 draws, teacher targets and fixed evaluation populations.',
            'Same width, parameter schema, encoder-plus-decoder arithmetic and closed test targets.',
            'Retain complete learning state with verified peer copy and explicit RAM/growth reserves.'],
        decision='Run seed1 first. Replicate only if both endpoint KL metrics gain at least0.5%, neither last-three mean regresses and no sustained overfit appears. Both paired seeds must pass before accepting the intervention. Review before any further experiment.',
        scope='A sequential fixed-data supervised scalar experiment. This is not a claim of global optimality or playing strength.')
    launch_scalar.inspect(plan, 'seed1')
    with a.output.open('xb') as stream:
        stream.write(scalar_source.canonical_json(plan))
    a.output.chmod(0o444)
    print(scalar_source.canonical_json(dict(status='registered', trial=plan['trial'],
        mechanism=plan['mechanism'], candidate_value=plan['candidate_value'],
        registration_sha256=sha(a.output), snapshots={k:v['snapshot'] for k,v in stages.items()},
        training_launched=False)).decode(), end='')


if __name__ == '__main__':
    main()
