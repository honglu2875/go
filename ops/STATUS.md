The larger fixed 9×9 comparison is complete and audited on two paired seeds.
Transformer endpoint policy KL was 3.82% higher per position and 4.36% higher
per opening family, averaging paired relative differences. Its measured
learning time was about 20% lower. No sustained overfit was detected.

The first follow-up raised the transformer LR schedule by 1.5×. It finished
0.94% worse on position KL and 1.33% worse on family KL, failing the registered
first-seed screen. No second seed was launched. The original LR remains the
parent. See research/studies/strong9_followups/lr15-results-001.md and its CSV.

The next registered intervention changes only encoder residual-scale
initialization from 1e-6 to 0.01. It preserves all architecture, parameter,
complete decoding-FLOP, data and horizon settings. Seed 1 is running; its
continuation will replicate only if the frozen endpoint/tail/overfit gate passes.
The scalar harness passed numerical preparation and corruption checks.

The combined source Muon/AuxAdam, Lookahead and norm/schedule implementation
passed small real-19×19 joint neural training, exact fresh-process recovery,
and native search inference. The full-size source-optimizer CNN now has a frozen execution configuration
and passed abstract distributed update shapes for complete histories. Actual
TPU execution/recovery, scientific settings and trained strength remain open. The mixed-strength
19×19 producer continues; inventory003 records 2,064 completed games and
856,540 positions, with a still-small teacher-self-play validation stratum.

The closed 9×9 dataset is available at https://huggingface.co/datasets/quintic/go9x9,
revision 4f579b0a21c456f3bb568174d384056351124cd5, tag final-9x9-v1.
Publication scope and historical artifact limitations are in PUBLICATION.md.

The paired architecture-gap diagnostic places 74–91% of each seed's overall
validation KL gap in moves 1–64. This is post-hoc description and does not
change selection rules. See strong9_followups/ARCHITECTURE_GAP.md and the CSV.
