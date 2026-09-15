The transformer does not meet the registered two-seed improvement criterion on the larger fixed 9×9 corpus.

Positive gains mean lower transformer KL. Endpoint is update 4,096; both primary metrics must improve in each seed, with at least 0.5% mean relative gain and no last-three-mean regression in either seed.

| Metric | Seed | CNN endpoint | Transformer endpoint | Relative gain | Last-three gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert_kl | 91312427 | 0.12994647 | 0.13349390 | -2.730% | -2.737% |
| expert_kl | 91312428 | 0.12851834 | 0.13482809 | -4.910% | -3.373% |
| family_kl | 91312427 | 0.08986163 | 0.09345388 | -3.998% | -3.762% |
| family_kl | 91312428 | 0.09000468 | 0.09425640 | -4.724% | -4.136% |

expert_kl: mean relative endpoint gain -3.820%.

family_kl: mean relative endpoint gain -4.361%.

Unmet criteria: expert_kl.both_seed_endpoints_improved, expert_kl.mean_relative_gain_at_least_half_percent, expert_kl.neither_seed_last_three_mean_regressed, family_kl.both_seed_endpoints_improved, family_kl.mean_relative_gain_at_least_half_percent, family_kl.neither_seed_last_three_mean_regressed.

Overfitting diagnostic: no audited sustained flag at the fixed endpoint.

| Seed | Arm | Parameters | Training positions | Reserved chip-hours | Trained decode median ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 91312427 | cnn | 232431872 | 49753641 | 39.0851 | 16.300 |
| 91312427 | transformer | 231181121 | 49753641 | 32.8264 | 9.797 |
| 91312428 | cnn | 232431872 | 49789538 | 39.2087 | 16.276 |
| 91312428 | transformer | 231181121 | 49789538 | 32.9055 | 10.494 |

Reserved chip-hours are observed attempt reservations, not a billing total. Matched deployed FLOPs do not imply equal training work. Both paired contrasts were reproduced from the pinned per-run audits, including exact draws and evaluation populations.

Every registered validation and fixed training-probe point follows. Best observed turns remain diagnostics and do not replace the registered endpoint.

Seed 91312427

| Update | CNN val position KL | T val position KL | CNN val family KL | T val family KL | CNN train-probe KL | T train-probe KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.01782775 | 2.16114497 | 1.49975514 | 1.62853050 | 1.86496818 | 2.00423861 |
| 256 | 0.46172929 | 0.49916625 | 0.33123517 | 0.35453010 | 0.39854991 | 0.41608250 |
| 512 | 0.32443810 | 0.36627436 | 0.22706008 | 0.25871062 | 0.25486100 | 0.28693616 |
| 768 | 0.26962781 | 0.30177295 | 0.18966699 | 0.21312571 | 0.20438147 | 0.22884429 |
| 1024 | 0.25844300 | 0.26660788 | 0.17949319 | 0.18985343 | 0.19272196 | 0.19792104 |
| 1280 | 0.21756816 | 0.23174894 | 0.15303636 | 0.16458845 | 0.15763915 | 0.16881990 |
| 1536 | 0.20752954 | 0.21663535 | 0.14226031 | 0.15411019 | 0.14349055 | 0.15403104 |
| 1792 | 0.19410026 | 0.20304847 | 0.13611627 | 0.14242125 | 0.13704503 | 0.14099252 |
| 2048 | 0.17938602 | 0.18816352 | 0.12443709 | 0.13186622 | 0.12192571 | 0.12695944 |
| 2304 | 0.17472637 | 0.17425907 | 0.12048674 | 0.12256169 | 0.11514091 | 0.11660016 |
| 2560 | 0.16278648 | 0.16704643 | 0.11301374 | 0.11620235 | 0.10621166 | 0.10858715 |
| 2816 | 0.15258014 | 0.15609777 | 0.10569930 | 0.10939431 | 0.09738481 | 0.10097373 |
| 3072 | 0.14665043 | 0.14883363 | 0.10270500 | 0.10548496 | 0.09307599 | 0.09502423 |
| 3328 | 0.14245403 | 0.14549100 | 0.09919333 | 0.10079885 | 0.08791125 | 0.08963561 |
| 3584 | 0.13652694 | 0.14078093 | 0.09416747 | 0.09785724 | 0.08173954 | 0.08558810 |
| 3840 | 0.13168478 | 0.13478041 | 0.09112000 | 0.09418917 | 0.07836044 | 0.08141434 |
| 4096 | 0.12994647 | 0.13349390 | 0.08986163 | 0.09345388 | 0.07577860 | 0.07980859 |

Seed 91312428

| Update | CNN val position KL | T val position KL | CNN val family KL | T val family KL | CNN train-probe KL | T train-probe KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.01523328 | 2.20688248 | 1.50008464 | 1.71661401 | 1.86339939 | 2.05770874 |
| 256 | 0.43802142 | 0.49400473 | 0.31427407 | 0.35832500 | 0.37510598 | 0.41658819 |
| 512 | 0.31920314 | 0.36985278 | 0.22830272 | 0.26623130 | 0.25650895 | 0.29272878 |
| 768 | 0.27262247 | 0.31124461 | 0.19036937 | 0.21988916 | 0.20702279 | 0.23847234 |
| 1024 | 0.24944186 | 0.26310146 | 0.17895555 | 0.18761754 | 0.18604493 | 0.19514310 |
| 1280 | 0.22270358 | 0.23863590 | 0.15564680 | 0.16607666 | 0.16151667 | 0.16969025 |
| 1536 | 0.20054889 | 0.21606421 | 0.14116478 | 0.15363193 | 0.14310551 | 0.15348518 |
| 1792 | 0.18819559 | 0.19590616 | 0.13419294 | 0.14107704 | 0.13245702 | 0.13899243 |
| 2048 | 0.17791164 | 0.18998671 | 0.12455916 | 0.13365889 | 0.12070286 | 0.12889993 |
| 2304 | 0.16973162 | 0.17577672 | 0.11736441 | 0.12470531 | 0.11218047 | 0.11905444 |
| 2560 | 0.16317832 | 0.17129970 | 0.11238360 | 0.11974764 | 0.10549521 | 0.11188102 |
| 2816 | 0.15850890 | 0.15867031 | 0.10784841 | 0.11085701 | 0.09937191 | 0.10169113 |
| 3072 | 0.14692676 | 0.15398192 | 0.10168409 | 0.10605311 | 0.09179223 | 0.09577048 |
| 3328 | 0.14039373 | 0.14669061 | 0.09683228 | 0.10159492 | 0.08590817 | 0.09029531 |
| 3584 | 0.13769937 | 0.14057660 | 0.09434628 | 0.09820414 | 0.08185387 | 0.08596325 |
| 3840 | 0.13153458 | 0.13576198 | 0.09153223 | 0.09483266 | 0.07820690 | 0.08200336 |
| 4096 | 0.12851834 | 0.13482809 | 0.09000468 | 0.09425640 | 0.07583117 | 0.08057082 |

Two paired fixed-data supervised seeds. Failure to meet this criterion does not establish CNN superiority. No test-set, playing-strength or RL-efficiency conclusion.
