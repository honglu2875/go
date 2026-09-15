The published 233M-family model `kata1-zhizi-b40c768nbt-s11472M-d5982M` records **Muon** as its optimizer. This was verified directly from the complete metadata member of its [official checkpoint archive](https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-zhizi-b40c768nbt-s11472M-d5982M.zip), retrieved through the [official network listing](https://katagotraining.org/networks/).

| Saved checkpoint field | Value |
|---|---|
| Optimizer name | Muon |
| Training sample counter | 11,472,785,216 |
| Data-row counter | 5,982,868,269 |
| Architecture | 40 nested blocks; 768 trunk / 384 inner channels; Mish |

KataGo's author documents that this model used **SGD early and Muon for most of its training**, largely on b28-generated data. The newer released transformers used Muon from the start. [Published training history](https://lightvector.github.io/katagostudies/202607-symmetry/)

The exact b40 LR schedule remains unverified. This export contains only `model`, `train_state`, `config`, and `swa_model`; it omits optimizer state, learning-rate values, scheduler settings, batch size and the launch command. The public author history identifies the optimizer sequence but does not give the numerical schedule. Current trainer defaults and another model's settings cannot fill that gap.

For the **separate 2019, 19-day main run**, the original paper explicitly reports SGD with momentum 0.9, batch size 256, and the following **per-sample** learning rates. These values are not AdamW rates for the 233M model. [Paper, Section 2](https://arxiv.org/pdf/1902.10565)

| Stage | Per-sample LR |
|---|---:|
| First 5 million training samples | 2e-5 |
| Main training | 6e-5 |
| From approximately day 17.5 | 6e-6 |

Our fixed-data CNN experiment uses our shared AdamW policy-training recipe. Its initial 1e-4 LR was our control, not an official recommendation. The default-SGD branch previously cited does not establish the optimizer used for the published b40 model.

The extracted metadata, retrieval receipts, source version and hashes are recorded in [the provenance record](official_training_provenance.json). Only the archive prefix and complete metadata were retrieved; full weight data and the full archive CRC were not verified. No pretrained weights entered the learnability study. The registered encoder-size grid remains unchanged.
