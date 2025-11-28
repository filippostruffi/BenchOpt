## Tasks (What and Why)

- Image Classification
  - Predict a single class per image. Ubiquitous benchmark for optimization stability and speed.
  - Loss: Cross-entropy (label smoothing 0.1). Primary metric: Accuracy.
  - Batching: Standard tensors (B×C×H×W, B labels). BN favors batch size >1.
  - Regularization: Data augmentation (CIFAR‑10), normalization, optional Mixup/CutMix.
  - Scheduler: cosine (default). Early stopping patience ~10.
  - Pitfalls: Without normalization/augmentation, may saturate quickly on small samples.

- Semantic Segmentation
  - Pixel-wise classification. Tests optimization under dense predictions and class imbalance.
  - Loss: Cross-entropy (with inverse-frequency class weights) + 0.5·Soft Dice loss; ignore_index=255. Primary: Dice.
  - Batching: (B×C×H×W, B×H×W mask). We drop_last=True to keep BN stable.
  - Data: Inputs normalized to ImageNet stats; both ADE20K and Oxford-IIIT Pet use 256×256 images.
  - Scheduler: cosine (default). Early stopping patience ~5. Gradient clipping enabled by preset.
  - Pitfalls: Class imbalance; tiny batch sizes destabilize BN/statistics. The combined Dice+CE and class weighting mitigate collapse to majority class and help metrics evolve over epochs.

- Sentiment Analysis
  - Text classification (positive/negative). Classic NLP task; quick, revealing for optimizer behavior.
  - Loss: Cross-entropy (label smoothing 0.05 on LSTM; HF models return built-in loss). Primary: Accuracy.
  - Batching: Either dict of tokenized tensors (HF) or our ids/lengths (LSTM).
  - Scheduler: ReduceLROnPlateau (default). Early stopping patience ~3. Gradient clipping 1.0.
  - Pitfalls: Tokenization choice affects results; max_len truncation matters.

- Machine Translation
  - Seq2seq generation from source to target language. Sensitive to optimizer/learning-rate choices.
  - Loss: Cross-entropy teacher-forcing (tgt_in→tgt_out) with label smoothing 0.1. Primary: BLEU (computed from ids).
  - Batching: pad_seq to max length per batch; decoder consumes tgt_in; predicts tgt_out.
  - Scheduler: ReduceLROnPlateau (default) or warmup+linear/inv-sqrt (ext). Early stopping ~5. Clip 1.0.
  - Pitfalls: Vocab quality (OOV) impacts BLEU; LR schedule improves Transformers.

- Named Entity Recognition (NER)
  - Sequence labeling with structured decoding (CRF). Evaluates optimizers in structured prediction.
  - Loss: CRF negative log-likelihood. Primary: F1 over tags.
  - Batching: dict with `input_ids`, `tags`, `lengths` (+ optional `char_ids`).
  - Scheduler: ReduceLROnPlateau (default). Early stopping ~3–5. Clip 1.0.
  - Pitfalls: Mask/length consistency is critical; char features help rare/OOOV tokens.

- Text Generation (Language Modeling)
  - Next-token prediction. Perplexity directly reflects optimization quality on generative models.
  - Loss: Cross-entropy on next token. Primary: Perplexity.
  - Batching: Fixed block size packing from concatenated token streams.
  - Scheduler: cosine (common) or plateau; clipping 1.0 often helps; label smoothing not used.
  - Pitfalls: Simple tokenization; block size influences difficulty (shorter = easier).

- Text Summarization
  - Seq2seq abstraction/compression. Good for optimizer comparison in encoder–decoder models.
  - Loss: Cross-entropy with label smoothing 0.1 on decoder outputs (teacher-forcing). Primary: ROUGE.
  - Batching: `{"src", "tgt_in", "tgt_out"}` with padding; causal mask in decoder.
  - Scheduler: ReduceLROnPlateau (default). Early stopping ~5. Clip 1.0.
  - Pitfalls: Short targets (e.g., AESLC) can saturate quickly; consider label smoothing.

- Question Answering (extractive)
  - Span prediction from context. Combines encoding, attention/fusion, and span heads.
  - Loss: start CE + end CE. Primary: F1 (token overlap).
  - Batching: dict with `context_ids`, `question_ids`, `starts`, `ends`.
  - Scheduler: ReduceLROnPlateau (default). Early stopping ~3–5. Clip 1.0.
  - Pitfalls: Exact-match spans rely on consistent tokenization; short contexts (TweetQA) can be tricky.

These cover core supervised ML regimes (classification, sequence labeling, seq2seq generation, dense prediction) across vision and NLP, making them practical and representative for optimizer benchmarking. 


