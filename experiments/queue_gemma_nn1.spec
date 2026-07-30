# Lane: gemini. Side experiment -- ONE nearest-neighbour exemplar, retrieved by DINOv2
# similarity to the query frame, attached with its label.
#
# This is a different question from the K-shot arm, not a smaller version of it:
#
#   zero-shot   0 exemplars                          -- what the criteria paragraph alone buys
#   K=1         3 exemplars, one per class, random   -- where the class boundaries are
#   nn1         1 exemplar, nearest neighbour        -- what does THIS frame look like
#
# K-shot shows the model the label space; nn1 shows it the single most visually similar
# labelled frame in the bank. So nn1 tests instance-based transfer rather than class
# calibration, and it does it at a third of the image payload.
#
# Leakage: the retriever masks the query's own season (where season != exclude_season), so a
# frame can never retrieve itself, and since events do not span seasons the neighbour is always
# event-disjoint from the query. That is the same LOSO discipline as every other arm. The bank
# is the same 1,592-frame curated pool used as the support pool elsewhere.
#
# Run on the FULL 1,592-frame evaluation set, matching the zero-shot cells one-to-one. The
# 375-row K-shot subsample is a strict subset of it, so restricting these results to those rows
# afterwards still gives an exact comparison against the K=1 cells -- full n buys the zero-shot
# comparison without costing the K-shot one.
#
# Rate: 2 images per call rather than 4, so lighter than the K=1 cells; rpm 6 / 2 workers.
# provider|model|rpm|out_stem|input_jsonl|extra flags
gemini|gemma-4-31b-it|6|nn1_gemma4_31b||--taxonomy 3class --image-rag --image-rag-mode similarity --n-exemplars 1 --workers 2
gemini|gemma-4-26b-a4b-it|6|nn1_gemma4_26b_a4b||--taxonomy 3class --image-rag --image-rag-mode similarity --n-exemplars 1 --workers 2
