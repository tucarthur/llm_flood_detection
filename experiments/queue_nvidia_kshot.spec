# Lane: nvidia NIM, K-shot sweep, then the remaining block-2 sensitivity cells.
# No published input-token/min cap on this path, so rpm stays at the request ceiling.
# provider|model|rpm|out_stem|input_jsonl|extra flags
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v2_3class_nemotron_nano_vl_8b||--taxonomy 3class --prompt-variant v2 --workers 12
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v3_3class_nemotron_nano_vl_8b||--taxonomy 3class --prompt-variant v3 --workers 12
# K-shot cells REMOVED for this model: nemotron-nano-vl-8b cannot attend to the query
# frame when reference images are attached. With K=1 it described the flood exemplar and
# called 11/12 frames flood, including dry ones, both before and after the image-order fix
# that demonstrably works on gemma-4-31b. It is an in-context capability floor, not a
# prompt bug -- and it retro-explains the earlier "image-RAG is catastrophic for nemotron"
# result, which was very likely this same defect rather than a model-quality finding.
