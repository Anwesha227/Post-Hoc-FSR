
---

**Creating Environment** 

1. **Clone the repository**

   ```
   git clone https://github.com/Anwesha227/Post-Hoc-FSR.git
   cd Post-Hoc-FSR
   ```
2. **Create the Conda Environment**

   ```
   conda env create -f environment.yml
   ```
3. **Activate the Environment**

   ```
   conda activate qwen-infer
   ```
4. **Optional - verify installation**

   ```
   python -c "import openai, pandas, matplotlib, networkx; print('Environment setup complete.')"

   ```
5. Please create a Nebius API Key and add it a **".env"** file like this:

   ```
   NEBIUS_API_KEY = "your API key"
   ```

---


**Inference Files**

[run_inference_qwen_explanation_top5.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_explanation_top5.py "run_inference_qwen_explanation_top5.py") - This is for running Top-5 Explanation Prompt with Common Name only.

[run_inference_qwen_explanation_top5_sci.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_explanation_top5_sci.py "run_inference_qwen_explanation_top5_sci.py") - This is for running Top-5 Explanation Prompt with Common Name + Scientific Name.

[run_inference_qwen_explanation_top5_complete.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_explanation_top5_complete.py "run_inference_qwen_explanation_top5_complete.py") - This is for running Top-5 Explanation Prompt with complete taxonomic Information.

[run_inference_qwen_explanation_top5_hierarchy.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_explanation_top5_hierarchy.py "run_inference_qwen_explanation_top5_hierarchy.py") - This is the simplest hierarchical prompt (grouped by Family).

[run_inference_qwen_image_only.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_image_only.py "run_inference_qwen_image_only.py") - This is for running Top-5 with only image examples (no common name or other taxonomic information)

[run_inference_qwen_multimodal_top5_kshot.py](https://github.com/Anwesha227/Post-Hoc-FSR/blob/main/inference_scripts/run_inference_qwen_multimodal_top5_1shot.py "run_inference_qwen_multimodal_top5_1shot.py") - These files are for running Top-5 with their Common Name and image examples, k is the number of images stitched into a single image.

The files starting with "rerun" are for re-running the files which timed out the first time.

---

**Running Inference**

You can edit the `OUTPUT_CSV` and `ERROR_FILE` paths as you need. Please make sure you have the Semi-Aves test set downloaded at `dataset/semi-aves/test` . For running any of the multi-modal inference files also make sure you have `dataset/semi-aves/trainval_images` and pre-generate the stiched images using one of the files from `dataset/semi-aves/pre_generate_reference_tiles_kshot.py`

```
conda activate qwen-infer
cd inference_scripts/
python run_inference_qwen_explanation_top5_sci.py
```

This will save a file at `OUTPUT_CSV`. For instance, `qwen_output/qwen_top5_explanations_sci.csv`. The next step is to process these results and store them in another csv file.

For this we will use the respective file from the folder `process_results/` . In this case,it would be `process_results/evaluate_qwen_responses_explanation_top5_updated.py`. Again, modify the `PRED_PATH`, `QWEN_PATH,` `OUTPUT_CSV`, `INVALID_CSV` as needed. 

```
cd process_results
python `evaluate_qwen_responses_explanation_top5_updated.py`
```

This will save a file to the folder post_processed_results/. For this exaple, it gets saved to `/post_processed_csv/qwen_evaluation_result_explanation_top5_sci.csv`. 

Then we will check the accuracy stats at different results. While in the main project directory, run:

```
python print_threshold_acc.py
```


---

While running inference and processing scripts, you will encounter TOP5_CSV and PRED_PATH. These are the base predictions from the SWIFT model.

prediction_analysis_top5_with_families.csv - This has top 5 predictions with their families.

prediction_analysis_top5.csv - This has top 5 predictions from the SWIFT model.

prediction_analysis_fsft_top5.csv - This has top 5 predictions from the few-shot fine tuned model.

---
