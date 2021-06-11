# FactualAblation

This is the official Repository for "Probing Factually Grounded Content Transfer with Factual Ablation"

Our paper proposes factual ablation, a method for measuring factual consistency, or how faithful grounded generation models are to the facts presented in grounding

Here, we include instructions for generating our Factual Ablation datasets, as well as an example of how to measure factual ablation

## Datasets

In the `data_gen` directory, we include scripts which can be used to produce our Factual Ablation datasets. 

For ease of use, we run these scripts ourselves and include the resulting datasets in the `data` directory. These can be loaded by calling ` FA_data_utils.get_FA_synthetic_dataset()` and `FA_data_utils.get_FA_wiki_dataset()`. 

These datasets are meant to test the factual consistency/factual ablation of models trained for the content transfer task. In our work, we use the raw dataset released [here](https://github.com/shrimai/Towards-Content-Transfer-through-Grounded-Text-Generation) for training content transfer models. 

## Example of Factual Ablation

Measuring factual ablation for a content transfer model will depend on how the model is trained. For instance, if special tokens are trained to separate grounding, context, and target for the content transfer task, these tokens should also be used for measuring factual ablation. Similarly, many documents in both the content transfer and factual ablation datasets exceed the typical history length for pretrained language models, and so require some form of truncation. 

To guide measurement of factual ablation, we include a simple example of measuring accuracy and margin-accuracy for a zero-shot GPT-2 model (GPT-2-zs from table 2 of our paper). Run this example:

```
python score_zeroshot_gpt2.py
```

Simply, the script measures the `-log P(target| grounding,context)` for both the correct and incorrect grounding, and tests whether their difference exceeds the given margin.

We plan to include trained models and methods from the original work in the future. We can share these upon request.
