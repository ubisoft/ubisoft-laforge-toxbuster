© [2023] Ubisoft Entertainment. All Rights Reserved

# ToxBuster

We aim to use language models to identify and classify toxicity inside in-game chat.

## Project Standards

For better collaboration and understanding of the project code and what has been done, the following sections outline what standards / loose rules are followed.


### Project File Layout
Follow something similar to under src [packaging-projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

```
> src
   > module_1
      __init__.py
   > module_2
   __init__.py
> tests
main.py
poetry.lock
```

### Code Simple Guide
1. Keep it simple stupid.
2. Don't add features / over-engineer until you need it.


### Code Standards

| Name | Value | Links | Notes |
| :--- | :--- |  :-- | :-- |
| Language | Python 3.x | | |
| Package Manager | Poetry | [Docs](https://python-poetry.org/docs/) ; [Useful TLDR](https://hackersandslackers.com/python-poetry-package-manager/) | On Windows, you may have to restart comp after installing to work with VSCode. |
| Python Env | conda |  |
| Code Linter | Pep8 | [Enable for VSCode](https://code.visualstudio.com/docs/python/linting) |
| Docstring  | Pep8  | Follow [Numpy's Style](https://numpydoc.readthedocs.io/en/latest/format.html) |
| Unit Tests | Python 3.x | [Sample](https://stackoverflow.com/questions/61151/where-do-the-python-unit-tests-go) |

### Poetry

1. Test if poetry package manager is up to date:
   ```Powershell
   poetry run python .\main.py Train --config ".\train\train_on_CONDA_no_context.json" --max_epochs_to_train 1
   ```
   Note: Poetry installs torch with CPU support and no CUDA support.

   [Issue 4231](https://github.com/python-poetry/poetry/issues/4231)
   -> User may have to separately install PyTorch with CUDA.

2. Use `poetry add` to add missing packages to pyproject.toml & poetry.lock

3. Use `poetry export -f requirements.txt > requirements.txt` to update `requirements.txt`.

## Understanding our model
We want our model to be able to classify span of words as non-toxic / specific categories of toxicity.   For this use case, the model is currently a token classification.  

### Basic information:
* Current model is `bert-base-uncased`; 
* Tokenizer configs can be found [here](https://huggingface.co/distilbert-base-uncased/raw/main/tokenizer.json).
* HuggingFace [Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)


### Collate Function
* HuggingFace Tokenization Documentation:  https://huggingface.co/docs/tokenizers/pipeline
* Useful Stackoverflow: https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt


## Trainer Logic / Terminology
1. Epoch: One run of the training dataset.
2. Batch Size: Number of samples to train on limited by memory size of CPU / GPU.
   * `per_gpu_batch_size`: number of samples to run on each gpu if more than one. Batch size will be `num_gpu` * `per_gpu_batch_size`
3. Global Step: Number of batches before the model will calculate gradient & perform back propagation.
   * To [prevent vanishing & exploding gradients](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem), we use [`clip_grad_norm_`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) & accumulate batches.
   * [Gradient Accumulation](https://towardsdatascience.com/gradient-accumulation-overcoming-memory-constraints-in-deep-learning-36d411252d01) is performed at every global step
4. Validation Loop:
   * We run validation at every `X` epochs. If we follow the paper, it was run 10 times per epoch.
   * Push metrics to TensorBoard
   * In normal ML models, we run validation every epoch or even every `X` epochs.
5. Save model at the end of every `X` epoch:
   * changed from global step since this is dependent on two config variables and can be inconsistent.
   * can be changed back if we save all the configs.


## Other Useful Links / Info
1. Trainer Logic Code samples:
   * https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/examples/bert_stable_fine_tuning/run_finetuning.py
   * https://towardsdatascience.com/how-to-make-the-most-out-of-bert-finetuning-d7c9f2ca806c
   * https://www.pluralsight.com/guides/data-visualization-deep-learning-model-using-matplotlib

© [2023] Ubisoft Entertainment. All Rights Reserved