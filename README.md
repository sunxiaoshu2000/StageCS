# StageCS
A Two-Stage Framework Integrating Prompt Learning and Fine-tuning for Code Summarization

## Dependency
* python==3.8
* torch==2.1.0
* transformers==4.32.1
* openai==0.28.0 (option)


## Dataset
We use the Java dataset from the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) code-to-text docstring
generation task, which is built upon the CodeSearchNet corpus and excludes defective data samples. 

We further process them to obtain two additional fields: 'clean_code' and 'clean_doc'.

### Download data and preprocess

    unzip dataset.zip
    cd dataset
    wget https://zenodo.org/record/7857872/files/java.zip  
    unzip java.zip
    python preprocessjava.py

    rm *.pkl
    rm -r */[^clean]*
    cd ..


### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. clean_train.jsonl, clean_valid.jsonl, clean_test.jsonl

For each file, each line in the uncompressed file represents one function. Here is an explanation of the fields:

* The fields contained in the original CodeXGLUE dataset:

  * repo: the owner/repo

  * path: the full path to the original file

  * func_name: the function or method name

  * original_string: the raw string before tokenization or parsing

  * language: the programming language

  * code/function: the part of the original_string that is code

  * code_tokens/function_tokens: tokenized version of code

  * docstring: the top-level comment or docstring, if it exists in the original string

  * docstring_tokens: tokenized version of docstring

* The additional fields we added:

  * clean_code: clean version of code that removing possible comments

  * clean_doc: clean version of docstring that obtaining by concatenating docstring_tokens

### Data Statistic

| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Java                 | 164,923  | 5,183  | 10,955 |


## Two Stage Training of StageCS

1 Prompt Learning Satge
    cd StageCS
    CUDA_VISIBLE_DEVICES=0 python run.py --mode StageCS1 --template [0,100] --reload False --model_name_or_path ../LLMs/codegen-350m --train_filename ../dataset/java/clean_train.jsonl --dev_filename ../dataset/java/clean_valid.jsonl --test_filename ../dataset/java/clean_test.jsonl --output_dir ./saved_models/PL --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 

2 Task-oriented Fine-tuning Satge
    CUDA_VISIBLE_DEVICES=0 python run.py --mode StageCS2 --template [0,100] --reload True --model_name_or_path ../LLMs/codegen-350m --load_model_finepath ./models/codegenmodels/saved_models_PL/checkpoint-best-bleufine/checkpoint-best-bleu/pytorch_model.bin --train_filename ../dataset/java/clean_train.jsonl --dev_filename ../dataset/java/clean_valid.jsonl --test_filename ../dataset/java/clean_test.jsonl --output_dir ./saved_models/FT --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5


### Arguments
The explanation for some of the arguments is as follows.

* model_name_or_path: Path to pre-trained model
* mode: Operational mode. Choices=["StageCS1", "StageCS2"]
* template: The concatenation method of pseudo tokens and code snippet. Default is the Back-end Mode [0, 100]
* load_model_finepath:The model after training in the first stage.
* output_dir: The output directory where the model predictions and checkpoints will be written.


## Evaluation

### BLEU and SentenceBERT
    cd PromptCS
    python evaluate.py --predict_file_path ./saved_models/test_0.output --ground_truth_file_path ./saved_models/test_0.gold --SentenceBERT_model_path ../all-MiniLM-L6-v2

### METEOR and ROUGE-L
To obtain METEOR and ROUGE-L, we need to activate the environment that contains python 2.7

    conda activate py27
    unzip evaluation
    cd evaluation
    python evaluate.py --predict_file_path ../PromptCS/saved_models/test_0.output --ground_truth_file_path ../PromptCS/saved_models/test_0.gold

Tip: The path should only contain English characters.
