@echo off
chcp 65001 >nul
set CUDA_VISIBLE_DEVICES=0
set PYTHON_PATH=D:\an\envs\py3.11\python.exe
set BASE_MODEL=D:\Desktop\eleven_project\models\bge-base-zh-v1.5
set FINETUNED_MODEL=output\bge-base-zh-v1.5-finetuned
set CORPUS_FILE=data\test_corpus.jsonl
set TEST_FILE=data\test_queries.jsonl
set BATCH_SIZE=256
echo Evaluating base model: %BASE_MODEL%
"%PYTHON_PATH%" scripts\evaluate.py --model_name_or_path "%BASE_MODEL%" --corpus_file "%CORPUS_FILE%" --test_file "%TEST_FILE%" --batch_size %BATCH_SIZE%
echo.
echo Evaluating finetuned model: %FINETUNED_MODEL%
if exist "%FINETUNED_MODEL%" (
    "%PYTHON_PATH%" scripts\evaluate.py --model_name_or_path "%FINETUNED_MODEL%" --corpus_file "%CORPUS_FILE%" --test_file "%TEST_FILE%" --batch_size %BATCH_SIZE%
) else (
    echo Finetuned model not found. Run run_finetune.bat first.
)
echo Done!
