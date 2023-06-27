@echo OFF
FOR %%x IN (43, 110, 450, 600, 800) do (
    echo %%x
    python ./main.py Train --config ".\configs\train_on_CONDA_no_context.json" --log_file ".\logs\CONDA_no_context.log" --datasplit_random_state %%x
    python ./main.py Train --config ".\configs\train_on_CONDA.json" --log_file ".\logs\CONDA.log" --datasplit_random_state %%x
)



