@echo OFF

timeout 25200

FOR %%x IN (43, 110, 450, 600, 800) do (
    echo %%x
    python ./main.py Train --config ".\configs\train_on_CCC_no_context.json" --log_file ".\logs\CCC_no_context.log" --datasplit_random_state %%x
    python ./main.py Train --config ".\configs\train_on_CCC.json" --log_file ".\logs\CCC.log" --datasplit_random_state %%x
)