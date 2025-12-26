SET aspect=0
SET dataset=sst2

echo Start running bertattack.py...
python run_guide.py ^
    --aspect %aspect% ^
    --dataset %dataset% ^
    --train_path data/%dataset%/%dataset%-train.json ^
    --dev_path data/%dataset%/%dataset%-validation.json ^
    --test_path data/%dataset%/%dataset%-test.json ^
    --max_length 256 ^
    --save_path results/%dataset%/ ^
    --save_name guide_st ^
    --fix_embedding ^
    --lr 0.0001 ^
    --sparsity 0.15 ^
    --sparsity_lambda 10 ^
    --continuity_lambda 10 ^
    --guide_lambda 10 ^
    --guide_decay 1e-5 ^
    --match_lambda 1.5 ^
    --model sep ^
    --print_rationale

echo Execution finished.
pause