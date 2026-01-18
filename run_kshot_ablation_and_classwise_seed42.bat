@echo off
setlocal

rem ===== Config =====
set "CONFIG=configs\experiment_suites.yaml"
set "SEED=42"

echo ============================================================
echo ForgetGate: K-shot prompt ablation + class-wise gap (seed %SEED%)
echo ============================================================

rem Ensure conda env is active
call conda activate forgetgate
if errorlevel 1 (
  echo ERROR: Failed to activate conda environment "forgetgate".
  exit /b 1
)

rem ===== Prompt-capacity ablation (10-shot, oracle + KL) =====
echo [Prompt Ablation] Oracle 10-shot (prompt length 1/2/5)
if exist "results\logs\vpt_oracle_vit_cifar10_forget0_prompt1_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] Oracle prompt1
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget0_10shot --seed %SEED% --prompt-length 1 || goto :error
)
if exist "results\logs\vpt_oracle_vit_cifar10_forget0_prompt2_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] Oracle prompt2
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget0_10shot --seed %SEED% --prompt-length 2 || goto :error
)
if exist "results\logs\vpt_oracle_vit_cifar10_forget0_prompt5_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] Oracle prompt5
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget0_10shot --seed %SEED% --prompt-length 5 || goto :error
)

echo [Prompt Ablation] Unlearned KL 10-shot (prompt length 1/2/5)
if exist "results\logs\vpt_resurrect_kl_forget0_prompt1_seed_%SEED%.jsonl" (
  echo [SKIP] KL prompt1
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0_10shot --seed %SEED% --prompt-length 1 || goto :error
)
if exist "results\logs\vpt_resurrect_kl_forget0_prompt2_seed_%SEED%.jsonl" (
  echo [SKIP] KL prompt2
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0_10shot --seed %SEED% --prompt-length 2 || goto :error
)
if exist "results\logs\vpt_resurrect_kl_forget0_prompt5_seed_%SEED%.jsonl" (
  echo [SKIP] KL prompt5
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0_10shot --seed %SEED% --prompt-length 5 || goto :error
)

rem ===== Class-wise gap runs (10-shot) =====
echo [Class-wise] Train oracles (forget classes 1/2/5/9)
if exist "checkpoints\oracle\oracle_vit_cifar10_forget1_seed_%SEED%_final.pt" (
  echo [SKIP] oracle forget1
) else (
  python scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite oracle_vit_cifar10_forget1 --seed %SEED% || goto :error
)
if exist "checkpoints\oracle\oracle_vit_cifar10_forget2_seed_%SEED%_final.pt" (
  echo [SKIP] oracle forget2
) else (
  python scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite oracle_vit_cifar10_forget2 --seed %SEED% || goto :error
)
if exist "checkpoints\oracle\oracle_vit_cifar10_forget5_seed_%SEED%_final.pt" (
  echo [SKIP] oracle forget5
) else (
  python scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite oracle_vit_cifar10_forget5 --seed %SEED% || goto :error
)
if exist "checkpoints\oracle\oracle_vit_cifar10_forget9_seed_%SEED%_final.pt" (
  echo [SKIP] oracle forget9
) else (
  python scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite oracle_vit_cifar10_forget9 --seed %SEED% || goto :error
)

echo [Class-wise] Train unlearned KL (forget classes 1/2/5/9)
if exist "results\logs\unlearn_kl_vit_cifar10_forget1_seed_%SEED%_history.json" (
  echo [SKIP] unlearn KL forget1
) else (
  python scripts\2_train_unlearning_lora.py --config %CONFIG% --suite unlearn_kl_vit_cifar10_forget1 --seed %SEED% || goto :error
)
if exist "results\logs\unlearn_kl_vit_cifar10_forget2_seed_%SEED%_history.json" (
  echo [SKIP] unlearn KL forget2
) else (
  python scripts\2_train_unlearning_lora.py --config %CONFIG% --suite unlearn_kl_vit_cifar10_forget2 --seed %SEED% || goto :error
)
if exist "results\logs\unlearn_kl_vit_cifar10_forget5_seed_%SEED%_history.json" (
  echo [SKIP] unlearn KL forget5
) else (
  python scripts\2_train_unlearning_lora.py --config %CONFIG% --suite unlearn_kl_vit_cifar10_forget5 --seed %SEED% || goto :error
)
if exist "results\logs\unlearn_kl_vit_cifar10_forget9_seed_%SEED%_history.json" (
  echo [SKIP] unlearn KL forget9
) else (
  python scripts\2_train_unlearning_lora.py --config %CONFIG% --suite unlearn_kl_vit_cifar10_forget9 --seed %SEED% || goto :error
)

echo [Class-wise] VPT oracle (10-shot)
if exist "results\logs\vpt_oracle_vit_cifar10_forget1_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt oracle forget1
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget1_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_oracle_vit_cifar10_forget2_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt oracle forget2
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget2_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_oracle_vit_cifar10_forget5_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt oracle forget5
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget5_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_oracle_vit_cifar10_forget9_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt oracle forget9
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget9_10shot --seed %SEED% || goto :error
)

echo [Class-wise] VPT unlearned KL (10-shot)
if exist "results\logs\vpt_resurrect_kl_forget1_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt kl forget1
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget1_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_resurrect_kl_forget2_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt kl forget2
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget2_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_resurrect_kl_forget5_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt kl forget5
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget5_10shot --seed %SEED% || goto :error
)
if exist "results\logs\vpt_resurrect_kl_forget9_10shot_seed_%SEED%.jsonl" (
  echo [SKIP] vpt kl forget9
) else (
  python scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget9_10shot --seed %SEED% || goto :error
)

echo [Analysis] k-shot summary (seed %SEED%)
python scripts\analyze_kshot_experiments.py --seeds %SEED% || goto :error

echo ============================================================
echo Done.
echo ============================================================
exit /b 0

:error
echo ============================================================
echo ERROR: Pipeline failed with exit code %errorlevel%.
echo ============================================================
exit /b %errorlevel%
