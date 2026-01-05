@echo off
setlocal

rem ===== User-configurable knobs =====
set "PYTHON=python"
set "CONFIG=configs\experiment_suites.yaml"
set "SEEDS=42 123"
set "KSHOTS=10 25 50 100"

set "BASE_SUITE=base_vit_cifar10"
set "ORACLE_SUITE=oracle_vit_cifar10_forget0"

set "UNLEARN_CE=unlearn_lora_vit_cifar10_forget0"
set "UNLEARN_KL=unlearn_kl_vit_cifar10_forget0"
set "UNLEARN_SALUN=unlearn_salun_vit_cifar10_forget0"
set "UNLEARN_SCRUB=unlearn_scrub_distill_vit_cifar10_forget0"

set "RUN_FULLDATA_VPT=1"
set "RUN_SALUN=1"
set "RUN_SCRUB=1"
rem ===== End knobs =====

echo ============================================================
echo ForgetGate resume pipeline
echo Seeds: %SEEDS%
echo K-shots: %KSHOTS%
echo ============================================================

rem 1) Base + Oracle + Unlearning
for %%S in (%SEEDS%) do (
  if exist "checkpoints\base\%BASE_SUITE%_seed_%%S_final.pt" (
    echo [SKIP] Base seed %%S
  ) else (
    echo [RUN] Base seed %%S
    %PYTHON% scripts\1_train_base.py --config %CONFIG% --suite %BASE_SUITE% --seed %%S || goto :error
  )

  if exist "checkpoints\oracle\%ORACLE_SUITE%_seed_%%S_final.pt" (
    echo [SKIP] Oracle seed %%S
  ) else (
    echo [RUN] Oracle seed %%S
    %PYTHON% scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite %ORACLE_SUITE% --seed %%S || goto :error
  )

  if exist "results\logs\%UNLEARN_CE%_seed_%%S_history.json" (
    echo [SKIP] Unlearn CE seed %%S
  ) else (
    echo [RUN] Unlearn CE seed %%S
    %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_CE% --seed %%S || goto :error
  )

  if exist "results\logs\%UNLEARN_KL%_seed_%%S_history.json" (
    echo [SKIP] Unlearn KL seed %%S
  ) else (
    echo [RUN] Unlearn KL seed %%S
    %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_KL% --seed %%S || goto :error
  )

  if "%RUN_SALUN%"=="1" (
    if exist "results\logs\%UNLEARN_SALUN%_seed_%%S_history.json" (
      echo [SKIP] Unlearn SalUn seed %%S
    ) else (
      echo [RUN] Unlearn SalUn seed %%S
      %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_SALUN% --seed %%S || goto :error
    )
  )

  if "%RUN_SCRUB%"=="1" (
    if exist "results\logs\%UNLEARN_SCRUB%_seed_%%S_history.json" (
      echo [SKIP] Unlearn SCRUB seed %%S
    ) else (
      echo [RUN] Unlearn SCRUB seed %%S
      %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_SCRUB% --seed %%S || goto :error
    )
  )
)

rem 2) VPT k-shot (oracle + KL)
for %%S in (%SEEDS%) do (
  for %%K in (%KSHOTS%) do (
    if exist "results\logs\vpt_oracle_vit_cifar10_forget0_%%Kshot_seed_%%S.jsonl" (
      echo [SKIP] VPT Oracle seed %%S k=%%K
    ) else (
      echo [RUN] VPT Oracle seed %%S k=%%K
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget0_%%Kshot --seed %%S || goto :error
    )

    if exist "results\logs\vpt_resurrect_kl_forget0_%%Kshot_seed_%%S.jsonl" (
      echo [SKIP] VPT KL seed %%S k=%%K
    ) else (
      echo [RUN] VPT KL seed %%S k=%%K
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0_%%Kshot --seed %%S || goto :error
    )
  )
)

rem 3) VPT full-data recovery (training logs)
if "%RUN_FULLDATA_VPT%"=="1" (
  for %%S in (%SEEDS%) do (
    if exist "results\logs\vpt_resurrect_vit_cifar10_forget0_seed_%%S.jsonl" (
      echo [SKIP] VPT CE seed %%S
    ) else (
      echo [RUN] VPT CE seed %%S
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_vit_cifar10_forget0 --seed %%S || goto :error
    )

    if exist "results\logs\vpt_resurrect_kl_forget0_seed_%%S.jsonl" (
      echo [SKIP] VPT KL seed %%S
    ) else (
      echo [RUN] VPT KL seed %%S
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0 --seed %%S || goto :error
    )

    if "%RUN_SALUN%"=="1" (
      if exist "results\logs\vpt_resurrect_salun_forget0_seed_%%S.jsonl" (
        echo [SKIP] VPT SalUn seed %%S
      ) else (
        echo [RUN] VPT SalUn seed %%S
        %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_salun_forget0 --seed %%S || goto :error
      )
    )

    if "%RUN_SCRUB%"=="1" (
      if exist "results\logs\vpt_resurrect_scrub_forget0_seed_%%S.jsonl" (
        echo [SKIP] VPT SCRUB seed %%S
      ) else (
        echo [RUN] VPT SCRUB seed %%S
        %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_scrub_forget0 --seed %%S || goto :error
      )
    )
  )
)

rem 4) Evaluation (clean accuracy for paper baselines)
for %%S in (%SEEDS%) do (
  if exist "results\logs\eval_paper_baselines_vit_cifar10_forget0_seed_%%S_evaluation.json" (
    echo [SKIP] Eval baselines seed %%S
  ) else (
    echo [RUN] Eval baselines seed %%S
    %PYTHON% scripts\4_adv_evaluate.py --config %CONFIG% --suite eval_paper_baselines_vit_cifar10_forget0 --seed %%S || goto :error
  )
)

rem 5) Analysis
if exist "results\analysis\kshot_analysis.png" (
  echo [SKIP] k-shot analysis
 ) else (
  echo [RUN] k-shot analysis
  %PYTHON% scripts\analyze_kshot_experiments.py --seeds %SEEDS% || goto :error
)

if exist "results\analysis\comparison_table_markdown.txt" (
  echo [SKIP] unlearning tables
 ) else (
  echo [RUN] unlearning tables
  %PYTHON% scripts\6_analyze_results.py --suite eval_paper_baselines_vit_cifar10_forget0 --seeds %SEEDS% || goto :error
)

echo ============================================================
echo Done.
echo ============================================================
exit /b 0

:error
echo ============================================================
echo ERROR: Pipeline failed with exit code %errorlevel%.
echo ============================================================
exit /b %errorlevel%
