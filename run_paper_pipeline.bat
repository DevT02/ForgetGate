@echo off
setlocal enabledelayedexpansion

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

set "VPT_CE=vpt_resurrect_vit_cifar10_forget0"
set "VPT_KL=vpt_resurrect_kl_forget0"
set "VPT_SALUN=vpt_resurrect_salun_forget0"
set "VPT_SCRUB=vpt_resurrect_scrub_forget0"

set "EVAL_BASELINES=eval_paper_baselines_vit_cifar10_forget0"

set "RUN_FULLDATA_VPT=1"
set "RUN_SALUN=1"
set "RUN_SCRUB=1"
rem ===== End knobs =====

echo ============================================================
echo ForgetGate paper pipeline
echo Seeds: %SEEDS%
echo K-shots: %KSHOTS%
echo ============================================================

rem 1) Base + Oracle + Unlearning
for %%S in (%SEEDS%) do (
  echo [Base] seed %%S
  %PYTHON% scripts\1_train_base.py --config %CONFIG% --suite %BASE_SUITE% --seed %%S || goto :error

  echo [Oracle] seed %%S
  %PYTHON% scripts\1b_train_retrained_oracle.py --config %CONFIG% --suite %ORACLE_SUITE% --seed %%S || goto :error

  echo [Unlearn CE] seed %%S
  %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_CE% --seed %%S || goto :error

  echo [Unlearn KL] seed %%S
  %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_KL% --seed %%S || goto :error

  if "%RUN_SALUN%"=="1" (
    echo [Unlearn SalUn] seed %%S
    %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_SALUN% --seed %%S || goto :error
  )

  if "%RUN_SCRUB%"=="1" (
    echo [Unlearn SCRUB] seed %%S
    %PYTHON% scripts\2_train_unlearning_lora.py --config %CONFIG% --suite %UNLEARN_SCRUB% --seed %%S || goto :error
  )
)

rem 2) VPT k-shot (oracle + KL)
for %%S in (%SEEDS%) do (
  for %%K in (%KSHOTS%) do (
    echo [VPT Oracle] seed %%S k=%%K
    %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_oracle_vit_cifar10_forget0_%%Kshot --seed %%S || goto :error

    echo [VPT KL] seed %%S k=%%K
    %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite vpt_resurrect_kl_forget0_%%Kshot --seed %%S || goto :error
  )
)

rem 3) VPT full-data recovery (training logs)
if "%RUN_FULLDATA_VPT%"=="1" (
  for %%S in (%SEEDS%) do (
    echo [VPT CE] seed %%S
    %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite %VPT_CE% --seed %%S || goto :error

    echo [VPT KL] seed %%S
    %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite %VPT_KL% --seed %%S || goto :error

    if "%RUN_SALUN%"=="1" (
      echo [VPT SalUn] seed %%S
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite %VPT_SALUN% --seed %%S || goto :error
    )

    if "%RUN_SCRUB%"=="1" (
      echo [VPT SCRUB] seed %%S
      %PYTHON% scripts\3_train_vpt_resurrector.py --config %CONFIG% --suite %VPT_SCRUB% --seed %%S || goto :error
    )
  )
)

rem 4) Evaluation (clean accuracy for paper baselines)
for %%S in (%SEEDS%) do (
  echo [Eval baselines] seed %%S
  %PYTHON% scripts\4_adv_evaluate.py --config %CONFIG% --suite %EVAL_BASELINES% --seed %%S || goto :error
)

rem 5) Analysis
echo [Analysis] k-shot tables/plots
%PYTHON% scripts\analyze_kshot_experiments.py --seeds %SEEDS% || goto :error

echo [Analysis] unlearning tables
%PYTHON% scripts\6_analyze_results.py --suite %EVAL_BASELINES% --seeds %SEEDS% || goto :error

echo ============================================================
echo Done.
echo ============================================================
exit /b 0

:error
echo ============================================================
echo ERROR: Pipeline failed with exit code %errorlevel%.
echo ============================================================
exit /b %errorlevel%
