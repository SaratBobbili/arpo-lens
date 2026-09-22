#!/bin/bash
# No-GPU check for S2: wrapper argv, override passthrough, snapshot, and resolved Hydra config.
set -e

TRAINING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPTS_DIR="${TRAINING_DIR}/scripts"
WORK="${TRAINING_DIR}/.s2_dryrun_work"
DRY_TRAIN="${SCRIPTS_DIR}/.train_dry.sh"
mkdir -p "${WORK}/out"
source "${SCRIPTS_DIR}/secrets.sh"

bash -n "${SCRIPTS_DIR}/train.sh"
bash -n "${SCRIPTS_DIR}/train_qwen7B.sh"

# 1) The exact argv train_qwen7B.sh hands to train.sh.
bash() { printf '%s\n' "$@" > "${WORK}/wrapper_argv.txt"; }
source "${SCRIPTS_DIR}/train_qwen7B.sh"
unset -f bash
mapfile -t ARGV < "${WORK}/wrapper_argv.txt"
[[ "${ARGV[0]}" == "${SCRIPTS_DIR}/train.sh" ]]
[[ "${ARGV[1]}" == "training_config/config_r3.yaml" ]]

# 2) A train.sh copy that stops short of launching: no wandb login, scratch output root.
sed -e 's|^wandb login.*|true|' \
    -e "s|^source .*secrets.sh\"|& ; OUTPUT_ROOT=\"${WORK}/out\"|" \
    -e "s|^python3 -m training.main_echo.*|printf '%s\\\\n' \"\${ARGS[@]}\" > \"${WORK}/hydra_args.txt\"|" \
    "${SCRIPTS_DIR}/train.sh" > "${DRY_TRAIN}"
bash "${DRY_TRAIN}" "${ARGV[@]:1}"

# 3) Overrides won over the profile defaults, and untouched profile keys survived.
#    Read the run identity out of the wrapper's OWN argv and the selected profile rather
#    than hardcoding it, so this check does not rot every time train_qwen7B.sh moves a
#    default (it had drifted three defaults behind).
_ov() { printf '%s\n' "${ARGV[@]:1}" | sed -n "s|^$1=||p" | tail -1; }
PROJ="$(_ov project_name)"; EXP="$(_ov experiment_name)"; SUBPATH="$(_ov actor_model_subpath)"
NHL="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['hl_num_iters'])" \
        "${TRAINING_DIR}/${ARGV[1]}")"
[[ -n "${PROJ}" && -n "${EXP}" && -n "${SUBPATH}" && -n "${NHL}" ]]

grep -qx -- "trainer.project_name=${PROJ}" "${WORK}/hydra_args.txt"
grep -qx -- "trainer.experiment_name=${EXP}" "${WORK}/hydra_args.txt"
grep -qx -- "phases.high_level.num_iters=${NHL}" "${WORK}/hydra_args.txt"
grep -q  -- "${SUBPATH##*/}" "${WORK}/hydra_args.txt"
! grep -q -- "Llama-3.1-8B" "${WORK}/hydra_args.txt"

# 4) Derived paths follow the overridden run identity, and the snapshot records the overrides.
SNAPSHOT="${WORK}/out/checkpoints/${EXP}/training_config"
grep -qx -- "actor_model_subpath=${SUBPATH}" "${SNAPSHOT}/launch_overrides.txt"
test -f "${SNAPSHOT}/launch_config.yaml"

# 5) An override key outside VALID_LAUNCH_KEYS fails loud.
rc=0
bash "${DRY_TRAIN}" "${ARGV[1]}" bogus_key=1 2>"${WORK}/bogus.err" || rc=$?
[[ ${rc} -ne 0 ]]
grep -q "Unknown launch override key: bogus_key" "${WORK}/bogus.err"

# 6) Hydra composes the emitted overrides into the intended run.
VERL_ROOT="$(cd "${TRAINING_DIR}/../../ARPO/verl_arpo_entropy" && pwd)" \
    "${CONDA_PATH}/envs/${CONDA_ENV}/bin/python" \
    "${TRAINING_DIR}/.s2_compose_check.py" "${WORK}/hydra_args.txt"

echo "S2 DRYRUN OK"
