#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

CLI="${SCRIPT_DIR}/run_treax_expedition.py"

# Runtime settings (override via env vars)
# stages: search (default), search+committor, or explore (one MCTSS run)
[ -z "${stages}" ] && stages='search'
[ -z "${nworkers}" ] && nworkers='4'
[ -z "${tau}" ] && tau='1.'
[ -z "${steps}" ] && steps='10'
[ -z "${eq}" ] && eq='0'
[ -z "${nprocs}" ] && nprocs='1'
# method is left unset on purpose: the CLI then picks the default of whichever
# back end is in use, gfn2-xtb on xtb and hf-3c on orca
[ -z "${accuracy}" ] && accuracy='1.'
[ -z "${etemp}" ] && etemp='300.'
# t12 is left unset on purpose: 10 for the explore stage, 3 for the search
# charge and mult are left unset on purpose: the CLI then takes them from the
# reaction's meta.json, and falls back to 0 and 1 when there is none
[ -z "${threshold}" ] && threshold='1.'
# calculator is left unset on purpose: the search takes it from meta.json,
# the explore stage falls back to xtb
[ -z "${seed}" ] && seed=''
[ -z "${step_scale}" ] && step_scale='1.'

# Input: an .xyz, a folder with reactant.xyz and product.xyz, or a reaction
# name resolved under reaction/input. Default is the sn2 reaction shipped here.
[ -z "${file}" ] && file='sn2'

# xtb is threaded with OpenMP, ORCA runs separate MPI processes, and the
# search runs worker processes of its own. Giving both the same count at once
# would oversubscribe the machine.
if [ "${calculator:-xtb}" = "xtb" ] && [ "${stages}" = "explore" ]; then
    export OMP_NUM_THREADS="${nprocs}"
else
    export OMP_NUM_THREADS=1
fi

# Optional flags: the ORCA back end, moving every atom, the GFN2-xTB screen
EXTRA=""
[ -n "${method}" ] && EXTRA="${EXTRA} --method=${method}"
[ -n "${orcapath}" ] && EXTRA="${EXTRA} --orcapath=${orcapath}"
[ -n "${basis}" ] && EXTRA="${EXTRA} --basis=${basis}"
[ -n "${state}" ] && EXTRA="${EXTRA} --state=${state}"
[ -n "${seed}" ] && EXTRA="${EXTRA} --seed=${seed}"
[ -n "${charge}" ] && EXTRA="${EXTRA} --charge=${charge}"
[ -n "${mult}" ] && EXTRA="${EXTRA} --mult=${mult}"
[ -n "${outdir}" ] && EXTRA="${EXTRA} --outdir=${outdir}"
[ "${all}" = "1" ] && EXTRA="${EXTRA} --all"
[ "${surrogate_xtb}" = "1" ] && EXTRA="${EXTRA} --surrogate_xtb"
[ -n "${calculator}" ] && EXTRA="${EXTRA} --calculator=${calculator}"
[ -n "${t12}" ] && EXTRA="${EXTRA} --t12=${t12}"
# search options
[ "${screen}" = "1" ] && EXTRA="${EXTRA} --screen"
[ -n "${seeds_per_rung}" ] && EXTRA="${EXTRA} --seeds_per_rung=${seeds_per_rung}"
[ -n "${explore_steps}" ] && EXTRA="${EXTRA} --explore_steps=${explore_steps}"
[ -n "${max_chains}" ] && EXTRA="${EXTRA} --max_chains=${max_chains}"
[ -n "${crossing_votes}" ] && EXTRA="${EXTRA} --crossing_votes=${crossing_votes}"
[ -n "${committor_iterations}" ] && EXTRA="${EXTRA} --committor_iterations=${committor_iterations}"
# ORCA may also be named once in the environment
[ -z "${orcapath}" ] && [ -n "${TREAX_ORCA}" ] && EXTRA="${EXTRA} --orcapath=${TREAX_ORCA}"

echo "file      : ${file}"
echo "stages    : ${stages}"
echo "back end  : ${calculator:-(meta.json, or xtb for explore)}"
echo "method    : ${method:-(back end default)} ${basis}"
echo "tau       : ${tau}"
echo "steps     : ${steps}"
echo "step scale: ${step_scale}"
echo ""

python "${CLI}" --stages="${stages}" --nworkers="${nworkers}" --file="${file}" --tau="${tau}" --steps="${steps}" --eq="${eq}" \
                --nprocs="${nprocs}" \
                --threshold="${threshold}" --step_scale="${step_scale}" \
                --accuracy="${accuracy}" --etemp="${etemp}" \
                ${EXTRA}
