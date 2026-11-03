#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

CLI="${SCRIPT_DIR}/run_muller.py"

# mode: search (default, the staged search) or scan (the tau x Ninter scan)
[ -z "${mode}" ] && mode='search'
[ -z "${nworkers}" ] && nworkers='4'

if [ "${mode}" = "search" ]; then
    SEARCH=""
    [ -n "${outdir}" ] && SEARCH="${SEARCH} --outdir=${outdir}"
    [ -n "${seed}" ] && SEARCH="${SEARCH} --seed=${seed}"
    [ -n "${mass}" ] && SEARCH="${SEARCH} --mass=${mass}"
    [ -n "${seeds_per_rung}" ] && SEARCH="${SEARCH} --seeds_per_rung=${seeds_per_rung}"
    [ -n "${explore_steps}" ] && SEARCH="${SEARCH} --explore_steps=${explore_steps}"
    [ -n "${max_chains}" ] && SEARCH="${SEARCH} --max_chains=${max_chains}"
    [ "${committor_stage}" = "1" ] && SEARCH="${SEARCH} --committor"
    [ -n "${committor_iterations}" ] && SEARCH="${SEARCH} --committor_iterations=${committor_iterations}"
    echo "mode       : search"
    echo "outdir     : ${outdir:-reaction/output/muller/search}"
    echo ""
    exec python "${CLI}" --nworkers="${nworkers}" ${SEARCH}
fi

# Runtime settings of the scan (override via env vars); empty ones keep the CLI default
[ -z "${outdir}" ] && outdir='.'
[ -z "${num_trials}" ] && num_trials='10'
[ -z "${n_max}" ] && n_max='10000'
[ -z "${threshold}" ] && threshold='0.5'
[ -z "${n_compress}" ] && n_compress='10'
[ -z "${seed}" ] && seed=''
[ -z "${mass}" ] && mass=''
[ -z "${tau}" ] && tau=''
[ -z "${kt}" ] && kt=''
[ -z "${ninter}" ] && ninter=''

echo "outdir     : ${outdir}"
echo "num_trials : ${num_trials}"
echo "n_max      : ${n_max}"
echo "threshold  : ${threshold}"
echo ""

EXTRA=""
[ -n "${seed}" ] && EXTRA="${EXTRA} --seed=${seed}"
[ -n "${mass}" ] && EXTRA="${EXTRA} --mass=${mass}"
# space-separated lists; tau in fs, kt in the energy unit of the potential
[ -n "${tau}" ] && EXTRA="${EXTRA} --tau ${tau}"
[ -n "${kt}" ] && EXTRA="${EXTRA} --kt ${kt}"
[ -n "${ninter}" ] && EXTRA="${EXTRA} --ninter ${ninter}"

python "${CLI}" --scan --outdir="${outdir}" --num_trials="${num_trials}" \
                --n_max="${n_max}" --threshold="${threshold}" \
                --n_compress="${n_compress}" ${EXTRA}
