# Sourced first by tests/run.sh and tests/check-leaks.sh. Re-runs the calling script with only PATH, HOME, LANG,
# LC_*, TERM and TMPDIR from the caller's environment (bash adds PWD, SHLVL and _). Settings meant for a real install,
# such as CLUSTER_ENV, DATA, DOCKER_HOST, a proxy or a PYTHON* variable, then cannot reach the checks or the scripts
# they run.
# shellcheck shell=bash
keep=() other=0
while IFS= read -r -d '' entry; do
  case ${entry%%=*} in
    PATH|HOME|LANG|LC_*|TERM|TMPDIR) keep+=("$entry") ;;
    PWD|OLDPWD|SHLVL|_) ;;
    *) other=1 ;;
  esac
done < <(env -0)
if [[ $other == 1 ]]; then
  [[ ${1:-} != --scrubbed ]] || { echo "${0##*/}: could not clear the environment" >&2; exit 2; }
  exec env -i "${keep[@]}" "$BASH" "$0" --scrubbed "$@"
fi
[[ ${1:-} != --scrubbed ]] || shift
unset keep other entry
