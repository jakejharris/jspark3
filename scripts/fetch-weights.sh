#!/usr/bin/env bash
# Download the pinned weights (and the DFlash2 drafter unless DRAFTER=none) into $DATA and check every file's sha256.
#
#   scripts/fetch-weights.sh [--weights base|ablit] [--drafter dflash2|none] [--drafter-only] [--verify-only]
#                            [--dry-run]
#
# base   BASE_REPO at BASE_REV -> $DATA/base/weights, checked against manifests/inputs/base-weights.sha256.
#        Anonymous: no Hugging Face token is sent, even if you are logged in.
# ablit  also needs the base snapshot (an input to the conversion), plus the files of the refusal-removed source
#        at ABLIT_SOURCE_REV that manifests/inputs/ablit-source.sha256 lists -> $DATA/ablit/source. The source
#        repository is gated: accept its terms on its Hugging Face page with your own account, then
#        export HF_TOKEN=<your token> before running this. Then run scripts/convert-ablit.sh.
# The drafter goes to $DATA/drafter, checked against manifests/inputs/drafter.sha256. It is licensed
# CC BY-NC-ND 4.0: non-commercial use only; never share it or a modified copy.
#
# A checked directory gets a marker beside it (<dir>.verified) that later steps rely on.
# --drafter-only fetches only the drafter. With ablit selected, a fetch checks source access first,
# --drafter-only included. --verify-only needs no token and re-checks what is on disk without downloading.
# Downloads use HF_HUB_DISABLE_XET=1 unless you set it to 0 yourself (the Xet transfer path has failed on some hosts;
# plain HTTPS resumes reliably).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }
VERIFY_ONLY=0
DRAFTER_ONLY=0
parse_common "$@"
for arg in "${REST[@]}"; do
  case $arg in
    -h|--help) usage; exit 0 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --drafter-only) DRAFTER_ONLY=1 ;;
    *) die "unknown option $arg (--help)" ;;
  esac
done
load_cluster

# Check DIR against a sha256 list of paths relative to DIR; on success write DIR.verified (the list's digest).
verify_dir() {
  local dir=$1 list=$2 what=$3
  if [[ $DRY_RUN == 1 ]]; then run sha256sum -c --strict "${list#"$HERE"/}"; return; fi
  rm -f "$dir.verified"
  [[ -d $dir ]] || die "$what: $dir does not exist (run without --verify-only to download it)"
  info "$what: checking $(wc -l <"$list") files against ${list#"$HERE"/}"
  log_open
  local out rc=0
  out=$(cd "$dir" && sha256sum -c --strict "$list" 2>&1) || rc=$?
  printf '%s\n' "$out" >>"$LOG"
  if [[ $rc != 0 ]]; then
    local bad
    bad=$(printf '%s\n' "$out" | grep -v ': OK$' | sed 's/^sha256sum: //; s/: .*//' | head -20 | tr '\n' ' ')
    die "$what does not match ${list#"$HERE"/}. Files that differ or are missing: ${bad:-see the log}. Delete them and run this again to re-download."
  fi
  rm -f "$dir.verified"
  sha256_of "$list" >"$dir.verified"
  info "$what: every file matches"
}

# Hugging Face, or the mirror HF_ENDPOINT names (the hf CLI honours the same variable).
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
ENDPOINT=${ENDPOINT%/}

# Fail early if the download will not fit (sizes from the Hugging Face API for the pinned revision). Only the gated
# source's request carries HF_TOKEN: the base weights and the drafter are read anonymously, metadata included.
check_space() {  # repo rev dir what gated [list]
  local repo=$1 rev=$2 dir=$3 what=$4 gated=$5 list=${6:-}
  [[ $DRY_RUN == 1 ]] && { TOTAL=0; return 0; }
  mkdir -p "$dir"
  local out
  out=$(python3 - "$ENDPOINT" "$repo" "$rev" "$dir" "$gated" "$list" "$HERE/scripts/ablit/fetch-source.py" <<'PY'
import json, os, runpy, shutil, sys, urllib.request
endpoint, repo, rev, folder, gated, listed, helper = sys.argv[1:]
opener = urllib.request.build_opener(runpy.run_path(helper)["SafeRedirect"](endpoint))
wanted = {line.split("  ", 1)[1] for line in open(listed).read().splitlines() if line} if listed else None
headers = {}
if gated == "1":
    headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
try:
    req = urllib.request.Request(f"{endpoint}/api/models/{repo}/tree/{rev}?recursive=true", headers=headers)
    with opener.open(req, timeout=30) as r:
        total = sum(item.get("size", 0) for item in json.load(r)
                    if item.get("type") == "file" and (wanted is None or item.get("path") in wanted))
except Exception:  # noqa: BLE001 - the download reports real errors
    print("unknown 0")
    sys.exit(0)
have = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(folder) for f in fs)
free = shutil.disk_usage(folder).free
print("ok" if max(total - have, 0) <= free else "short", total, max(total - have, 0), free)
PY
)
  local state total need free
  read -r state total need free <<<"$out"
  TOTAL=${total:-0}
  case $state in
    unknown) warn "$what: could not read sizes from Hugging Face; skipping the space check" ;;
    short) die "$what needs $(bc_div "$need") GB more but the DATA disk has $(bc_div "$free") GB free" ;;
    ok) info "$what: $(bc_div "$total") GB in total, $(bc_div "$need") GB still to download, $(bc_div "$free") GB free" ;;
  esac
}

download() {  # repo rev dir what [gated [list]]: with a list, only the files it names (the gated source)
  local repo=$1 rev=$2 dir=$3 what=$4 gated=${5:-0} list=${6:-} files=()
  [[ -n $list ]] && mapfile -t files < <(sed -n 's/^[0-9a-f]\{64\}  //p' "$list")
  check_space "$repo" "$rev" "$dir" "$what" "$gated" "$list"
  info "$what: downloading $repo at ${rev:0:12} (progress every 30 s)"
  # Pin the wrapper's endpoint in every CLI process. In the pinned client, staging overrides HF_ENDPOINT.
  if [[ $gated == 1 ]]; then
    # The user's own token, from the environment; never printed (the log records the command, not the environment).
    run_watch "$what" "$dir" "$TOTAL" env -u HUGGINGFACE_CO_STAGING HF_ENDPOINT="$ENDPOINT" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" hf download "$repo" "${files[@]}" --revision "$rev" --local-dir "$dir" \
      || die "$what: download failed. If the log shows 401 or 403, accept the source's terms on its Hugging Face page and export HF_TOKEN; otherwise run this again (it resumes)"
  else
    # Public inputs are fetched the way a stranger gets them: no stored or environment token is sent.
    run_watch "$what" "$dir" "$TOTAL" env -u HF_TOKEN -u HUGGINGFACE_CO_STAGING HF_ENDPOINT="$ENDPOINT" \
      HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      hf download "$repo" --revision "$rev" --local-dir "$dir" \
      || die "$what: download failed; run this again (it resumes)"
  fi
}

# Refuse plainly before a gated download when the token is missing or rejected, or the terms are not accepted yet.
# The probe reads one byte of the first weight shard the list names: small files of a gated repository can be public.
gate_check() {  # repo rev list
  [[ $DRY_RUN == 1 ]] && return 0
  local code probe
  probe=$(awk '{ f = substr($0, 67) } NR == 1 { first = f } f ~ /\.safetensors$/ { print f; found = 1; exit }
               END { if (!found) print first }' "$3")
  code=$(python3 - "$ENDPOINT" "$1" "$2" "$probe" "$HERE/scripts/ablit/fetch-source.py" <<'PY'
import os, runpy, sys, urllib.error, urllib.parse, urllib.request
endpoint, repo, rev, probe, helper = sys.argv[1:]
opener = urllib.request.build_opener(runpy.run_path(helper)["SafeRedirect"](endpoint))
req = urllib.request.Request(f"{endpoint}/{repo}/resolve/{rev}/{urllib.parse.quote(probe)}",
                             headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"], "Range": "bytes=0-0"})
try:
    opener.open(req, timeout=30).close()
    print(200)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:  # noqa: BLE001
    print(0)
PY
)
  case $code in
    200|206) info "your Hugging Face account has access to the gated source" ;;
    401) die "Hugging Face did not accept HF_TOKEN: create a read token in your Hugging Face settings, export HF_TOKEN=<your token>, and run this again" ;;
    403) die "your Hugging Face account has not accepted the source's terms yet: open https://huggingface.co/$1, accept them, and run this again" ;;
    0) warn "could not reach Hugging Face to check access; trying the download anyway" ;;
    *) die "Hugging Face answered HTTP $code for the gated source; check https://huggingface.co/$1 and run this again" ;;
  esac
}

if [[ $WEIGHTS == ablit && $VERIFY_ONLY == 0 ]]; then
  cat "$HERE/config/ablit-notice.txt"
  ablit_ready || die "$ABLIT_NOT_READY"
  [[ -n ${HF_TOKEN:-} ]] || die "the refusal-removed source is gated: accept its terms on its Hugging Face page with your account, then export HF_TOKEN=<your token> and run this again"
  gate_check "$ABLIT_SOURCE_REPO" "$ABLIT_SOURCE_REV" "$HERE/manifests/inputs/ablit-source.sha256"
fi
if [[ $VERIFY_ONLY == 0 && $DRY_RUN == 0 ]]; then
  hf_install="python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install \"huggingface_hub==$HF_HUB_VERSION\", then export PATH=\"\$HOME/hf-cli/bin:\$PATH\" (INSTALL.md, 'What you need')"
  command -v hf >/dev/null || die "no 'hf' command: install the pinned Hugging Face CLI in a virtual environment: $hf_install"
  # Read hf's venv metadata as data; disable Python startup hooks before the helper starts.
  hf_version=$(python3 -I -S "$HERE/scripts/hf-version.py" 2>/dev/null) || hf_version=''
  [[ $hf_version == "$HF_HUB_VERSION" ]] \
    || die "the Hugging Face CLI is version ${hf_version:-unknown}; this release pins HF_HUB_VERSION=$HF_HUB_VERSION. Install it in a virtual environment: $hf_install"
fi
if [[ $DRAFTER_ONLY == 1 ]]; then
  [[ $DRAFTER == dflash2 ]] || die "--drafter-only with --drafter none fetches nothing"
  WEIGHTS=none
fi

if [[ $WEIGHTS != none ]]; then
  [[ $VERIFY_ONLY == 1 ]] || download "$BASE_REPO" "$BASE_REV" "$(snapshot_dir base)" "base weights"
  verify_dir "$(snapshot_dir base)" "$HERE/manifests/inputs/base-weights.sha256" "base weights"
fi

if [[ $WEIGHTS == ablit ]]; then
  [[ $VERIFY_ONLY == 0 ]] || cat "$HERE/config/ablit-notice.txt"
  if [[ $VERIFY_ONLY == 0 ]]; then
    download "$ABLIT_SOURCE_REPO" "$ABLIT_SOURCE_REV" "$(variant_dir ablit)/source" "refusal-removed source" 1 \
      "$HERE/manifests/inputs/ablit-source.sha256"
  fi
  verify_dir "$(variant_dir ablit)/source" "$HERE/manifests/inputs/ablit-source.sha256" "refusal-removed source"
fi

if [[ $DRAFTER == dflash2 ]]; then
  echo "DFlash2 is licensed CC BY-NC-ND 4.0 for research and evaluation: non-commercial use only."
  echo "It is downloaded unmodified from its publisher; never share it or a modified copy. DRAFTER=none avoids it."
  if [[ $VERIFY_ONLY == 0 ]]; then
    download "$DRAFTER_REPO" "$DRAFTER_REV" "$(drafter_dir)" "drafter"
  fi
  verify_dir "$(drafter_dir)" "$HERE/manifests/inputs/drafter.sha256" "drafter"
fi
info "done"
