# Sourced by the hooks: sets $py to a Python 3.11+, or exits 1 (fail closed).
# $py is used unquoted so that "py -3" splits into two words.
py=""
for c in "python3" "python3.14" "python3.13" "python3.12" "python3.11" "py -3" "python"; do
    # command -v alone would find Windows' Store placeholder, which can't run.
    if $c -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then py="$c"; break; fi
done
if [ -z "$py" ]; then
    echo "No Python 3.11+ found; the secret scan can't run, so this is blocked."
    exit 1
fi
