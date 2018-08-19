#! /bin/bash
#
# Face manager 0.1
# Run micro benchmark suite several times and create CSV file with combined results
#
# Copyright (c) 2018 David Snowdon. All rights reserved.
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

RESULT_FILE=$2
TMP_FILE=$1

read -d '' awkScript << 'EOF'
function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
function trim(s) { return rtrim(ltrim(s)); }
{split($0, a, ":"); key=trim(a[1]); value=trim(a[2]); if (key in results) { results[key] = (results[key] "," value) } else { results[key] = value } }
END {
    for (v in results)
        print v, ",", results[v]
}
EOF

grep 'End:' < $TMP_FILE  \
    | sed 's/End: //g' \
    | sed 's/ seconds//g' \
    | awk "$awkScript" \
    | sort > ${RESULT_FILE}

echo "Results in ${RESULT_FILE}"

