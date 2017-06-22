#!/usr/bin/env bash

set -e

fullname="$1"
fname=${fullname%.*}

python pyast64.py $1 > $fname.s
as $fname.s -o $fname.o
ld $fname.o -e _main -o $fname.exe -w
./$fname.exe
echo
