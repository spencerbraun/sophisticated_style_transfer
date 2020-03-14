#!/usr/bin/env bash
# AUTHOR:   fuzx
# FILE:     run.sh
# ROLE:     TODO (some explanation)
# CREATED:  2017-06-13 20:56:12
# MODIFIED: 2017-12-06 10:39:36
num=7000
head -n $num q.txt > q_train.txt
head -n $num q.txt > r_train.txt
head -n $num s.txt > s_train.txt

tail -n 500 q.txt > q_tmp.txt
tail -n 500 q.txt > r_tmp.txt
tail -n 500 s.txt > s_tmp.txt

head -n 1000 q_tmp.txt > q_val.txt
head -n 1000 r_tmp.txt > r_val.txt
head -n 1000 s_tmp.txt > s_val.txt

tail -n 500 q_tmp.txt > q_test.txt
tail -n 500 r_tmp.txt > r_test.txt
tail -n 500 s_tmp.txt > s_text.txt


