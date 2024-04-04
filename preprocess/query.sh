nohup python -u preprocess/query_claude.py --dataname lipo --begin 343 --diagram --structure > out_query_lipo.file 2>&1 &
tail -f out_query_lipo.file


nohup python -u preprocess/query_claude.py --dataname hiv --begin 0 --diagram --structure > out_query_hiv.file 2>&1 &
tail -f out_query_hiv.file
