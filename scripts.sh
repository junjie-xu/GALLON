nohup python trainMoleBert.py > ./logs/mole_bace.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_bbbp.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_clintox.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_hiv.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_esol.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_freesolv.log 2>&1 &
nohup python trainMoleBert.py > ./logs/mole_lipo.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_bace_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_bace_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_bace_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_bace_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_bbbp_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_bbbp_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_bbbp_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_bbbp_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_clintox_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_clintox_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_clintox_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_clintox_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_hiv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_hiv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_hiv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_hiv_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_esol_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_esol_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_esol_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_esol_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_freesolv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_freesolv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_freesolv_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_freesolv_scaffold.log 2>&1 &

nohup python -u trainGNN.py > ./logs/chebnet_lipo_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gat_lipo_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/sage_lipo_scaffold.log 2>&1 &
nohup python -u trainGNN.py > ./logs/gin_lipo_scaffold.log 2>&1 &


nohup python -u trainMoleBert.py > ./logs/mole_hiv_scaffold.log 2>&1 &