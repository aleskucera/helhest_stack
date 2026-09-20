# Source inside the helhest Apptainer to run the closed-loop demo on dasenka.
#
# Mirrors ostrich-odinsim/odin_env.sh, with the three repos this demo needs added. The helhest
# checkout is a WORKTREE of the migration branch, not /local/kuceral4/projects/helhest_stack:
# that one is on main and carries uncommitted work, and the planner this demo drives exists only
# on study/tvf-migration. Create it once with
#
#   cd /local/kuceral4/projects/helhest_stack
#   git worktree add /local/kuceral4/projects/hs_tvf study/tvf-migration
#
# The venv is NOT activated -- inside the container its python is a host symlink that cannot see
# the container's own interpreter.
_SIM=/local/kuceral4/projects/ostrich-odinsim
_VENV_SITE=/local/kuceral4/projects/ostrich/.venv/lib/python3.12/site-packages

source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# hs_tvf FIRST so it can never be shadowed by the main-branch checkout on the venv's path.
export PYTHONPATH="/local/kuceral4/projects/hs_tvf/src\
:/local/kuceral4/projects/elevation_belief/src\
:/local/kuceral4/projects/terrain_value_field/src\
:$_SIM/src:$_SIM:$_SIM/third_party/newton:$_VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
