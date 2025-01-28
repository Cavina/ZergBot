from pysc2.lib import actions
from pysc2.lib import features


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SPAWNING_POOL = actions.FUNCTIONS.Build_SpawningPool_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index


_PLAYER_SELF = 1


