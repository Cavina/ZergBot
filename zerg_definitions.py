from pysc2.lib import actions
from pysc2.lib import features


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SPAWNINGPOOL = actions.FUNCTIONS.Build_SpawningPool_screen.id
_TRAIN_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_TRAIN_ZERGLING = actions.FUNCTIONS.Train_Zergling_quick.id
_TRAIN_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_TRAIN_QUEEN = actions.FUNCTIONS.Train_Queen_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_EXTRACTOR = actions.FUNCTIONS.Build_Extractor_screen.id
_BUILD_HATCHERY = actions.FUNCTIONS.Build_Hatchery_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index


_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_ZERG_HATCHERY = 86
_ZERG_DRONE = 104
_ZERG_SPAWNINGPOOL = 89
_ZERG_LARVA = 151
_ZERG_OVERLORD = 106
_ZERG_EXTRACTOR = 88
_NEUTRAL_MINERAL_FIELD = 341
_NEUTRAL_VESPENE_GEYSER = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]



