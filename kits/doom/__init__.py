from .vizdoomenv import VizdoomEnv
from .vizdoombasic import VizdoomBasic
from .vizdoomcorridor import VizdoomCorridor
from .vizdoomdefendcenter import VizdoomDefendCenter
from .vizdoomdefendline import VizdoomDefendLine
from .vizdoomhealthgathering import VizdoomHealthGathering
from .vizdoommywayhome import VizdoomMyWayHome
from .vizdoompredictposition import VizdoomPredictPosition
from .vizdoomtakecover import VizdoomTakeCover
from .vizdoomdeathmatch import VizdoomDeathmatch
from .vizdoomhealthgatheringsupreme import VizdoomHealthGatheringSupreme

from gym.envs.registration import register

register(
    id='VizdoomBasic-v0',
    entry_point='kits.doom:VizdoomBasic'
)

register(
    id='VizdoomCorridor-v0',
    entry_point='kits.doom:VizdoomCorridor'
)

register(
    id='VizdoomDefendCenter-v0',
    entry_point='kits.doom:VizdoomDefendCenter'
)

register(
    id='VizdoomDefendLine-v0',
    entry_point='kits.doom:VizdoomDefendLine'
)

register(
    id='VizdoomHealthGathering-v0',
    entry_point='kits.doom:VizdoomHealthGathering'
)

register(
    id='VizdoomMyWayHome-v0',
    entry_point='kits.doom:VizdoomMyWayHome'
)

register(
    id='VizdoomPredictPosition-v0',
    entry_point='kits.doom:VizdoomPredictPosition'
)

register(
    id='VizdoomTakeCover-v0',
    entry_point='kits.doom:VizdoomTakeCover'
)

register(
    id='VizdoomDeathmatch-v0',
    entry_point='kits.doom:VizdoomDeathmatch'
)

register(
    id='VizdoomHealthGatheringSupreme-v0',
    entry_point='kits.doom:VizdoomHealthGatheringSupreme'
)