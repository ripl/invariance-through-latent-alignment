from .sac import SAC
from .rad import RAD
from .curl import CURL
from .pad import PAD
from .soda import SODA
from .drq import DrQ
from .svea import SVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
